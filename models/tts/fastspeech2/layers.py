import logging
import numpy as np
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tts.fastspeech2.sublayers import ConvNorm, LayerNorm, SinusoidalPositionalEmbedding
from models.tts.fastspeech2.blocks.utils import LinearNorm
from models.tts.fastspeech2.function import b_mas, get_phoneme_level, get_mask_from_lengths, pad_list, reparameterize
from espnet2.tts.gst.style_encoder import ReferenceEncoder
from espnet.nets.pytorch_backend.nets_utils import pad_list


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self,
                 n_channels: int,
                 hidden_dim: int,
                 config: Dict,
                 stats: Dict,
                 ) -> None:
        super(VarianceAdaptor, self).__init__()
        
        # intialize hparams
        self.stats       = stats
        self.hidden_size = hidden_dim
        self.n_channels  = n_channels
        self.conf        = config

        # initialize duration modelling
        self.learn_alignment = self.conf["learn_alignment"]
        self.aligner, self.binarization_start_steps = None, None
        if self.learn_alignment is True:
            self.aligner = AlignmentEncoder(
                n_mel_channels  = self.n_channels,
                n_att_channels  = self.n_channels,
                n_text_channels = self.hidden_size,
                temperature     = self.conf["duration_modelling"]["aligner_temperature"]
            )
            self.binarization_start_steps = self.conf["duration_modelling"]["binarization_start_steps"]
        self.duration_predictor = DurationPredictor(
            self.hidden_size,
            n_chans      = self.n_channels,
            n_layers     = self.conf["variance_predictor"]["dur_predictor_layers"],
            dropout_rate = self.conf["variance_predictor"]["dropout"], 
            padding      = self.conf["variance_predictor"]["ffn_padding"],
            kernel_size  = self.conf["variance_predictor"]["dur_predictor_kernel"]
        )

        self.use_gaussian = self.conf["duration_modelling"]["use_gaussian"]
        if self.use_gaussian is True:
            self.length_regulator = GaussianUpsampling()
        else:
            self.length_regulator = LengthRegulator()

        # initilize pitch modelling
        self.pitch_feature_level = self.conf["variance_embedding"]["pitch_feature"]
        assert self.pitch_feature_level in ["frame_level", "phoneme_level"]
        self.pitch_predictor = VariancePredictor(
            idim         = self.hidden_size,
            n_chans      = self.conf["variance_predictor"]["filter_size"],
            n_layers     = self.conf["variance_predictor"]["pit_predictor_layers"],
            dropout_rate = self.conf["variance_predictor"]["dropout"],
            odim         = 1,
            padding      = self.conf["variance_predictor"]["ffn_padding"], 
            kernel_size  = self.conf["variance_predictor"]["pit_predictor_kernel"]
        )
        self.pitch_embedding = nn.Embedding(
            num_embeddings = self.conf["variance_embedding"]["n_bins"],
            embedding_dim  = self.hidden_size
        )

        self.pitch_quantization = self.conf["variance_embedding"]["pitch_quantization"]
        assert self.pitch_quantization in ["linear", "log"]
        if self.pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                data = torch.exp(torch.linspace(
                    np.log(self.stats["pitch"]["min"]), 
                    np.log(self.stats["pitch"]["max"]), 
                    self.conf["variance_embedding"]["n_bins"] - 1
                )),
                requires_grad = False
            )
        else:
            self.pitch_bins = nn.Parameter(
                data = torch.linspace(
                    self.stats["pitch"]["min"], 
                    self.stats["pitch"]["max"], 
                    self.conf["variance_embedding"]["n_bins"] - 1
                ),
                requires_grad = False
            )

        # initilize energy modelling
        self.energy_feature_level = self.conf["variance_embedding"]["energy_feature"]
        assert self.energy_feature_level in ["frame_level", "phoneme_level"]
        self.energy_predictor = VariancePredictor(
            idim         = self.hidden_size,
            n_chans      = self.conf["variance_predictor"]["filter_size"],
            n_layers     = self.conf["variance_predictor"]["ener_predictor_layers"],
            dropout_rate = self.conf["variance_predictor"]["dropout"],
            odim         = 1,
            padding      = self.conf["variance_predictor"]["ffn_padding"], 
            kernel_size  = self.conf["variance_predictor"]["ener_predictor_kernel"]
        )
        self.energy_embedding = nn.Embedding(
            num_embeddings = self.conf["variance_embedding"]["n_bins"],
            embedding_dim  = self.hidden_size
        )
        
        self.energy_quantization = self.conf["variance_embedding"]["energy_quantization"]
        assert self.energy_quantization in ["linear", "log"]
        if self.energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                data = torch.exp(torch.linspace(
                    np.log(self.stats["energy"]["min"]),
                    np.log(self.stats["energy"]["max"]), 
                    self.conf["variance_embedding"]["n_bins"] - 1
                )),
                requires_grad = False
            )
        else:
            self.energy_bins = nn.Parameter(
                data = torch.linspace(
                    self.stats["energy"]["min"], 
                    self.stats["energy"]["max"], 
                    self.conf["variance_embedding"]["n_bins"] - 1
                ),
                requires_grad = False
            )

    def binarize_attention_parallel(self, attn: torch.Tensor, in_lens: torch.Tensor, out_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """For training purposes only. Binarizes attention with MAS.
        These will no longer recieve a gradient.
        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = b_mas(attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1)

        return torch.from_numpy(attn_out).to(attn.device)

    def get_pitch_embedding(self, x: torch.Tensor, target: torch.Tensor, control: float) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = self.pitch_predictor(x, squeeze=True)
        pitch = target if target is not None else prediction * control        
        pitch = torch.bucketize(pitch, self.pitch_bins)

        embedding = self.pitch_embedding(pitch)

        return prediction, embedding

    def get_energy_embedding(self, x: torch.Tensor, target: torch.Tensor, control: float) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = self.energy_predictor(x, squeeze=True)
        energy = target if target is not None else prediction * control
        energy = torch.bucketize(energy, self.energy_bins)
        
        embedding = self.energy_embedding(energy)

        return prediction, embedding

    def forward(
        self, 
        x: torch.Tensor,
        txt_emb: torch.Tensor,
        txt_lens: torch.Tensor, 
        txt_mask: torch.Tensor,
        max_txt_len: int,
        spk_emb: torch.Tensor, 
        mel: torch.Tensor=None, 
        mel_lens: torch.Tensor=None,
        mel_mask: torch.Tensor=None, 
        max_mel_len: int=None,
        pitch_target: torch.Tensor=None, 
        energy_target: torch.Tensor=None, 
        duration_target: torch.Tensor=None,
        attn_prior: torch.Tensor=None,
        p_control: float=1.0, 
        e_control: float=1.0, 
        d_control: float=1.0,
        step: int=None
    ):
        if spk_emb is not None:
            x = x + spk_emb.unsqueeze(1).expand(-1, x.shape[1], -1) 

        # duration modelling
        log_duration_prediction = self.duration_predictor(x, txt_mask)
        attn_out = None
        if self.learn_alignment is True and attn_prior is not None:
            attn_soft, attn_logprob = self.aligner(
                mel.transpose(1, 2),
                txt_emb.transpose(1, 2),
                txt_mask.unsqueeze(-1),
                attn_prior,
                spk_emb
            )
            attn_hard = self.binarize_attention_parallel(attn_soft, txt_lens, mel_lens)
            attn_hard_dur = attn_hard.sum(2)[:, 0, :]
            attn_out = (attn_soft, attn_hard, attn_hard_dur, attn_logprob)
            duration_rounded = attn_hard_dur.long()
        elif self.learn_alignment is False and duration_target is not None:
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0
            ).long()
            mel_lens = torch.sum(duration_rounded, dim=1)
            mel_mask = get_mask_from_lengths(mel_lens)

        # phoneme level feature
        pitch_embedding, energy_embedding = None, None
        if self.pitch_feature_level == "phoneme_level":
            if pitch_target is not None:
                pitch_target = get_phoneme_level(pitch_target, txt_lens, duration_rounded, max_txt_len)
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, p_control)
        if self.energy_feature_level == "phoneme_level":
            if energy_target is not None:
                energy_target = get_phoneme_level(energy_target, txt_lens, duration_rounded, max_txt_len)
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, e_control)
        
        if pitch_embedding is not None: x = x + pitch_embedding
        if energy_embedding is not None: x = x + energy_embedding

        # expand duration (text lens -> melspec lens)
        if attn_prior is not None and step < self.binarization_start_steps:
            x = torch.bmm(attn_soft.squeeze(1), x)
        else:
            if self.use_gaussian:
                x = self.length_regulator(x, duration_rounded, ~mel_mask, ~txt_mask)
            else:
                x = self.length_regulator(x, duration_rounded)

        # frame level feature
        pitch_embedding, energy_embedding = None, None
        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, p_control)
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, e_control)

        if pitch_embedding is not None: x = x + pitch_embedding
        if energy_embedding is not None: x = x + energy_embedding

        return (
            x, 
            log_duration_prediction,  
            pitch_prediction, 
            energy_prediction, 
            mel_lens, 
            mel_mask, 
            attn_out
        ), (
            duration_rounded,
            pitch_target,
            energy_target
        )


class AlignmentEncoder(nn.Module):
    """ Alignment Encoder for Unsupervised Duration Modeling """

    def __init__(self, 
                 n_mel_channels: int,
                 n_att_channels: int,
                 n_text_channels: int,
                 temperature: float,
                 n_spk_channels: int=None):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=3)
        self.log_softmax = nn.LogSoftmax(dim=3)

        self.key_proj = nn.Sequential(
            ConvNorm(
                n_text_channels,
                n_text_channels * 2,
                kernel_size=3,
                bias=True,
                w_init_gain='relu'
            ),
            nn.ReLU(),
            ConvNorm(
                n_text_channels * 2,
                n_att_channels,
                kernel_size=1,
                bias=True,
            ),
        )

        self.query_proj = nn.Sequential(
            ConvNorm(
                n_mel_channels,
                n_mel_channels * 2,
                kernel_size=3,
                bias=True,
                w_init_gain='relu',
            ),
            nn.ReLU(),
            ConvNorm(
                n_mel_channels * 2,
                n_mel_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU(),
            ConvNorm(
                n_mel_channels,
                n_att_channels,
                kernel_size=1,
                bias=True,
            ),
        )
        if n_spk_channels is None:
            n_spk_channels = n_text_channels
        self.key_spk_proj   = LinearNorm(n_spk_channels, n_text_channels)
        self.query_spk_proj = LinearNorm(n_spk_channels, n_mel_channels)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor=None, attn_prior: torch.Tensor=None, speaker_embed: torch.Tensor=None):
        """Forward pass of the aligner encoder.
        Args:
            queries (torch.tensor): B x C x T1 tensor (probably going to be mel data).
            keys (torch.tensor): B x C2 x T2 tensor (text data).
            mask (torch.tensor): uint8 binary mask for variable length entries (should be in the T2 domain).
            attn_prior (torch.tensor): prior for attention matrix.
            speaker_embed (torch.tensor): B x C tnesor of speaker embedding for multi-speaker scheme.
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask. Final dim T2 should sum to 1.
            attn_logprob (torch.tensor): B x 1 x T1 x T2 log-prob attention mask.
        """

        if speaker_embed is not None:
            keys = keys + self.key_spk_proj(speaker_embed.unsqueeze(1).expand(
                -1, keys.shape[-1], -1
            )).transpose(1, 2)
            queries = queries + self.query_spk_proj(speaker_embed.unsqueeze(1).expand(
                -1, queries.shape[-1], -1
            )).transpose(1, 2)
        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2
        queries_enc = self.query_proj(queries)

        # Simplistic Gaussian Isotopic Attention
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2  # B x n_attn_dims x T1 x T2
        attn = -self.temperature * attn.sum(1, keepdim=True)

        if attn_prior is not None:
            # print(f"AlignmentEncoder \t| mel: {queries.shape} phone: {keys.shape} mask: {mask.shape} attn: {attn.shape} attn_prior: {attn_prior.shape}")
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None] + 1e-8)
            # print(f"AlignmentEncoder \t| After prior sum attn: {attn.shape}")
        attn_logprob = attn.clone()

        if mask is not None:
            attn.data.masked_fill_(mask.permute(0, 2, 1).unsqueeze(2), -float("inf"))

        attn = self.softmax(attn)  # softmax along T2
        return attn, attn_logprob


class DurationPredictor(nn.Module):
    """Duration Predictor module.
    This is a module of duration Predictor described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration Predictor predicts a duration of each frame in log domain from the hidden embeddings of encoder.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    Note:
        The outputs are calculated in log domain.
    """

    def __init__(self, idim: int, n_layers: int=2, n_chans: int=384, kernel_size: int=3, dropout_rate: float=0.1, offset: float=1.0, padding="SAME"):
        """Initilize duration Predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.
        """
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [nn.Sequential(
                nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                                       if padding == "SAME"
                                       else (kernel_size - 1, 0), 0),
                nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                nn.Dropout(dropout_rate)
            )]
        self.linear = nn.Linear(n_chans, 1)

    def forward(self, xs: torch.Tensor, x_masks: torch.Tensor=None):
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
            if x_masks is not None:
                xs = xs * (1 - x_masks.float())[:, None, :]

        xs = self.linear(xs.transpose(1, -1))  # [B, T, C]
        xs = xs * (1 - x_masks.float())[:, :, None]  # (B, T, C)

        return xs.squeeze(-1) # (B, Tmax)


class LengthRegulator(nn.Module):
    """Length regulator module for feed-forward Transformer.

    This is a module of length regulator described in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The length regulator expands char or
    phoneme-level embedding features to frame-level by repeating each
    feature based on the corresponding predicted durations.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, pad_value: float=0.0):
        """Initilize length regulator module.

        Args:
            pad_value (float, optional): Value used for padding.

        """
        super().__init__()
        self.pad_value = pad_value

    def forward(self, xs: torch.Tensor, ds: torch.LongTensor, alpha: float=1.0) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of sequences of char or phoneme embeddings (B, Tmax, D).
            ds (LongTensor): Batch of durations of each frame (B, T).
            alpha (float, optional): Alpha value to control speed of speech.

        Returns:
            Tensor: replicated input tensor based on durations (B, T*, D).

        """
        if alpha != 1.0:
            assert alpha > 0
            ds = torch.round(ds.float() * alpha).long()

        if ds.sum() == 0:
            logging.warning(
                "predicted durations includes all 0 sequences. "
                "fill the first element with 1."
            )
            # NOTE(kan-bayashi): This case must not be happened in teacher forcing.
            #   It will be happened in inference with a bad duration Predictor.
            #   So we do not need to care the padded sequence case here.
            ds[ds.sum(dim=1).eq(0)] = 1

        repeat = [torch.repeat_interleave(x, d, dim=0) for x, d in zip(xs, ds)]
        
        return pad_list(repeat, self.pad_value)


class GaussianUpsampling(nn.Module):
    """Gaussian upsampling with fixed temperature as in:

    https://arxiv.org/abs/2010.04301

    """

    def __init__(self, delta: float=0.1):
        super().__init__()
        self.delta = delta

    def forward(self, hs: torch.Tensor, ds: torch.Tensor, h_masks: torch.Tensor=None, d_masks: torch.Tensor=None):
        """Upsample hidden states according to durations.

        Args:
            hs (Tensor): Batched hidden state to be expanded (B, T_text, adim).
            ds (Tensor): Batched token duration (B, T_text).
            h_masks (Tensor): Mask tensor (B, T_feats).
            d_masks (Tensor): Mask tensor (B, T_text).

        Returns:
            Tensor: Expanded hidden state (B, T_feat, adim).

        """
        B = ds.size(0)
        device = ds.device

        if ds.sum() == 0:
            logging.warning(
                "predicted durations includes all 0 sequences. "
                "fill the first element with 1."
            )
            # NOTE(kan-bayashi): This case must not be happened in teacher forcing.
            #   It will be happened in inference with a bad duration Predictor.
            #   So we do not need to care the padded sequence case here.
            ds[ds.sum(dim=1).eq(0)] = 1

        if h_masks is None:
            T_feats = ds.sum().int()
        else:
            T_feats = h_masks.size(-1)
        t = torch.arange(0, T_feats).unsqueeze(0).repeat(B, 1).to(device).float()
        if h_masks is not None:
            t = t * h_masks.float()

        c = ds.cumsum(dim=-1) - ds / 2
        energy = -1 * self.delta * (t.unsqueeze(-1) - c.unsqueeze(1)) ** 2
        if d_masks is not None:
            energy = energy.masked_fill(
                ~(d_masks.unsqueeze(1).repeat(1, T_feats, 1)), -float("inf")
            )

        p_attn = torch.softmax(energy, dim=2)  # (B, T_feats, T_text)
        hs = torch.matmul(p_attn, hs)

        return hs


class VariancePredictor(nn.Module):
    """ Pitch & Energy Predictor """

    def __init__(self, idim: int, n_layers: int=5, n_chans: int=384, odim: int=2, kernel_size: int=5, dropout_rate: float=0.1, padding="SAME"):
        """Initilize pitch Predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super(VariancePredictor, self).__init__()
        self.conv = nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [nn.Sequential(
                nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                                       if padding == "SAME"
                                       else (kernel_size - 1, 0), 0),
                nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                nn.Dropout(dropout_rate)
            )]
        self.linear = nn.Linear(n_chans, odim)
        self.embed_positions = SinusoidalPositionalEmbedding(idim, 0, init_size=4096)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))

    def forward(self, xs: torch.Tensor, squeeze: bool=False):
        """

        :param xs: [B, T, H]
        :return: [B, T, H]
        """
        positions = self.pos_embed_alpha * self.embed_positions(xs[..., 0])
        xs = xs + positions
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
        # NOTE: calculate in log domain
        xs = self.linear(xs.transpose(1, -1))  # (B, Tmax, H)

        return xs.squeeze(-1) if squeeze else xs


class Postnet(nn.Module):
    """
    Post-net:
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, n_channels: int, config: Dict) -> None:
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.conf = config
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(in_channels  = n_channels,
                         out_channels = self.conf["embedding_dim"],
                         kernel_size  = self.conf["kernel_size"],
                         stride       = 1,
                         padding      = int((self.conf["kernel_size"] - 1) / 2),
                         dilation     = 1,
                         w_init_gain  = "tanh"),
                nn.BatchNorm1d(num_features = self.conf["embedding_dim"]))
        )

        for _ in range(1, self.conf["conv_layers"] - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(in_channels  = self.conf["embedding_dim"],
                             out_channels = self.conf["embedding_dim"],
                             kernel_size  = self.conf["kernel_size"], stride=1,
                             padding      = int((self.conf["kernel_size"] - 1) / 2),
                             dilation     = 1, 
                             w_init_gain  = "tanh"),
                    nn.BatchNorm1d(num_features = self.conf["embedding_dim"]))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(in_channels  = self.conf["embedding_dim"],
                         out_channels = n_channels,
                         kernel_size  = self.conf["kernel_size"],
                         stride       = 1,
                         padding      = int((self.conf["kernel_size"] - 1) / 2),
                         dilation     = 1,
                         w_init_gain = "linear"),
                nn.BatchNorm1d(num_features = n_channels))
        )

    def forward(self, x: torch.Tensor):
        x = x.contiguous().transpose(1, 2)
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        x = x.contiguous().transpose(1, 2)

        return x


class AccentEncoderNetwork(nn.Module):
    def __init__(self, idim: int, n_speakers: int, n_accents: int, ref_hparams: Dict, cvae_hparams: Dict):
        super().__init__()
                
        # x_dim, y_dim=10, z_dim, n_classes
        self.encoder = ReferenceEncoder(
            idim = idim,
            **ref_hparams
        )
        
        # self.acc_vae = CVAEnet(model_config["reference_encoder"]["ref_enc_gru_size"],model_config["accent_encoder"]["y_dim"],model_config["accent_encoder"]["z_dim"])
        # self.spk_vae = CVAEnet(model_config["reference_encoder"]["ref_enc_gru_size"],model_config["speaker_encoder"]["y_dim"],model_config["speaker_encoder"]["z_dim"])
        self.cvae    = CVAEnet(
            n_classes_acc = n_accents,
            n_classes_spk = n_speakers,
            **cvae_hparams
        )

    def forward(self, inputs: torch.Tensor, acc_labels: torch.Tensor, spk_labels: torch.Tensor):
        ref_embs = self.encoder(inputs)

        # (z_acc, y_acc, (mu_acc, var_acc)) = self.acc_vae(enc_out,acc_labels)
        # (z_spk, y_spk, (mu_spk, var_spk)) = self.spk_vae(enc_out,spk_labels)
        (
            z_acc, 
            y_acc, 
            z_spk, 
            y_spk,
            accent_prediction
        ) = self.cvae(
            x         = ref_embs, 
            label_acc = acc_labels, 
            label_spk = spk_labels
        )
        # print(out['prob_cat'].shape, out['logits'].shape)

        return (z_acc, y_acc, z_spk, y_spk, accent_prediction)

    def inference(self, inputs: torch.Tensor, acc_labels: torch.Tensor, spk_labels: torch.Tensor):
        ref_embs = self.encoder(inputs)

        # (z, (mu, var, y_mu, y_var, logits)) = self.mlvae.inference(enc_out,labels)
        # (z_acc, y_acc, (mu_acc, var_acc)) = self.acc_vae.inference(enc_out,acc_labels)
        # (z_spk, y_spk, (mu_spk, var_spk)) = self.spk_vae.inference(enc_out,spk_labels)
        # print(out['prob_cat'].shape, out['logits'].shape)
        (
            z_acc, 
            y_acc, 
            z_spk, 
            y_spk,
            accent_prediction
        ) = self.cvae.inference(
            x         = ref_embs, 
            label_acc = acc_labels, 
            label_spk = spk_labels
        )

        return (z_acc, y_acc, z_spk, y_spk, accent_prediction)
        # return (z_acc, z_spk)


class CVAEnet(nn.Module):
    """Accented Text-to-Speech Synthesis with a Conditional Variational Autoencoder as in:

    https://arxiv.org/abs/2211.03316

    """
    def __init__(self, n_classes_acc: int, n_classes_spk: int, x_dim: int, y_dim: int, z_dim: int):

        super(CVAEnet, self).__init__()

        # initialize conf
        self.n_classes_acc = n_classes_acc
        self.n_classes_spk = n_classes_spk

        # accent embeddings
        self.x_dim_acc = x_dim
        self.y_dim_acc = y_dim
        self.z_dim_acc = z_dim
        self.embedding_layer_acc = nn.Embedding(
            num_embeddings = self.n_classes_acc,
            embedding_dim  = self.y_dim_acc
        )
        self.linear_model_acc = nn.Sequential(
            nn.Linear(
                in_features  = self.x_dim_acc + self.y_dim_acc, 
                out_features = 256, 
                bias         = True
            ),
            nn.Tanh()
        )
        self.mu_layer_acc = LinearNorm(
            in_features  = 256, 
            out_features = self.z_dim_acc,
            bias         = True
        )
        self.logvar_layer_acc = LinearNorm(
            in_features  = 256,
            out_features = self.z_dim_acc,
            bias         = True
        )

        #SPK
        self.x_dim_spk = x_dim
        self.y_dim_spk = y_dim
        self.z_dim_spk = z_dim
        self.embedding_layer_spk=nn.Embedding(
            num_embeddings = self.n_classes_spk,
            embedding_dim  = self.y_dim_spk
        )
        self.linear_model_spk = nn.Sequential(
            nn.Linear(
                in_features  = self.x_dim_spk + self.y_dim_spk, 
                out_features = 256, 
                bias         = True
            ),
            nn.Tanh()
        )
        self.mu_layer_spk = LinearNorm(
            in_features  = 256, 
            out_features = self.z_dim_spk,
            bias         = True
        )
        self.logvar_layer_spk = LinearNorm(
            in_features  = 256,
            out_features = self.z_dim_spk,
            bias         = True
        )

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, label_acc: torch.Tensor, label_spk: torch.Tensor):
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of padded target features (B, Tmax, D).
            label_acc (LongTensor): Batch of accent labels (B, ).
            label_spk (LongTensor): Batch of speaker labels (B, ).

        Returns:
            Tensor: replicated input tensor based on durations (B, T*, D).

        """

        x = x.view(x.size(0), -1)

        #ACC
        y_acc        = self.embedding_layer_acc(label_acc)
        x_acc        = self.linear_model_acc(torch.cat([x,y_acc],axis=1))
        x_mu_acc     = self.mu_layer_acc(x_acc)
        x_logvar_acc = self.logvar_layer_acc(x_acc)
                
        z_acc = reparameterize(training=True, mu=x_mu_acc, logvar=x_logvar_acc)

        #SPK
        y_spk        = self.embedding_layer_spk(label_spk)
        x_spk        = self.linear_model_spk(torch.cat([x,y_spk],axis=1))
        x_mu_spk     = self.mu_layer_spk(x_spk)
        x_logvar_spk = self.logvar_layer_spk(x_spk)
                
        z_spk = reparameterize(training=True, mu=x_mu_spk, logvar=x_logvar_spk)
                
        # cat_prob = self.categorical_layer(class_latent_embeddings)

        return (z_acc, y_acc, z_spk, y_spk, (x_mu_acc, x_logvar_acc, x_mu_spk, x_logvar_spk))

    def inference(self, x, label_acc, label_spk):
        x = x.view(x.size(0), -1)

        #ACC
        y_acc        = self.embedding_layer_acc(label_acc)
        x_acc        = self.linear_model_acc(torch.cat([x,y_acc],axis=1))
        x_mu_acc     = self.mu_layer_acc(x_acc)
        x_logvar_acc = self.logvar_layer_acc(x_acc)

        z_acc = x_mu_acc
        # z_acc = reparameterize(training=False, mu=x_mu_acc, logvar=x_logvar_acc)

        #SPK
        y_spk        = self.embedding_layer_spk(label_spk)
        x_spk        = self.linear_model_spk(torch.cat([x,y_spk],axis=1))
        x_mu_spk     = self.mu_layer_spk(x_spk)
        x_logvar_spk = self.logvar_layer_spk(x_spk)

        z_spk = x_mu_spk
        # z_spk = reparameterize(training=False, mu=x_mu_spk, logvar=x_logvar_spk)
        # cat_prob = self.categorical_layer(class_latent_embeddings)

        return (z_acc, y_acc, z_spk, y_spk, (x_mu_acc, x_logvar_acc, x_mu_spk, x_logvar_spk))
