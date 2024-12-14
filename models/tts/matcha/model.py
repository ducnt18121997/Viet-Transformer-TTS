from typing import Dict, Tuple, Optional
import numpy as np 
import torch
import torch.nn as nn
from models.tts.matcha.layers import TextEncoder, CFM
from models.tts.matcha.function import sequence_mask, fix_len_compatibility
from models.tts.fastspeech2.layers import AlignmentEncoder, DurationPredictor, GaussianUpsampling, LengthRegulator, AccentEncoderNetwork
from models.tts.fastspeech2.function import b_mas


class MatchaTTS(nn.Module):
    """MatchaTTS module.

    This is a module of MatchaTTS described in `Matcha-TTS: A fast TTS 
    architecture with conditional flow matching`_.

    .. _`Matcha-TTS: A fast TTS architecture with conditional flow matching`:
        https://arxiv.org/abs/2309.03199

    """

    def __init__(self,
                 n_symbols: int,
                 n_speakers: int,
                 n_channels: int,
                 hparams: Dict,
                 stats: Dict,
                 n_accents: int=None,
                 ) -> None:
        super(MatchaTTS, self).__init__()
        """ Initialize MatchaTTS module. """
        
        self.hparams    = hparams
        self.n_symbols  = n_symbols
        self.n_channels = n_channels
        self.n_speakers = n_speakers
        self.n_accents  = n_accents
        
        # initialize encoder 
        self.speaker_emb_dim = self.hparams["spk_emb_dim"]
        self.encoder = TextEncoder(
            n_vocab = self.n_symbols,
            n_feats = self.n_channels,
            n_spks  = self.n_speakers,
            spk_emb_dim = self.speaker_emb_dim,
            encoder_params = self.hparams["text_encoder"]
        )

        # intialize speaker embedding 
        self.spk_dims = self.hparams.get("spk_dims", -1)
        self.use_cvae = self.hparams.get("use_cvae", False)
        if self.spk_dims != -1:
            self.speaker_emb = nn.Linear(
                in_features  = self.spk_dims,
                out_features = self.hparams["spk_emb_dim"]
            )
        elif self.use_cvae is True:
            self.speaker_emb = AccentEncoderNetwork(
                idim         = self.n_channels,
                n_speakers   = self.n_speakers,
                n_accents    = self.n_accents, # only for hn & hcm
                ref_hparams  = self.hparams["vcae"],
                cvae_hparams = {
                    "x_dim": self.hparams["vcae"]["gru_units"],
                    "y_dim": 10,
                    "z_dim": self.hparams["spk_emb_dim"] // 2
                }
            )
        else:
            self.speaker_emb = nn.Embedding(
                num_embeddings = n_speakers,
                embedding_dim  = self.hparams["spk_emb_dim"]
            )

        # intialize duration modelling
        self.aligner = AlignmentEncoder(
            n_mel_channels  = self.n_channels,
            n_att_channels  = self.n_channels,
            n_text_channels = self.encoder.encoder_params["hidden_dim"],
            temperature     = self.hparams["duration_modelling"]["aligner_temperature"],
            n_spk_channels  = self.speaker_emb_dim
        )
        self.binarization_start_steps = self.hparams["duration_modelling"]["binarization_start_steps"]
        self.duration_predictor = DurationPredictor(
            self.n_channels,
            n_chans      = self.n_channels,
            n_layers     = self.hparams["variance_predictor"]["dur_predictor_layers"],
            dropout_rate = self.hparams["variance_predictor"]["dropout"], 
            padding      = self.hparams["variance_predictor"]["ffn_padding"],
            kernel_size  = self.hparams["variance_predictor"]["dur_predictor_kernel"]
        )

        self.use_gaussian = self.hparams["duration_modelling"]["use_gaussian"]
        if self.use_gaussian is True:
            self.length_regulator = GaussianUpsampling()
        else:
            self.length_regulator = LengthRegulator()

        # initialize decoder
        self.decoder = CFM(
            in_channels    = 2 * self.n_channels,
            out_channel    = self.n_channels,
            cfm_params     = self.hparams["flow_matching"],
            decoder_params = self.hparams["decoder"],
            n_spks         = self.n_speakers,
            spk_emb_dim    = self.speaker_emb_dim
        )
        self.data_statistics = stats
    
    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        mu, std = self.data_statistics["mel_mean"], self.data_statistics["mel_std"]

        if not isinstance(mu, float):
            if isinstance(mu, list):
                mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
            elif isinstance(mu, torch.Tensor):
                mu = mu.to(data.device)
            elif isinstance(mu, np.ndarray):
                mu = torch.from_numpy(mu).to(data.device)
            mu = mu.unsqueeze(-1)

        if not isinstance(std, float):
            if isinstance(std, list):
                std = torch.tensor(std, dtype=data.dtype, device=data.device)
            elif isinstance(std, torch.Tensor):
                std = std.to(data.device)
            elif isinstance(std, np.ndarray):
                std = torch.from_numpy(std).to(data.device)
            std = std.unsqueeze(-1)

        return data * std + mu

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

    def forward(
        self, 
        text: torch.Tensor, 
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        sids: Optional[torch.Tensor]=None, 
        aids: Optional[torch.Tensor]=None, 
        duration: torch.Tensor=None,
        step: int=0,
        *kwags
    ):
        
        # integrate with speaker embedding
        accent_probs = None
        if self.use_cvae is True:
            assert aids is not None, "Accent labels should not be None"
            (z_acc, y_acc, z_spk, y_spk, accent_probs) = \
                self.speaker_emb(feats.transpose(1, 2), acc_labels=aids, spk_labels=sids)
            sid_embs = torch.cat([z_acc, z_spk], axis=1)
        else:
            sid_embs = self.speaker_emb(sids)
        
        # Get encoder forward
        mu_x, txt_emb, txt_mask = self.encoder(text, text_lengths, sid_embs)
        
        # Get duration modelling
        mu_x = mu_x.transpose(1, 2)
        txt_mask = ~txt_mask
        log_duration_prediction = self.duration_predictor(mu_x, txt_mask)
        attn_soft, attn_logprob = self.aligner(feats, txt_emb, txt_mask.unsqueeze(-1), duration, sid_embs)

        attn_hard     = self.binarize_attention_parallel(attn_soft, text_lengths, feats_lengths)
        attn_hard_dur = attn_hard.sum(2)[:, 0, :]
        attn_out      = (attn_soft, attn_hard, attn_hard_dur, attn_logprob)
        
        duration_rounded = attn_hard_dur.long()
        
        # expand duration (text lens -> melspec lens)
        feats_max_length = feats.shape[-1]
        feats_mask = sequence_mask(feats_lengths, feats_max_length).unsqueeze(1).to(txt_mask.device)
        if duration is not None and step < self.binarization_start_steps:
            mu_y = torch.bmm(attn_soft.squeeze(1), mu_x).transpose(1, 2)
        else:
            if self.use_gaussian:
                mu_y = self.length_regulator(mu_x, duration_rounded, feats_mask.squeeze(1), ~txt_mask).transpose(1, 2)
            else:
                mu_y = self.length_regulator(mu_x, duration_rounded).transpose(1, 2)
        dec_outs, u = self.decoder(feats, mask=feats_mask, mu=mu_y, spks=sid_embs)

        return (dec_outs, u, mu_y, txt_mask, feats_mask, feats_lengths, duration_rounded, log_duration_prediction, attn_out, accent_probs)

    def inference(
        self, 
        text: torch.Tensor, 
        text_lengths: torch.Tensor, 
        sids: Optional[torch.Tensor] = None,
        aids: Optional[torch.Tensor] = None,
        feats: Optional[torch.Tensor] = None,
        d_control: float = 1.0, # NOTE: length_scale
        n_timesteps: int = 10, 
        temperature: float = 0., # repo default: 0.677
        **kwags
    ):
        # integrate with speaker embedding            
        if self.use_cvae is True:
            if len(sids.shape) != 1:
                # NOTE: specified case when use cvae
                sid_embs = sids if aids is None else torch.cat([aids, sids], axis=1)
            else:
                if feats is None: raise RuntimeError("missing required argument: 'feats'")
                if aids is None: raise RuntimeError("missing required argument: 'aids'")
                (z_acc, _, z_spk, _, _) = self.speaker_emb.inference(feats, acc_labels=aids, spk_labels=sids)
                sid_embs = torch.cat([z_acc, z_spk], axis=1)
        else:
            sid_embs = self.speaker_emb(sids)
        sid_embs = sid_embs.expand(text.shape[0], -1)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, _, txt_mask = self.encoder(text, text_lengths, sid_embs)
        mu_x = mu_x.transpose(1, 2)
        txt_mask = ~txt_mask
        
        # Integrated with duration modelling
        log_duration_prediction = self.duration_predictor(mu_x, txt_mask)
        duration_rounded = torch.clamp(
            (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
            min=0
        ).long()
        
        feats_lengths = torch.sum(duration_rounded, dim=1)
        feats_max_length  = int(feats_lengths.max())
        feats_max_length_ = fix_len_compatibility(feats_max_length)
        feats_mask = sequence_mask(feats_lengths, feats_max_length_).unsqueeze(1).to(txt_mask.device)
        if self.use_gaussian:
            mu_y = self.length_regulator(mu_x, duration_rounded, feats_mask.squeeze(1), ~txt_mask).transpose(1, 2)
        else:
            mu_y = self.length_regulator(mu_x, duration_rounded).transpose(1, 2)

        # Generate sample tracing the probability flow
        dec_outs = self.decoder.inference(mu_y, feats_mask, n_timesteps, temperature, sid_embs)
        dec_outs = dec_outs[:, :, :feats_max_length]

        outs = self.denormalize(dec_outs)

        return outs, feats_lengths
