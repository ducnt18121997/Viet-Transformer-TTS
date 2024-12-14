import numpy as np
from typing import Tuple, Dict
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tts.fastspeech2.layers import AlignmentEncoder, DurationPredictor, VariancePredictor
from models.tts.fastspeech2.sublayers import LayerNorm
from models.tts.fastspeech2.function import b_mas, get_phoneme_level, get_mask_from_lengths
from models.tts.adaspeech.function import get_mel_phoneme_level


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
        self.stats      = stats
        self.hidden_dim = hidden_dim
        self.n_channels = n_channels
        self.conf       = config

        # initialize duration modelling
        self.learn_alignment = self.conf["learn_alignment"]
        self.aligner, self.binarization_start_steps = None, None
        if self.learn_alignment is True:
            self.aligner = AlignmentEncoder(
                n_mel_channels  = self.n_channels,
                n_att_channels  = self.n_channels,
                n_text_channels = self.hidden_dim,
                temperature     = self.conf["duration_modelling"]["aligner_temperature"]
            )
            self.binarization_start_steps = self.conf["duration_modelling"]["binarization_start_steps"]
        self.duration_predictor = DurationPredictor(
            self.hidden_dim,
            n_chans      = self.n_channels,
            n_layers     = self.conf["variance_predictor"]["dur_predictor_layers"],
            dropout_rate = self.conf["variance_predictor"]["dropout"], 
            padding      = self.conf["variance_predictor"]["ffn_padding"],
            kernel_size  = self.conf["variance_predictor"]["dur_predictor_kernel"]
        )

        self.use_gaussian = self.conf["duration_modelling"]["use_gaussian"]
        if self.use_gaussian is True:
            from models.tts.fastspeech2.layers import GaussianUpsampling
            self.length_regulator = GaussianUpsampling()
        else:
            from models.tts.fastspeech2.layers import LengthRegulator
            self.length_regulator = LengthRegulator()

        # initialize ada modules
        self.phoneme_level_encoder_step = self.conf["reference_encoder"]["phoneme_level_encoder_step"]
        self.utterance_encoder = UtteranceEncoder(
            **self.conf["reference_encoder"]["utterance_encoder"]
        )
        self.phoneme_level_encoder = PhonemeLevelEncoder(
            **self.conf["reference_encoder"]["phoneme_level_encoder"],
            phn_latent_dim = self.conf["reference_encoder"]["phn_latent_dim"]
        )
        self.phoneme_level_predictor = PhonemeLevelPredictor(
            **self.conf["reference_encoder"]["phoneme_level_predictor"],
            phn_latent_dim = self.conf["reference_encoder"]["phn_latent_dim"]
        )
        self.phone_level_embed = nn.Linear(
            in_features  = self.conf["reference_encoder"]["phn_latent_dim"],
            out_features = self.hidden_dim
        )

        # initilize pitch modelling
        self.pitch_feature_level = self.conf["variance_embedding"]["pitch_feature"]
        assert self.pitch_feature_level in ["frame_level", "phoneme_level"]
        self.pitch_predictor = VariancePredictor(
            idim         = self.hidden_dim,
            n_chans      = self.conf["variance_predictor"]["filter_size"],
            n_layers     = self.conf["variance_predictor"]["pit_predictor_layers"],
            dropout_rate = self.conf["variance_predictor"]["dropout"],
            odim         = 1,
            padding      = self.conf["variance_predictor"]["ffn_padding"], 
            kernel_size  = self.conf["variance_predictor"]["pit_predictor_kernel"]
        )
        self.pitch_embedding = nn.Embedding(
            num_embeddings = self.conf["variance_embedding"]["n_bins"],
            embedding_dim  = self.hidden_dim
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
            idim         = self.hidden_dim,
            n_chans      = self.conf["variance_predictor"]["filter_size"],
            n_layers     = self.conf["variance_predictor"]["ener_predictor_layers"],
            dropout_rate = self.conf["variance_predictor"]["dropout"],
            odim         = 1,
            padding      = self.conf["variance_predictor"]["ffn_padding"], 
            kernel_size  = self.conf["variance_predictor"]["ener_predictor_kernel"]
        )
        self.energy_embedding = nn.Embedding(
            num_embeddings = self.conf["variance_embedding"]["n_bins"],
            embedding_dim  = self.hidden_dim
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
        step: int=0
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

        # ada feature
        if mel.size(-1) == 80:
            x = x + self.utterance_encoder(mel.transpose(1, 2)).transpose(1, 2)
            avg_mels = get_mel_phoneme_level(mel, txt_lens, duration_rounded, max_txt_len)
        else:
            # NOTE(by deanng): consider `mel`` as presentation of speaker voices
            x = x + mel.unsqueeze(0).expand(x.size(0), -1, -1)
            avg_mels = None

        # NOTE(by deanng): training step
        phn_pred = None
        if step >= self.phoneme_level_encoder_step:
            phn_pred = self.phoneme_level_predictor(x.transpose(1, 2))
            with torch.no_grad():
                phn_encode = self.phoneme_level_encoder(avg_mels.transpose(1, 2))
            x = x + self.phone_level_embed(phn_encode.detach())
        else:
            if avg_mels is not None:
                phn_encode = self.phoneme_level_encoder(avg_mels.transpose(1, 2))
            else:
                # NOTE(by deanng): inference step
                phn_encode = self.phoneme_level_predictor(x.transpose(1, 2))
            x = x + self.phone_level_embed(phn_encode)

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
            attn_out,
            phn_pred
        ), (
            duration_rounded,
            pitch_target,
            energy_target,
            phn_encode
        )


class UtteranceEncoder(nn.Module):
    """ Acoustic modeling """
    
    def __init__(self, idim: int, n_layers: int, n_chans: int, kernel_size: int, pool_kernel: int, dropout_rate: float, stride: int):
        super(UtteranceEncoder, self).__init__()
        
        self.idim         = idim
        self.n_layers     = n_layers
        self.n_chans      = n_chans
        self.kernel_size  = kernel_size
        self.pool_kernel  = pool_kernel
        self.dropout_rate = dropout_rate
        self.stride       = stride

        self.conv = nn.Sequential(
            OrderedDict(
                [
                    ("conv1d_1",
                    nn.Conv1d(
                        self.idim,
                        self.n_chans,
                        self.kernel_size,
                        stride = self.stride,
                        padding = (self.kernel_size - 1) // 2,
                    )
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", LayerNorm(self.n_chans, dim=1)),
                    ("dropout_1", nn.Dropout(self.dropout_rate)),
                    ("conv1d_2",
                    nn.Conv1d(
                        self.n_chans,
                        self.n_chans,
                        self.kernel_size,
                        stride = self.stride,
                        padding = (self.kernel_size - 1) // 2,
                    )
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", LayerNorm(self.n_chans, dim=1)),
                    ("dropout_2", nn.Dropout(self.dropout_rate)),
                ]
            )
        )
        
    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        xs = self.conv(xs)
        xs = F.avg_pool1d(xs, xs.size(-1))

        return xs


class PhonemeLevelEncoder(nn.Module):
    """ Phoneme level encoder """
    def __init__(self, idim: int, n_layers: int, n_chans: int, kernel_size: int, dropout_rate: int, stride: int, phn_latent_dim: int):
        super(PhonemeLevelEncoder, self).__init__()
        self.idim         = idim
        self.n_layers     = n_layers
        self.n_chans      = n_chans
        self.kernel_size  = kernel_size
        self.dropout_rate = dropout_rate
        self.stride       = stride
        
        self.conv = nn.Sequential(
            OrderedDict(
                [
                    ("conv1d_1",
                    nn.Conv1d(
                        self.idim,
                        self.n_chans,
                        self.kernel_size,
                        stride = self.stride,
                        padding = (self.kernel_size - 1) // 2,
                    )
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", LayerNorm(self.n_chans, dim=1)),
                    ("dropout_1", nn.Dropout(self.dropout_rate)),
                    ("conv1d_2",
                    nn.Conv1d(
                        self.n_chans,
                        self.n_chans,
                        self.kernel_size,
                        stride = self.stride,
                        padding = (self.kernel_size - 1) // 2,
                    )
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", LayerNorm(self.n_chans, dim=1)),
                    ("dropout_2", nn.Dropout(self.dropout_rate)),
                ]
            )
        )
        self.linear = nn.Linear(self.n_chans, phn_latent_dim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        xs = self.conv(xs)
        xs = self.linear(xs.transpose(1,2))
        return xs


class PhonemeLevelPredictor(nn.Module):
    """ PhonemeLevelPredictor """
    def __init__(self, idim: int, n_layers: int, n_chans: int, kernel_size: int, dropout_rate: float, stride: int, phn_latent_dim: int):
        super(PhonemeLevelPredictor, self).__init__()
        self.idim         = idim
        self.n_layers     = n_layers
        self.n_chans      = n_chans
        self.kernel_size  = kernel_size
        self.dropout_rate = dropout_rate
        self.stride       = stride
        
        self.conv = nn.Sequential(
            OrderedDict(
                [
                    ("conv1d_1",
                    nn.Conv1d(
                        self.idim,
                        self.n_chans,
                        self.kernel_size,
                        stride = self.stride,
                        padding = (self.kernel_size - 1) // 2,
                    )
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", LayerNorm(self.n_chans, dim=1)),
                    ("dropout_1", nn.Dropout(self.dropout_rate)),
                    ("conv1d_2",
                    nn.Conv1d(
                        self.n_chans,
                        self.n_chans,
                        self.kernel_size,
                        stride = self.stride,
                        padding = (self.kernel_size - 1) // 2,
                    )
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", LayerNorm(self.n_chans, dim=1)),
                    ("dropout_2", nn.Dropout(self.dropout_rate)),
                ]
            )
        )
        self.linear = nn.Linear(self.n_chans, phn_latent_dim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        xs = self.conv(xs)
        xs = self.linear(xs.transpose(1, 2))
        
        return xs
