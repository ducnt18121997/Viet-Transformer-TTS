from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from models.gan_tts.hifigan import HiFiGAN as AudioUpsampler
from espnet2.gan_tts.utils import get_random_segments


class Text2Wav(nn.Module):
    """Joint-train module (FastSpeech2 (unsupervised) + HifiGan generator).
    
    This is a module of JETS described in `JETS: Jointly Training FastSpeech2 
    and HiFi-GAN for End to End Text to Speech'_.

    .. _`JETS: Jointly Training FastSpeech2 and HiFi-GAN for End to End Text to Speech`
        : https://arxiv.org/abs/2203.16852

    """

    def __init__(
            self,
            vocabs: int,
            aux_channels: int,
            text2mel_params: Dict, 
            mel2wav_params: Dict,
            version: str = "fastspeech2",
        ) -> None:
        super(Text2Wav, self).__init__()

        self.vocabs          = vocabs
        self.aux_channels    = aux_channels
        self.text2mel_params = text2mel_params
        self.mel2wav_params  = mel2wav_params

        self.version = version
        if self.version == "fastspeech2":
            from models.tts.fastspeech2 import FastSpeech2 as PhonemesAcoustic
        elif self.version == "adaspeech":
            from models.tts.adaspeech import AdaSpeech as PhonemesAcoustic
        elif self.version == "matcha":
            from models.tts.matcha import MatchaTTS as PhonemesAcoustic
        else:
            raise NotImplementedError(f"Not implemented for {self.version} yet")
        self.text2mel = PhonemesAcoustic(
            n_symbols  = self.vocabs,
            n_channels = self.aux_channels,
            **text2mel_params,
        )
        self.mel2wav = AudioUpsampler(
            in_channels     = self.aux_channels,
            out_channels    = 1,
            global_channels = -1
        )
        self.segment_size = self.mel2wav_params["segment_size"]
        self.upsample_factor = self.mel2wav.upsample_factor

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        duration: Optional[torch.Tensor]=None,
        pitch: Optional[torch.Tensor]=None,
        energy: Optional[torch.Tensor]=None,
        sids: Optional[torch.Tensor]=None,
        aids: Optional[torch.Tensor]=None,
        step: int=0
    ) -> Tuple:
        """Calculate forward propagation.

        Args:
            sids (Optional[Tensor]): Speaker index tensor (B,) or (B, 1).
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            duration (Tensor): In supervised model, batch of padded duration (B, T_feat, 1) else padded of aligment (B, T_text, T_feats)
            pitch (Tensor): Batch of padded pitch (B, T_feat, 1).
            energy (Tensor): Batch of padded energy (B, T_feat, 1).
            step (int): current step when training

        """
        if self.version in ["fastspeech2", "adaspeech"]:
            outputs, extra_inputs = self.text2mel(
                text          = text, 
                text_lengths  = text_lengths,
                feats         = feats, 
                feats_lengths = feats_lengths,
                duration      = duration, 
                pitch         = pitch, 
                energy        = energy,
                sids          = sids, 
                aids          = aids,
                step          = step
            )
            # slice outputs
            out_feats = outputs[0] if outputs[1] is None else outputs[1]
            out_segments, start_idxs = get_random_segments(
                x            = out_feats.transpose(1, 2),
                x_lengths    = feats_lengths,
                segment_size = self.segment_size,
            )
            out_wavs  = self.mel2wav(
                c = out_segments, 
                g = None if self.mel2wav.global_channels == -1 else outputs[-1].unsqueeze(-1)
            )
            outputs = (out_wavs, out_segments, None) + outputs[2: ]

        elif self.version == "matcha":
            outputs = self.text2mel(
                text          = text, 
                text_lengths  = text_lengths,
                feats         = feats, 
                feats_lengths = feats_lengths,
                sids          = sids, 
                aids          = aids, 
                duration      = duration, 
                step          = step
            )
            out_feats = outputs[0]
            # slice outputs
            out_segments, start_idxs = get_random_segments(
                x            = out_feats,
                x_lengths    = feats_lengths,
                segment_size = self.segment_size,
            )
            out_wavs  = self.mel2wav(
                c = self.text2mel.denormalize(out_segments), 
                g = None # if self.mel2wav.global_channels == -1 else sids.unsqueeze(-1)
            )
            outputs = (out_wavs, out_segments) + outputs[1: ]
            extra_inputs = None

        else:
            raise NotImplementedError()

        return (outputs, extra_inputs, start_idxs)

    def inference(self,
        sids: torch.Tensor, 
        text: torch.Tensor, 
        text_lengths: torch.Tensor, 
        feats: Optional[torch.Tensor]=None,
        aids: Optional[torch.Tensor]=None,
        d_control: float=1.0, 
        p_control: float=1.0, 
        e_control: float=1.0,
        **kwangs
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        outputs, feat_lens, sembs = self.text2mel.inference(
            sids         = sids,
            aids         = aids, 
            text         = text,
            text_lengths = text_lengths,
            feats        = feats,
            d_control    = d_control,
            p_control    = p_control,
            e_control   = e_control
        )
        outputs  = self.mel2wav(
            c = outputs,
            g = None if self.mel2wav.global_channels == -1 else sembs.unsqueeze(-1)
        )
        out_lens = feat_lens * self.mel2wav.upsample_factor

        return (outputs, out_lens)
