from typing import Optional, Tuple
import torch
import torch.nn as nn
from models.tts.fastspeech2.layers import VarianceAdaptor, AccentEncoderNetwork, Postnet
from models.tts.fastspeech2.function import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """FastSpeech2 module.

    This is a module of FastSpeech2 described in `FastSpeech 2: Fast and
    High-Quality End-to-End Text to Speech`_. Instead of quantized pitch and
    energy, we use token-averaged value introduced in `FastPitch: Parallel
    Text-to-speech with Pitch Prediction`_.

    .. _`FastSpeech 2: Fast and High-Quality End-to-End Text to Speech`:
        https://arxiv.org/abs/2006.04558
    .. _`FastPitch: Parallel Text-to-speech with Pitch Prediction`:
        https://arxiv.org/abs/2006.06873

    """

    def __init__(self,
                 n_symbols: int,
                 n_channels: int,
                 hparams: dict,
                 stats: dict,
                 n_speakers: int,
                 n_accents: int=2,
                 ) -> None:
        super(FastSpeech2, self).__init__()
        """ Initialize FastSpeech2 module. """

        self.hparams = hparams
        self.idim    = n_symbols
        self.odim    = n_channels
        self.building_block = self.hparams["building_block"]["block_type"]

        if self.building_block == "transformer":
            from models.tts.fastspeech2.blocks.transformer import Encoder, Decoder
        elif self.building_block == "conformer":
            from models.tts.fastspeech2.blocks.conformer import Encoder, Decoder
        else:
            raise NotImplementedError(f"Building block {self.building_block} will be implemented in future!")

        self.encoder = Encoder(
            layers      = self.hparams["encoder_layers"],
            hidden_dim  = self.hparams["encoder_hidden"],
            max_seq_len = self.hparams["max_seq_len"],
            n_symbols   = self.idim,
            config      = self.hparams["building_block"][self.building_block]
        )
        self.decoder = Decoder(
            layers      = self.hparams["decoder_layers"],
            hidden_dim  = self.hparams["decoder_hidden"],
            max_seq_len = self.hparams["max_seq_len"],
            config      = self.hparams["building_block"][self.building_block]
        )

        self.spk_dims = self.hparams.get("spk_dims", -1)
        self.use_cvae = self.hparams.get("use_cvae", False)
        if self.spk_dims != -1:
            self.speaker_emb = nn.Linear(
                in_features  = self.spk_dims,
                out_features = self.hparams["encoder_hidden"]
            )
        elif self.use_cvae is True:
            self.speaker_emb = AccentEncoderNetwork(
                idim         = self.odim,
                n_speakers   = n_speakers,
                n_accents    = n_accents, # only for hn & hcm
                ref_hparams  = self.hparams["vcae"],
                cvae_hparams = {
                    "x_dim": self.hparams["vcae"]["gru_units"],
                    "y_dim": 10,
                    "z_dim": self.hparams["encoder_hidden"] // 2
                }
            )
        else:
            self.speaker_emb = nn.Embedding(
                num_embeddings = n_speakers,
                embedding_dim  = self.hparams["encoder_hidden"]
            )
        self.spk_dims = self.hparams["encoder_hidden"]

        self.variance_adaptor = VarianceAdaptor(
            n_channels = self.odim,
            hidden_dim = self.hparams["encoder_hidden"],
            config     = self.hparams["variance"],
            stats      = stats
        )
        self.feats_linear = nn.Linear(
            in_features  = self.hparams["decoder_hidden"],
            out_features = self.odim
        )
        self.postnet = None
        if self.hparams.get("use_postnet") is True:
            self.postnet = Postnet(
                n_channels = self.odim,
                config     = self.hparams["postnet"]
            )

        self.learn_alignment = self.variance_adaptor.learn_alignment

    def forward(
        self, 
        text: torch.Tensor, 
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        duration: torch.Tensor,
        pitch: torch.Tensor,
        energy: torch.Tensor,
        sids: torch.Tensor,
        aids: Optional[torch.Tensor]=None,
        step: int=0,
        *kwags
    ) -> Tuple:

        # encoder forward
        max_text_length  = torch.max(text_lengths).item()
        text_masks       = get_mask_from_lengths(text_lengths, max_text_length)
        max_feats_length = torch.max(feats_lengths).item()
        feats_masks      = get_mask_from_lengths(feats_lengths, max_feats_length)

        hs, text_embs    = self.encoder(text, text_masks)

        # integrate with speaker embedding
        accent_probs = None
        if self.use_cvae is True:
            assert aids is not None, "Accent labels should not be None"
            (z_acc, y_acc, z_spk, y_spk, accent_probs) = \
                self.speaker_emb(feats, acc_labels=aids, spk_labels=sids)
            sid_embs = torch.cat([z_acc, z_spk], axis=1)
        else:
            sid_embs = self.speaker_emb(sids)

        # variance adaptor forward
        (
            hs, 
            log_d_outs, 
            p_outs,
            e_outs,  
            feats_lengths,
            feats_masks,
            attn_outs
        ), (
            ds,
            ps,
            es
        ) = self.variance_adaptor(
            x               = hs,
            txt_emb         = text_embs,
            txt_lens        = text_lengths,
            txt_mask        = text_masks,
            max_txt_len     = max_text_length,
            spk_emb         = sid_embs,
            mel             = feats,
            mel_lens        = feats_lengths,
            mel_mask        = feats_masks,
            max_mel_len     = max_feats_length,
            pitch_target    = pitch, 
            energy_target   = energy, 
            duration_target = None if self.learn_alignment is True else duration,
            attn_prior      = duration if self.learn_alignment is True else None,
            step            = step
        )

        # decoder forward
        hs, feats_masks = self.decoder(hs, feats_masks)
        outs = self.feats_linear(hs)
        postnet_outs = None
        if self.postnet is not None: postnet_outs = self.postnet(outs) + outs

        return (
            outs,
            postnet_outs,
            log_d_outs,
            p_outs,
            e_outs,
            text_lengths,
            text_masks, 
            feats_lengths,
            feats_masks,
            accent_probs,
            attn_outs,
            sid_embs
        ), (
            ds,
            ps,
            es
        )

    def inference(
        self, 
        sids: torch.Tensor, 
        text: torch.Tensor, 
        text_lengths: torch.Tensor, 
        feats: Optional[torch.Tensor]=None,
        aids: Optional[torch.Tensor]=None,
        d_control: float=1.0, 
        p_control: float=1.0, 
        e_control: float=1.0,
        **kwangs
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        
        # encode forward
        max_text_length  = torch.max(text_lengths).item()
        text_masks       = get_mask_from_lengths(text_lengths, max_text_length)

        hs, text_embs    = self.encoder(text, text_masks)

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

        # variance adapt forward
        (
            hs, 
            _, 
            _,
            _,  
            feats_lengths,
            feats_masks,
            _
        ), (
            _,
            _,
            d_outs
        ) = self.variance_adaptor(
            x           = hs,
            txt_emb     = text_embs,
            txt_lens    = text_lengths,
            txt_mask    = text_masks,
            max_txt_len = max_text_length,
            spk_emb     = sid_embs,
            d_control   = d_control, 
            p_control   = p_control, 
            e_control   = e_control
        )

        # decode forward
        hs, feats_masks = self.decoder(hs, feats_masks)
        outs = self.feats_linear(hs)
        if self.postnet is not None:
            outs = self.postnet(outs) + outs

        return outs.transpose(1, 2), feats_lengths, sid_embs
