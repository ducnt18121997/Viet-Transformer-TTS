import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from models.tts.fastspeech2.loss import FastSpeech2Loss


class AdaSpeechLoss(FastSpeech2Loss):
    def __init__(self, n_iter: int, config: dict, is_finetune: bool=False):
        super().__init__(n_iter, config, is_finetune)
    
    def _forward_avg_phn_losses(self, avg_phn_predictions: torch.Tensor, avg_phn_targets: torch.Tensor, masks: Optional[torch.Tensor]) -> dict:
        loss = {}

        avg_phn_targets.requires_grad = False
        avg_phn_predictions = avg_phn_predictions.masked_select(masks.unsqueeze(-1))
        avg_phn_targets     = avg_phn_targets.masked_select(masks.unsqueeze(-1))
        loss["avg_mel_phn"] = F.mse_loss(avg_phn_predictions, avg_phn_targets)

        return loss

    def forward(self, predictions: Tuple, targets: Tuple, is_joint: bool=False, step: Optional[int]=None):
        feats_predictions, postnet_feats_predictions, \
            log_duration_predictions, pitch_predictions, energy_predictions, \
            src_lens, src_masks, feats_lens, feats_masks, accent_probs, attn_outs, \
            avg_mel_phn_predictions = predictions
        feats_targets, word_boundaries, duration_targets, \
            pitch_targets, energy_targets, avg_mel_phn_encode= targets

        src_masks   = ~src_masks
        feats_masks = ~feats_masks

        feats_targets = feats_targets[:, : feats_masks.shape[1], :]
        feats_masks   = feats_masks[:, :feats_masks.shape[1]]

        # calculate mel-spectrograms loss
        losses = self._forward_feats_losses(
            feats_predictions         = feats_predictions,
            feats_targets             = feats_targets,
            masks                     = feats_masks if is_joint is False else None,
            postnet_feats_predictions = postnet_feats_predictions
        )
        if step is not None:
            # calculate self-align loss
            if attn_outs is not None:
                losses.update(self._forward_align_losses(
                    attn_matrix = attn_outs,
                    input_lens  = src_lens,
                    output_lens = feats_lens,
                    step        = step
                ))
            # calculate cvae loss
            if accent_probs is not None:
                losses.update(self._forward_cvae_losses(
                    prob_ = accent_probs,
                    step  = step
                ))
            # calculate ada module loss
            if avg_mel_phn_predictions is not None:
                losses.update(self._forward_avg_phn_losses(
                    avg_phn_predictions = avg_mel_phn_predictions,
                    avg_phn_targets     = avg_mel_phn_encode,
                    masks               = src_masks 
                ))
            # calculate duration loss (with 3 target: phonemes, words and sentences)
            losses.update(self._forward_duration_losses(
                log_duration_predictions = log_duration_predictions,
                duration_targets         = duration_targets,
                word_boundaries          = word_boundaries,
                masks                    = src_masks
            ))
            # calculate pitch loss
            losses.update(self._forward_pitch_losses(
                pitch_predictions = pitch_predictions,
                pitch_targets     = pitch_targets,
                masks             = src_masks if self.pitch_feature_level == "phoneme_level" else feats_masks
            ))
            # calculate energy loss
            losses.update(self._forward_energy_losses(
                energy_predictions = energy_predictions,
                energy_targets     = energy_targets,
                masks              = src_masks if self.pitch_feature_level == "phoneme_level" else feats_masks
            ))

        return losses
