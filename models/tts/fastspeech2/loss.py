import numpy as np
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tts.fastspeech2.function import phone2words


class FastSpeech2Loss(nn.Module):
    """ Unsupervised-durations learning FastSpeech2 Loss """

    def __init__(self, n_iter: int, config: dict, is_finetune: bool=False):
        super(FastSpeech2Loss, self).__init__()

        self.is_finetune = is_finetune
        self.linbuild    = config["linbuild"]

        # initialize losses
        self.sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()
        if is_finetune is True:
            self.L = 5e-4
        else:
            self.L = linear_buildup(
                n_iter = n_iter,
                n_stop = self.linbuild["n_stop"], 
                n_up   = self.linbuild["n_up"],
                start  = self.linbuild["start"],
                stop   = self.linbuild["stop"]
            )

        # initialize lamda values
        self.binarization_loss_enable_steps = config["binarization_loss_enable_steps"]
        self.binarization_loss_warmup_steps = config["binarization_loss_warmup_steps"]
        self.dur_loss_lambda                = config["dur_loss_lambda"]
        
        self.pitch_feature_level  = config["pitch_feature_level"]
        self.energy_feature_level = config["energy_feature_level"]
    
    def _forward_duration_losses(self, log_duration_predictions: torch.Tensor, duration_targets: torch.Tensor, word_boundaries: list, masks: torch.Tensor) -> dict:
        nonpadding = masks.float()
        duration_targets     = duration_targets.float() * nonpadding
        duration_predictions = torch.clamp(torch.exp(log_duration_predictions) - 1, min=0)

        loss = {}
        # calculate phonemes duration loss (log_d_predictions, attn_hard_dur)
        duration_targets.requires_grad = False
        log_duration_targets = torch.log(duration_targets + 1)
        pdur_loss = F.mse_loss(log_duration_predictions, log_duration_targets)
        loss["pdur"] = pdur_loss
        
        # calculate words duration loss
        wdur_loss = torch.zeros(1).to(log_duration_predictions.device)
        if self.dur_loss_lambda["wdur"] > 0:
            word_duration_predictions = phone2words(duration_predictions, word_boundaries)
            word_duration_targets = phone2words(duration_targets, word_boundaries)
            wdur_loss = F.mse_loss(
                torch.log(word_duration_predictions + 1), 
                torch.log(word_duration_targets + 1), 
                reduction="none"
            )
            word_nonpadding = (word_duration_predictions > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
        loss["wdur"] = wdur_loss
        
        # calculate sentences duration loss
        sdur_loss = torch.zeros(1).to(log_duration_predictions.device)
        if self.dur_loss_lambda["sdur"] > 0:
            sentence_duration_predictions = duration_predictions.sum(-1)
            sentence_duration_targets = duration_targets.sum(-1)
            sdur_loss = F.mse_loss(
                torch.log(sentence_duration_predictions + 1), 
                torch.log(sentence_duration_targets + 1), 
                reduction="mean"
            )
            sdur_loss = sdur_loss.mean()
        loss["sdur"] = sdur_loss

        return loss

    def _forward_align_losses(self, attn_matrix: torch.Tensor, input_lens: torch.Tensor, output_lens: torch.Tensor, step: Optional[int]) -> dict:
        attn_soft, attn_hard, _, attn_logprob = attn_matrix

        loss = {}
        loss["ctc"] = self.sum_loss(attn_logprob=attn_logprob, in_lens=input_lens, out_lens=output_lens)
        if step < self.binarization_loss_enable_steps:
            bin_loss_weight = 0.
        else:
            bin_loss_weight = min((step - self.binarization_loss_enable_steps) / self.binarization_loss_warmup_steps, 1.0) * 1.0
        loss["bin"] = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft) * bin_loss_weight

        return loss

    def _forward_pitch_losses(self, pitch_predictions: torch.Tensor, pitch_targets: torch.Tensor, masks: torch.Tensor) -> dict:

        loss = {}
        pitch_targets.requires_grad = False
        pitch_predictions = pitch_predictions.masked_select(masks)
        pitch_targets     = pitch_targets.masked_select(masks)
        loss["pitch"]     = F.mse_loss(pitch_predictions, pitch_targets)

        return loss

    def _forward_energy_losses(self, energy_predictions: torch.Tensor, energy_targets: torch.Tensor, masks: torch.Tensor) -> dict:

        loss = {}
        energy_targets.requires_grad = False
        energy_predictions = energy_predictions.masked_select(masks)
        energy_targets     = energy_targets.masked_select(masks)
        loss["energy"]     = F.mse_loss(energy_predictions, energy_targets)

        return loss

    def _forward_feats_losses(self, feats_predictions: torch.Tensor, feats_targets: torch.Tensor, masks: Optional[torch.Tensor] = None, postnet_feats_predictions: Optional[torch.Tensor] = None) -> dict:
        
        loss = {}
        feats_targets.requires_grad = False
        if masks is not None:
            feats_predictions = feats_predictions.masked_select(masks.unsqueeze(-1))
            feats_targets = feats_targets.masked_select(masks.unsqueeze(-1))

        loss["feat"] = F.l1_loss(feats_predictions, feats_targets)
        # calculate mel-specs after postnet if exist
        if postnet_feats_predictions is not None:
            if masks is not None:
                postnet_feats_predictions = postnet_feats_predictions.masked_select(masks.unsqueeze(-1))
            loss["feat_postnet"] = F.l1_loss(postnet_feats_predictions, feats_targets)

        return loss

    def _forward_cvae_losses(self, prob_: Tuple, step: int):
        x_mu_acc, x_logvar_acc, x_mu_spk, x_logvar_spk = prob_
        
        loss = {}
        L = self.L if isinstance(self.L, float) else self.L[step]
        acc_kl_loss = L * kl_loss(x_mu_acc, x_logvar_acc)
        spk_kl_loss = L * kl_loss(x_mu_spk, x_logvar_spk)
        loss["cvae"] = acc_kl_loss + spk_kl_loss

        return loss
        
    def forward(self, predictions: Tuple, targets: Tuple, is_joint: bool=False, step: Optional[int]=None):
        feats_predictions, postnet_feats_predictions, \
            log_duration_predictions, pitch_predictions, energy_predictions, \
            src_lens, src_masks, feats_lens, feats_masks, accent_probs, attn_outs = predictions[: -1]
        feats_targets, word_boundaries, duration_targets, pitch_targets, energy_targets = targets

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


class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]

        return total_loss


class BinLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1], min=1e-12)).sum()

        return -log_sum / hard_attention.sum()


def linear_buildup(n_iter, n_stop=25000, n_up=5000, start=0.0001, stop=1.0):
    Llow  = np.ones(n_up) * start
    Lhigh = np.ones(n_iter - n_stop) * stop
    Lramp = np.linspace(start, stop, n_stop - n_up)
    
    return np.concatenate((Llow, Lramp, Lhigh))


def kl_loss(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        
    return torch.mean(0.5 * torch.sum(torch.exp(var) + mu ** 2 - 1. - var, 1))
