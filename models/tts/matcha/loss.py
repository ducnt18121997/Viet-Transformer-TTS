import math
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tts.fastspeech2.loss import BinLoss, ForwardSumLoss
from models.tts.fastspeech2.loss import linear_buildup, kl_loss


class MatchaTTSLoss(nn.Module):
    """ Unsupervised-durations learning FastSpeech2 Loss """

    def __init__(self, n_iter: int, config: dict, is_finetune: bool=False):
        super(MatchaTTSLoss, self).__init__()

        self.config = config
        self.n_channels = self.config["n_channels"]
        
        self.sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()
        self.linbuild = config["linbuild"]
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
        self.binarization_loss_enable_steps = self.config["binarization_loss_enable_steps"]
        self.binarization_loss_warmup_steps = self.config["binarization_loss_warmup_steps"]

    def _forward_duration_losses(self, log_duration_predictions: torch.Tensor, duration_targets: torch.Tensor, masks: torch.Tensor) -> dict:
        nonpadding = masks.float()
        duration_targets = duration_targets.float() * nonpadding

        loss = {}
        # calculate phonemes duration loss (log_d_predictions, attn_hard_dur)
        duration_targets.requires_grad = False
        log_duration_targets = torch.log(duration_targets + 1)
        pdur_loss = F.mse_loss(log_duration_predictions, log_duration_targets)
        loss["dur"] = pdur_loss

        return loss
    
    def _forward_align_losses(self, attn_matrix: torch.Tensor, input_lens: torch.Tensor, output_lens: torch.Tensor, step: int=0) -> dict:
        attn_soft, attn_hard, _, attn_logprob = attn_matrix

        loss = {}
        loss["ctc"] = self.sum_loss(attn_logprob=attn_logprob, in_lens=input_lens, out_lens=output_lens)
        if step < self.binarization_loss_enable_steps:
            bin_loss_weight = 0.
        else:
            bin_loss_weight = min((step - self.binarization_loss_enable_steps) / self.binarization_loss_warmup_steps, 1.0) * 1.0
        loss["bin"] = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft) * bin_loss_weight

        return loss

    def _forward_diff_losses(self, y_pred: torch.Tensor, u: torch.Tensor, mask: torch.Tensor=None) -> Dict:
        loss = {}
        if mask is None:
            mask = torch.ones((y_pred.size(0), 1, y_pred.size(-1)), device=y_pred.device)
        loss["diff"]= F.mse_loss(y_pred, u, reduction="sum") / (torch.sum(mask) * u.shape[1])
    
        return loss
    
    def _forward_prior_losses(self, feats: torch.Tensor, mu_y: torch.Tensor, mask: torch.Tensor=None) -> Dict:
        loss = {}
        if mask is None:
            mask = torch.ones((feats.size(0), 1, feats.size(-1)), device=feats.device)
            
        prior_loss = torch.sum(0.5 * ((feats - mu_y) ** 2 + math.log(2 * math.pi)) * mask)
        prior_loss = prior_loss / (torch.sum(mask) * self.n_channels)

        loss["prior"] = prior_loss

        return loss

    def _forward_cvae_losses(self, prob_: Tuple, step: int):
        x_mu_acc, x_logvar_acc, x_mu_spk, x_logvar_spk = prob_
        
        loss = {}
        L = self.L if isinstance(self.L, float) else self.L[step]
        acc_kl_loss = L * kl_loss(x_mu_acc, x_logvar_acc)
        spk_kl_loss = L * kl_loss(x_mu_spk, x_logvar_spk)
        loss["cvae"] = acc_kl_loss + spk_kl_loss

        return loss

    def forward(self, predictions: Tuple, targets: Tuple, is_joint: bool=False, step: int=None,**kwags) -> Dict:
        dec_outs, u, mu_y, text_mask, feats_mask, feats_lengths, duration_targets, log_duration_predictions, \
            attn_outs, accent_probs = predictions
        feats, text_lengths = targets
        
        text_mask = ~text_mask

        losses = self._forward_diff_losses(dec_outs, u=u, mask=feats_mask if is_joint==False else None)
        losses.update(self._forward_prior_losses(feats, mu_y, mask=feats_mask if is_joint==False else None))
        if step is not None:
            if attn_outs is not None:
                losses.update(self._forward_cvae_losses(
                    prob_ = accent_probs,
                    step  = step
                ))
                losses.update(self._forward_align_losses(
                    attn_matrix = attn_outs,
                    input_lens  = text_lengths,
                    output_lens = feats_lengths,
                    step        = step
                ))
                losses.update(self._forward_duration_losses(
                    log_duration_predictions = log_duration_predictions,
                    duration_targets         = duration_targets,
                    masks                    = text_mask
                ))

        return losses
