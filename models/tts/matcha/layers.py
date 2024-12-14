import math
from abc import ABC
from typing import Any, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tts.matcha.sub_layers import ConvReluNorm, Encoder, Decoder
from models.tts.matcha.function import sequence_mask


class TextEncoder(nn.Module):
    """ Text Encoder """

    def __init__(self, 
        n_vocab: int,
        n_feats: int,
        n_spks: int,
        spk_emb_dim: int, 
        encoder_params: Dict,
    ):
        super().__init__()

        # initialize configuration
        self.encoder_params = encoder_params
        self.n_vocab = n_vocab
        self.n_feats = n_feats
        self.n_spks  = n_spks

        # initialize layer
        self.hidden_dim   = self.encoder_params["hidden_dim"]
        self.spk_emb_dim  = spk_emb_dim
        self.src_word_emb = nn.Embedding(
            num_embeddings = self.n_vocab,
            embedding_dim  = self.hidden_dim
        )
        nn.init.normal_(self.src_word_emb.weight, 0.0, self.hidden_dim ** -0.5)

        self.use_prenet = self.encoder_params["use_prenet"]
        if self.use_prenet:
            self.prenet = ConvReluNorm(
                in_channels     = self.hidden_dim,
                hidden_channels = self.hidden_dim,
                out_channels    = self.hidden_dim,
                **self.encoder_params["prenet"]
            )
        else:
            self.prenet = lambda x, x_mask: x

        self.hidden_dim = self.hidden_dim + (self.spk_emb_dim if self.n_spks > 1 else 0)
        self.encoder = Encoder(
            hidden_channels = self.hidden_dim,
            **self.encoder_params["encoder"]
        )

        self.proj_m = nn.Conv1d(
            in_channels  = self.hidden_dim, 
            out_channels = self.n_feats, 
            kernel_size  = 1
        )

    def forward(
        self, 
        src_seq: torch.Tensor, 
        src_lengths: torch.Tensor, 
        sids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass to the transformer based encoder and duration predictor

        Args:
            src_seq (torch.Tensor): text input
                shape: (batch_size, max_text_length)
            src_lengths (torch.Tensor): text input lengths
                shape: (batch_size,)
            sids (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size,)

        Returns:
            mu (torch.Tensor): average output of the encoder
                shape: (batch_size, n_feats, max_text_length)
            x_mask (torch.Tensor): mask for the text input
                shape: (batch_size, 1, max_text_length)
        """
        src_word_emb = self.src_word_emb(src_seq) * math.sqrt(self.hidden_dim)
        src_word_emb = torch.transpose(src_word_emb, 1, -1)
        src_mask = torch.unsqueeze(sequence_mask(src_lengths, src_word_emb.size(2)), 1).to(src_word_emb.dtype)

        enc_outs = self.prenet(src_word_emb, src_mask)
        enc_outs = torch.cat([enc_outs, sids.unsqueeze(-1).repeat(1, 1, enc_outs.shape[-1])], dim=1)
        enc_outs = self.encoder(enc_outs, src_mask)
        mu = self.proj_m(enc_outs) * src_mask

        return  mu, src_word_emb, src_mask.squeeze(1).bool()


class BASECFM(nn.Module, ABC):
    def __init__(
        self,
        n_feats: int,
        n_spks: int,
        spk_emb_dim: int,
        cfm_params: Dict,
    ):
        super().__init__()
        
        self.cfm_params = cfm_params
        self.n_feats = n_feats
        self.n_spks  = n_spks

        self.spk_emb_dim = spk_emb_dim
        self.solver      = self.cfm_params["solver"]
        if hasattr(self.cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None

    def inference(
        self, 
        mu: torch.Tensor, 
        mask: torch.Tensor, 
        n_timesteps: int, 
        temperature: float=1.0, 
        spks: Optional[torch.Tensor]=None, 
        cond: Any = None
    ):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    def solve_euler(
        self, 
        x: torch.Tensor, 
        t_span: torch.Tensor, 
        mu: torch.Tensor, 
        mask: torch.Tensor, 
        spks: torch.Tensor, 
        cond: Any
    ):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        steps = 1
        while steps <= len(t_span) - 1:
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if steps < len(t_span) - 1:
                dt = t_span[steps + 1] - t
            steps += 1

        return sol[-1]

    def forward(
        self, 
        x1: torch.Tensor, 
        mask: torch.Tensor, 
        mu: torch.Tensor, 
        spks: Optional[torch.Tensor]=None, 
        cond: Any=None
    ):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z
        
        y_pred = self.estimator(y, mask, mu, t.squeeze(), spks)

        return y_pred, u


class CFM(BASECFM):
    def __init__(
        self, 
        in_channels: int, 
        out_channel: int, 
        n_spks: int, 
        spk_emb_dim: int,
        cfm_params: Dict, 
        decoder_params: Dict
    ):
        super().__init__(n_feats=in_channels, n_spks=n_spks, spk_emb_dim=spk_emb_dim, cfm_params=cfm_params)
        
        self.spk_emb_dim = spk_emb_dim
        self.in_channels = in_channels + self.spk_emb_dim
        self.out_channel = out_channel

        # Just change the architecture of the estimator here
        self.estimator = Decoder(
            in_channels  = self.in_channels, 
            out_channels = self.out_channel, 
            **decoder_params
        )
