import math
from typing import List
import torch
import torch.nn as nn
from espnet2.gan_tts.vits import monotonic_align
from espnet2.gan_tts.utils import get_random_segments
from models.gan_tts.vits2.layers import TextEncoder, PosteriorEncoder, Generator, ResidualCouplingTransformersBlock
from models.gan_tts.vits2.utils import generate_path, sequence_mask
AVAILABLE_FLOW_TYPES = [
    "pre_conv",
    "pre_conv2",
    "fft",
    "mono_layer_inter_residual",
    "mono_layer_post_residual",
]
AVAILABLE_DURATION_DISCRIMINATOR_TYPES = [
    "dur_disc_1",
    "dur_disc_2",
]


class VITS2(nn.Module):
    """VITS2 module.

    This is a module of VITS2 described in `VITS2: Improving 
    Quality and Efficiency of Single-Stage Text-to-Speech with 
    Adversarial Learning and Architecture Design`_.

    .. _`VITS2: Improving Quality and Efficiency of Single-Stage Text-
    to-Speech with Adversarial Learning and Architecture Design`:
        https://arxiv.org/abs/2307.16430

    """

    def __init__(
        self,
        n_vocab: int,
        spec_channels: int,
        n_speakers: int = 0,
        segment_size: int = 8192,
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        resblock: str = "1",
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        gin_channels: int = 0,
        use_sdp: bool = True,
        use_transformer_flows: bool = True,
        transformer_flow_type: str = "pre_conv",
        use_spk_conditioned_encoder: bool = False,
        use_noise_scaled_mas: bool = True,
        mas_noise_scale_initial: float = 0.01,
        noise_scale_delta: float = 2e-6,
        **kwargs
    ) -> None:
        super(VITS2, self).__init__()
        """ Initialize VITS2 module. """

        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_spk_conditioned_encoder = use_spk_conditioned_encoder
        self.use_transformer_flows = use_transformer_flows
        self.transformer_flow_type = transformer_flow_type
        if self.use_transformer_flows:
            assert (
                self.transformer_flow_type in AVAILABLE_FLOW_TYPES
            ), f"transformer_flow_type must be one of {AVAILABLE_FLOW_TYPES}"
        self.use_sdp = use_sdp
        # self.use_duration_discriminator = kwargs.get("use_duration_discriminator", False)
        self.use_noise_scaled_mas = use_noise_scaled_mas
        self.mas_noise_scale_initial = mas_noise_scale_initial
        self.noise_scale_delta = noise_scale_delta

        self.current_mas_noise_scale = self.mas_noise_scale_initial
        if self.use_spk_conditioned_encoder and gin_channels > 0:
            self.enc_gin_channels = gin_channels
        else:
            self.enc_gin_channels = 0

        # initialize encoder
        self.enc_p = TextEncoder(
            n_vocab=self.n_vocab,
            out_channels=self.inter_channels,
            hidden_channels=self.hidden_channels,
            filter_channels=self.filter_channels,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout,
            gin_channels=self.enc_gin_channels
        )
        self.enc_q = PosteriorEncoder(
            in_channels=self.spec_channels,
            out_channels=self.inter_channels,
            hidden_channels=self.hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
            gin_channels=self.gin_channels
        )
        self.emb_g = nn.Embedding(
            num_embeddings=self.n_speakers, 
            embedding_dim=self.gin_channels
        )

        # initalize duration modelling
        if self.use_sdp:
            from models.gan_tts.vits2.layers import StochasticDurationPredictor
            self.dp = StochasticDurationPredictor(
                in_channels=self.hidden_channels, 
                filter_channels=192,
                kernel_size=3, 
                p_dropout=0.5, 
                n_flows=4, 
                gin_channels=self.gin_channels
            )
        else:
            from models.gan_tts.vits2.layers import DurationPredictor
            self.dp = DurationPredictor(
                in_channels=self.hidden_channels, 
                filter_channels=256, 
                kernel_size=3, 
                p_dropout=0.5, 
                gin_channels=self.gin_channels
            )

        # initialize decoder
        self.dec = Generator(
            initial_channel=self.inter_channels,
            resblock=self.resblock,
            resblock_kernel_sizes=self.resblock_kernel_sizes,
            resblock_dilation_sizes=self.resblock_dilation_sizes,
            upsample_rates=self.upsample_rates,
            upsample_initial_channel=self.upsample_initial_channel,
            upsample_kernel_sizes=self.upsample_kernel_sizes,
            gin_channels=self.gin_channels
        )
        self.flow = ResidualCouplingTransformersBlock(
            channels=self.inter_channels,
            hidden_channels=self.hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
            gin_channels=self.gin_channels,
            use_transformer_flows=self.use_transformer_flows,
            transformer_flow_type=self.transformer_flow_type
        )

    def forward(
        self, 
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        sids: torch.Tensor=None,
        *kwags
    ):
        g = self.emb_g(sids).unsqueeze(-1)  # [b, h, 1]

        x, m_p, logs_p, x_mask = self.enc_p(text, text_lengths, g=g)
        z, m_q, logs_q, y_mask = self.enc_q(feats, feats_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)  # [b, 1, t_s]
            neg_cent2 = torch.matmul(-0.5 * (z_p**2).transpose(1, 2), s_p_sq_r)  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True)  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            if self.use_noise_scaled_mas:
                epsilon = (torch.std(neg_cent) * torch.randn_like(neg_cent) * self.current_mas_noise_scale)
                neg_cent = neg_cent + epsilon
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        w = attn.sum(2)
        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
            logw  = self.dp(x, x_mask, g=g, reverse=True, noise_scale=1.0)
            logw_ = torch.log(w + 1e-6) * x_mask
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw  = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)  # for averaging

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = get_random_segments(
            z, feats_lengths, self.segment_size // self.dec.upsample_factor
        )
        o = self.dec(z_slice, g=g)

        return (
            o,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            (x, logw, logw_),
        )

    def inference(
        self, 
        sids: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        d_control: float=1, # length_scale
        noise_scale: int=0.0,
        noise_scale_w: float=0.0,
        max_len: int=None,
        **kwargs
    ):
        g = self.emb_g(sids).unsqueeze(-1)  # [b, h, 1]
        x, m_p, logs_p, x_mask = self.enc_p(text, text_lengths, g=g)
        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * d_control
        w_ceil = torch.ceil(w)
        feats_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(sequence_mask(feats_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2))\
            .transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2))\
            .transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        outs = self.dec((z * y_mask)[:, :, :max_len], g=g)
        out_lengths = feats_lengths * self.dec.upsample_factor
        # return o, attn, y_mask, (z, z_p, m_p, logs_p)

        return outs, out_lengths
