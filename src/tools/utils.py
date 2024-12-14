import os
import torch
import requests
import numpy as np
import soundfile as sf
from scipy.stats import betabinom
from modules.g2p import _symbols_to_sequence 


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def load_wav_to_torch(path: str) -> torch.Tensor:
    # espnet use soundfile
    data, sampling_rate = sf.read(path)

    return torch.FloatTensor(data), sampling_rate


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling_factor=1.0):
    P, M = phoneme_count, mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M + 1):
        a, b = scaling_factor * i, scaling_factor * (M + 1 - i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)

    return np.array(mel_text_probs)


def extract_embedding(path: str) -> np.array:
    URL = "https://speech.aiservice.vn/speaker_verify/v1.0/get_embed"
    res = requests.post(URL, files=[("file", (os.path.basename(path), open(path, "rb"), "audio/wav"))])    

    if res.status_code == 200:
        emb = np.array(res.json()["embed"])
    else:
        emb = None

    return emb


def fix_len_compatibility(length: torch.Tensor, num_downsamplings_in_unet: int=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1
