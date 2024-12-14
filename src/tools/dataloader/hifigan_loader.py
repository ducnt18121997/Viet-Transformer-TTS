import os
import random
from typing import Dict
from librosa.util import normalize

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from espnet2.gan_tts.utils import get_random_segments, get_segments

from src.tools.tools_for_data import load_wav_to_torch, build_feat_extractor
from src.tools.utils import extract_embedding


class HifiGanLoader(Dataset):
    """ Dataloader for training vocodear model"""
    def __init__(self, data_list: str, config: dict, use_speaker: bool=True):
        self.inputs = data_list
        random.shuffle(self.inputs)

        self.config       = config
        from espnet2.tts.feats_extract.log_mel_fbank import LogMelFbank
        self.wav_to_mel = LogMelFbank(
            fs         = self.config["signal"]["sampling_rate"],
            n_fft      = self.config["stft"]["filter_length"],
            win_length = self.config["stft"]["win_length"],
            hop_length = self.config["stft"]["hop_length"],
            window     = self.config["stft"]["window"],
            n_mels     = self.config["mel"]["channels"],
            fmin       = self.config["mel"]["fmin"],
            fmax       = self.config["mel"]["fmax"],
            log_base   = self.config["mel"]["log_base"]
        )

        self.hop_size = self.config["stft"]["hop_length"]
        self.use_speaker = use_speaker

    def __getitem__(self, index):
        fid = self.inputs[index]
        audio, sampling_rate = load_wav_to_torch(fid[0])
        if sampling_rate != self.config["signal"]["sampling_rate"]:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.config["signal"]["sampling_rate"]))

        # degrade audio if doesn't use mel from acoustic models
        audio  = torch.FloatTensor(normalize(audio.numpy() * 0.95)).unsqueeze(0)
        mel, _ = self.wav_to_mel(audio)

        return dict(speech=audio, melspec=mel.transpose(1, 2).squeeze())

    def __len__(self):
        return len(self.inputs)


class HifiGanCollate():
    """ Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, segment_size: int=32, upsample_factor: int=256):

        self.segment_size    = segment_size
        self.upsample_factor = upsample_factor
    
    def pad_feat(self, feats):
        # right zero-pad features
        n_feats      = feats[0].size(0)
        max_feat_len = max([x.size(1) for x in feats])
        
        # features padded
        feat_padded  = torch.zeros(len(feats), n_feats, max_feat_len)
        feat_lengths = torch.LongTensor(len(feats))

        for i in range(len(feat_padded)):
            feat_padded[i, :, :feats[i].size(1)] = feats[i]
            feat_lengths[i] = feats[i].size(1)

        return feat_padded, feat_lengths

    def __call__(self, batch): 
        melspec_padded, melspec_lengths = self.pad_feat([x["melspec"] for x in batch])
        speech_padded, _   = self.pad_feat([x["speech"] for x in batch])
        semb_padded = [x["semb"] for x in batch]

        melspec_padded, start_idxs = get_random_segments(
            x            = melspec_padded,
            x_lengths    = melspec_lengths, 
            segment_size = self.segment_size
        )
        speech_padded  = get_segments(
            x            = speech_padded,
            start_idxs   = start_idxs * self.upsample_factor,
            segment_size = self.segment_size * self.upsample_factor
        )
        if semb_padded[0] is not None:
            semb_padded = torch.stack(semb_padded, dim=0).float().unsqueeze(-1)
        else:
            semb_padded = None

        return melspec_padded, speech_padded, semb_padded

