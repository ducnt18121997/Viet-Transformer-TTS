import os
import tqdm
import json
import numpy as np
import soundfile as sf
from typing import Dict

import torch
from torch.utils.data import Dataset

from modules.g2p import _symbols_to_sequence
from src.tools.tools_for_data import load_wav_to_torch, build_feat_extractor
from src.tools.utils import beta_binomial_prior_distribution, fix_len_compatibility


class MatchaLoader(Dataset):
    """ Dataloader for training jets model: built base on espnet"""
    def __init__(self, dataset_list: list, config: dict, speakers: dict, accents: dict, stats: dict = None, feat_extractor_choice: str="fbank", **kwags):

        self.inputs   = dataset_list
        self.speakers = speakers
        self.accents  = accents

        # initialize stft tranform
        self.config        = config
        self.sampling_rate = self.config["signal"]["sampling_rate"]
        self.n_mel_channel = self.config["mel"]["channels"]

        # created waveform variance inputs
        self.add_blank      = True
        self.feat_extractor = build_feat_extractor(feat_extractor_choice, config=self.config)
        self.feats_odim     = self.feat_extractor.output_size() 
        
        # initialize stats
        self.stats = self.compute_stats() if stats is None else stats

    def compute_stats(self) -> Dict:
        """ Generate data mean and standard deviation helpful in data normalisation """
        total_feats_sum, total_feats_len, total_feats_sq_sum = 0, 0, 0
        for line in tqdm.tqdm(self.inputs, desc="Creating prosody"):
            fpath = line[0]
            feats = self.get_feat(fpath)

            total_feats_len    += feats.size(0)
            total_feats_sum    += torch.sum(feats)
            total_feats_sq_sum += torch.sum(torch.pow(feats, 2))

        data_mean = total_feats_sum / (total_feats_len * self.feat_extractor.output_size())
        data_std  = torch.sqrt((total_feats_sq_sum / (total_feats_len * self.feat_extractor.output_size())) - torch.pow(data_mean, 2) )
        
        stats = {"mel_mean": data_mean.item(), "mel_std": data_std.item()}
        print(json.dumps(stats, indent=4))
        return stats

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        mu, std = self.stats["mel_mean"], self.stats["mel_std"]
        if not isinstance(mu, (float, int)):
            if isinstance(mu, list):
                mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
            elif isinstance(mu, torch.Tensor):
                mu = mu.to(data.device)
            elif isinstance(mu, np.ndarray):
                mu = torch.from_numpy(mu).to(data.device)
            mu = mu.unsqueeze(-1)

        if not isinstance(std, (float, int)):
            if isinstance(std, list):
                std = torch.tensor(std, dtype=data.dtype, device=data.device)
            elif isinstance(std, torch.Tensor):
                std = std.to(data.device)
            elif isinstance(std, np.ndarray):
                std = torch.from_numpy(std).to(data.device)
            std = std.unsqueeze(-1)

        return (data - mu) / std
    
    def get_feat(self, fid) -> torch.Tensor:
        feat_location = os.path.join(os.path.dirname(os.path.dirname(fid)), "feats", f"{'.'.join(os.path.basename(fid).split('.')[: -1])}.pt")

        if os.path.exists(feat_location):
            feats = torch.load(feat_location).squeeze()
        else:
            audio, sr = load_wav_to_torch(fid)
            if sr != self.config["signal"]["sampling_rate"]:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sr, self.config["signal"]["sampling_rate"]))
            audio = audio.unsqueeze(0)
            feats, _ = self.feat_extractor(audio)
            torch.save(feats.squeeze().detach().cpu(), feat_location)
        
        return feats

    def get_accent(self, fid: str) -> torch.Tensor:
        acc = None
        if self.accents is not None:
            acc = os.path.basename(fid).split("-")[0]
            acc = torch.IntTensor([self.accents[acc]])
        else:
            acc = torch.IntTensor([0 if fid.split("_")[0] == "hn" else 1])

        return acc

    def get_text(self, pid) -> torch.Tensor:
        text_norm = _symbols_to_sequence(pid)
        # if self.add_blank:
        #     text_norm = intersperse(text_norm, 0)
        
        return torch.LongTensor(text_norm)

    def get_alignment(self, fid, phoneme_count, mel_count) -> torch.Tensor:
        aligment_location = os.path.join(os.path.dirname(os.path.dirname(fid)), "alignment", f"{'.'.join(os.path.basename(fid).split('.')[: -1])}.npy")

        if os.path.exists(aligment_location):
            alignment = np.load(aligment_location)
        else:
            alignment = beta_binomial_prior_distribution(phoneme_count, mel_count)
            os.makedirs(os.path.dirname(aligment_location), exist_ok=True)
            np.save(aligment_location, alignment)
        
        # print(f"{alignment.shape} - {phoneme_count} - {mel_count}")

        return torch.from_numpy(alignment)

    def get_speaker(self, sid: str) -> torch.Tensor:
        if self.speakers is not None:
            emb = torch.LongTensor([self.speakers[sid]])
        else:
            # NOTE: sid is considered to fid
            raise NotImplementedError("Not supported for speaker embeddings yet")

        return emb

    def get_audio(self, fid) -> torch.Tensor:
        audio, sr = sf.read(fid)
        assert sr == self.sampling_rate, (
            f"Audio sample rate missmatch: given {sr} Hz, expected {self.sampling_rate} Hz"
        )

        return torch.FloatTensor(audio).unsqueeze(0)

    def __getitem__(self, index):
        pid  = self.get_text(self.inputs[index][2])
        feat = self.normalize(self.get_feat(self.inputs[index][0])).transpose(0, 1)
        
        return dict(
            sid    = self.get_speaker(self.inputs[index][1]),
            aid    = self.get_accent(self.inputs[index][0] if self.accents is not None else self.inputs[index][1]),
            pid    = pid,
            feat   = feat, 
            align  = self.get_alignment(self.inputs[index][0], len(pid), feat.size(1)).transpose(0, 1),
            speech = self.get_audio(self.inputs[index][0])
        )

    def __len__(self):
        return len(self.inputs)


class MatchaCollate():
    """ Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, n_speakers: int, use_accent: bool=False, return_waveform: bool=True, *kwags):
        
        self.use_accent      = use_accent
        self.n_speakers      = n_speakers
        self.return_waveform = return_waveform

    def pad_text(self, pid):
        # Right zero-pad all one-hot pid sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x) for x in pid]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]
        text_padded = torch.zeros(len(pid), max_input_len).long()

        for i in range(len(ids_sorted_decreasing)):
            text_padded[i, :pid[ids_sorted_decreasing[i]].size(0)] = pid[ids_sorted_decreasing[i]]

        return text_padded, input_lengths, ids_sorted_decreasing, max_input_len
    
    def pad_2D(self, x, ids_sorted_decreasing, fix: bool=True):
        # right zero-pad mel-spec
        axis_0_maxlen = max([_.size(0) for _ in x])
        axis_1_maxlen = max([_.size(1) for _ in x])
        if fix:
            axis_1_maxlen = fix_len_compatibility(axis_1_maxlen)

        # mel padded
        x_padded  = torch.zeros(len(x), axis_0_maxlen, axis_1_maxlen)
        x_lengths = torch.LongTensor(len(x))

        for i in range(len(ids_sorted_decreasing)):
            x_padded[i, : x[ids_sorted_decreasing[i]].size(0), : x[ids_sorted_decreasing[i]].size(1)] = x[ids_sorted_decreasing[i]]
            x_lengths[i] = x[ids_sorted_decreasing[i]].size(1)

        return x_padded, x_lengths

    def __call__(self, batch):
        """ Collate"s training batch from normalized pid and mel-spectrograms """
        
        texts = [x["pid"] for x in batch]
        text_padded, text_lengths, ids_sorted_decreasing, max_text_lengths = self.pad_text(texts)
        
        feats    = [x["feat"] for x in batch]
        feats_padded, feat_lengths = self.pad_2D(feats, ids_sorted_decreasing)
        speechs  = [x["speech"] for x in batch]
        if self.return_waveform:
            speechs_padded, speech_lengths = self.pad_2D(speechs, ids_sorted_decreasing, fix=False)
        else:
            speechs_padded = None

        alignment = [x["align"] for x in batch]
        alignment = self.pad_2D(alignment, ids_sorted_decreasing)[0].transpose(1, 2)

        sids = torch.LongTensor([batch[idx]["sid"] for idx in ids_sorted_decreasing])
        aids = None
        if self.use_accent:
            aids = torch.LongTensor([batch[idx]["aid"] for idx in ids_sorted_decreasing])

        return dict(
            text           = text_padded,
            text_lengths   = text_lengths,
            feats          = feats_padded,
            feats_lengths  = feat_lengths,
            sids           = sids,
            aids           = aids,
            duration       = alignment
        ), (speechs_padded, feats_padded, text_lengths)
