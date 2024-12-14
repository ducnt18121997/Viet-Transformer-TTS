import os
import tqdm
import random
import numpy as np
import soundfile as sf

import torch
from torch.utils.data import Dataset

from modules.g2p import _symbols_to_sequence
from src.tools.tools_for_data import prepare_inputs


class Fastspeech2Loader(Dataset):
    """ Dataloader for training Fastspeech2 model"""
    def __init__(self, dataset_list: list, config: dict, speakers: dict, accents:dict, stats: dict=None, feat_extractor_choice: str="fbank"):
        
        self.speakers = speakers
        self.accents  = accents
        self.inputs   = dataset_list
        random.shuffle(self.inputs)

        # initialize stft tranform
        self.config        = config
        self.sampling_rate = self.config["signal"]["sampling_rate"]
        self.n_mel_channel = self.config["mel"]["channels"]

        # created waveform variance inputs
        self.prosody_path, self.feats_odim = prepare_inputs(
            list_segments         = self.inputs, 
            hparams               = self.config,
            feat_extractor_choice = feat_extractor_choice
        )
        self.stats = self.compute_stats() if stats is None else stats

    def compute_stats(self) -> dict:
        scaler = {
            "pitch" : {"sum": 0, "sum_square": 0, "count": 0},
            "energy": {"sum": 0, "sum_square": 0, "count": 0}
        } 
        # re-implement global mvn
        list_segments = ["_".join([_[1], os.path.basename(_[0])]) for _ in self.inputs]
        for fid in tqdm.tqdm(list_segments, desc="Computing statistic quantities"):
            for pros in scaler:
                values = torch.load(self.prosody_path[fid][pros])
                scaler[pros]["sum"]        += values.sum(0)
                scaler[pros]["sum_square"] += (values**2).sum(0)
                scaler[pros]["count"]      += len(values)

        stats = {"pitch": {}, "energy": {}}
        for pros in scaler:
            _mean = scaler[pros]["sum"] / scaler[pros]["count"]
            _var  = scaler[pros]["sum_square"] / scaler[pros]["count"] - _mean * _mean

            stats[pros]["mean"] = _mean.item()
            stats[pros]["std"]  = np.sqrt(np.maximum(_var.item(), 1.0e-20))

        energy_max = pitch_max = None
        energy_min = pitch_min = None
        for fid in tqdm.tqdm(list_segments, desc="Build model stats"):
            # build pitch stats
            values = torch.load(self.prosody_path[fid]["pitch"])
            values = (values - stats["pitch"]["mean"]) / stats["pitch"]["std"]
            pitch_max = torch.max(values) if pitch_max is None else max(pitch_max, torch.max(values))
            pitch_min = torch.min(values) if pitch_min is None else min(pitch_min, torch.min(values))

            # build energy stats
            values = torch.load(self.prosody_path[fid]["energy"])
            values = (values - stats["energy"]["mean"]) / stats["energy"]["std"]
            energy_max = torch.max(values) if energy_max is None else max(energy_max, torch.max(values))
            energy_min = torch.min(values) if energy_min is None else min(energy_min, torch.min(values))

        stats["pitch"].update(dict(max=pitch_max.item(), min=pitch_min.item()))
        stats["energy"].update(dict(max=energy_max.item(), min=energy_min.item()))

        return stats

    def get_feat(self, fid: str) -> torch.Tensor:

        return torch.load(self.prosody_path[fid]["feat"])

    def get_text(self, pid: str) -> torch.Tensor:

        return torch.IntTensor(_symbols_to_sequence(pid))

    def get_boundary(self, bound: str) -> torch.Tensor:

        return [int(_) for _ in bound.split(", ")]

    def get_prosody(self, fid: str, _type: str) -> torch.Tensor:
        pros = torch.load(self.prosody_path[fid][_type])
        pros = (pros - self.stats[_type]["mean"]) / self.stats[_type]["std"]

        return pros

    def get_speaker(self, sid: str) -> torch.Tensor:
        if self.speakers is not None:
            emb = torch.IntTensor([self.speakers[sid]])
        else:
            # NOTE: sid is considered to fid
            emb = torch.load(self.prosody_path[sid]["embedding"])[0]
            emb = torch.FloatTensor(emb)

        return emb
            
    def get_accent(self, fid: str) -> torch.Tensor:
        acc = None
        if self.accents is not None:
            acc = os.path.basename(fid).split("-")[0]
            acc = torch.IntTensor([self.accents[acc]])
        else:
            acc = torch.IntTensor([0 if fid.split("_")[0] == "hn" else 1])

        return acc

    def get_audio(self, fid: str) -> torch.Tensor:
        audio, sr = sf.read(self.prosody_path[fid]["wav"])
        assert sr == self.sampling_rate, (
            f"Audio sample rate missmatch: given {sr} Hz, expected {self.sampling_rate} Hz"
        )

        return torch.FloatTensor(audio).unsqueeze(0)

    def get_duration(self, fid: str) -> torch.Tensor:
        with open(self.prosody_path[fid]["duration"], "r", encoding="utf8") as f:
            dur = [int(_) for _ in f.read().strip().split(", ")]

        return torch.IntTensor(dur)

    def get_alignment(self, fid: str) -> torch.Tensor:
        align = np.load(self.prosody_path[fid]["alignment"])

        return torch.from_numpy(align)

    def __getitem__(self, index):
        fid = f"{self.inputs[index][1]}_{os.path.basename(self.inputs[index][0])}"

        text = self.get_text(self.inputs[index][2])
        feat = self.get_feat(fid).transpose(0, 1)

        return dict(
            sid      = self.get_speaker(self.inputs[index][1] if self.speakers is not None else fid),
            aid      = self.get_accent(self.inputs[index][0] if self.accents is not None else self.inputs[index][1]),
            pid      = text,
            bid      = self.get_boundary(self.inputs[index][3]),
            feat     = feat, 
            speech   = self.get_audio(fid),
            duration = self.get_alignment(fid) if self.config["self_learning"] is True else self.get_duration(fid),
            pitch    = self.get_prosody(fid, "pitch"),
            energy   = self.get_prosody(fid, "energy")
        )

    def __len__(self):
        return len(self.inputs)


class Fastspeech2Collate():
    """ Zero-pads model inputs and targets based on number of frames per step"""
    def __init__(self, n_speakers: int, use_accent: bool, return_waveform: bool=True):
        
        self.n_speakers       = n_speakers
        self.use_accent       = use_accent
        self.return_waveform  = return_waveform

    def pad_2D(self, x, ids_sorted_decreasing):
        
        # right zero-pad mel-spec
        axis_0_maxlen = max([_.size(0) for _ in x])
        axis_1_maxlen = max([_.size(1) for _ in x])

        # mel padded
        x_padded  = torch.zeros(len(x), axis_0_maxlen, axis_1_maxlen)
        x_lengths = torch.LongTensor(len(x))

        for i in range(len(ids_sorted_decreasing)):
            x_padded[i, : x[ids_sorted_decreasing[i]].size(0), : x[ids_sorted_decreasing[i]].size(1)] = x[ids_sorted_decreasing[i]]
            x_lengths[i] = x[ids_sorted_decreasing[i]].size(1)

        return x_padded, x_lengths

    def pad_1D(self, x, ids_sorted_decreasing):
        max_len   = max([_.size(0) for _ in x])

        x_padded  = torch.zeros(len(x), max_len)
        x_lengths = torch.LongTensor(len(x_padded))
        for i in range(len(ids_sorted_decreasing)):
            x_padded[i, : x[ids_sorted_decreasing[i]].size(0)] = x[ids_sorted_decreasing[i]]
            x_lengths[i] = x[ids_sorted_decreasing[i]].size(0)
            
        return x_padded, x_lengths
        
    def __call__(self, batch):
        """ Collate"s training batch from normalized pid and mel-spectrograms """
        texts    = [x["pid"] for x in batch]
        feats    = [x["feat"] for x in batch]
        speech   = [x["speech"] for x in batch]
        duration = [x["duration"] for x in batch]
        pitch    = [x["pitch"] for x in batch]
        energy   = [x["energy"] for x in batch]
        
        # sort by legth (by text or by feats)
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([fea.size()[1] for fea in feats]),
            dim=0, descending=True
        )        
        aids = None
        if self.use_accent:
            aids = torch.LongTensor([batch[idx]["aid"] for idx in ids_sorted_decreasing])
        
        if self.n_speakers == -1: 
            sids = torch.stack([batch[idx]["sid"] for idx in ids_sorted_decreasing])
        else:
            sids = torch.LongTensor([batch[idx]["sid"] for idx in ids_sorted_decreasing])

        bids = [batch[idx]["bid"] for idx in ids_sorted_decreasing]
        text_padded, text_lengths  = self.pad_1D(texts, ids_sorted_decreasing)
        feats_padded, feat_lengths = self.pad_2D(feats, ids_sorted_decreasing)
        feats_padded = feats_padded.transpose(1, 2)
        if self.return_waveform:
            speech_padded, speech_lengths = self.pad_2D(speech, ids_sorted_decreasing)
        else:
            speech_padded, speech_lengths = None, None

        pitch_padded, _  = self.pad_1D(pitch, ids_sorted_decreasing)
        energy_padded, _ = self.pad_1D(energy, ids_sorted_decreasing)

        if len(duration[0].size()) == 2:
            duration_padded, _ = self.pad_2D(duration, ids_sorted_decreasing)
        else:
            duration_padded, _ = self.pad_1D(duration, ids_sorted_decreasing)
            duration_padded = duration_padded.long()

        return dict(
            text           = text_padded.long(),
            text_lengths   = text_lengths,
            feats          = feats_padded,
            feats_lengths  = feat_lengths,
            duration       = duration_padded,
            pitch          = pitch_padded,
            energy         = energy_padded,
            sids           = sids,
            aids           = aids
        ), (speech_padded, feats_padded, bids)
