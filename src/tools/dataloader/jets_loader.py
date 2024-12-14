import os
import tqdm
import json
import random
import numpy as np
import soundfile as sf

import torch
from torch.utils.data import Dataset

from modules.g2p import _symbols_to_sequence
from src.tools.tools_for_data import prepare_inputs


class JETSLoader(Dataset):
    """ Dataloader for training jets model: built base on espnet"""
    def __init__(self, dataset_list: list, config: dict, speakers: dict, stats: dict = None, feat_extractor_choice: str="fbank"):
        
        self.speakers = speakers
        self.inputs = dataset_list
        random.shuffle(self.inputs)

        # initialize stft tranform
        self.config        = config
        self.sampling_rate = self.config["signal"]["sampling_rate"]
        self.n_mel_channel = self.config["mel"]["channels"]

        # created waveform variance inputs
        self.prosody_path, self.feats_odim = prepare_inputs(
            list_segments          = self.inputs, 
            hparams                = self.config,
            feat_extractor_choice = feat_extractor_choice
        )
        self.stats = self.compute_stats() if stats is None else stats

    def compute_stats(self):
        scaler = {
            "pitch" : {"sum": 0, "sum_square": 0, "count": 0},
            "energy": {"sum": 0, "sum_square": 0, "count": 0}
        } 
        # implement of global mvn
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
        print(json.dumps(stats, ensure_ascii=False, indent=4))
    
        return stats

    def get_feat(self, fid):

        return torch.load(self.prosody_path[fid]["feat"])

    def get_text(self, pid):

        return torch.IntTensor(_symbols_to_sequence(pid))

    def get_prosody(self, fid, _type):
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

    def get_audio(self, fid):
        audio, sr = sf.read(self.prosody_path[fid]["wav"])
        assert sr == self.sampling_rate, (
            f"Audio sample rate missmatch: given {sr} Hz, expected {self.sampling_rate} Hz"
        )

        return torch.FloatTensor(audio).unsqueeze(0)

    def __getitem__(self, index):
        fid = f"{self.inputs[index][1]}_{os.path.basename(self.inputs[index][0])}"

        return dict(
            sid    = self.get_speaker(self.inputs[index][1] if self.speakers is not None else fid),
            pid    = self.get_text(self.inputs[index][2]),
            feat   = self.get_feat(fid).transpose(0, 1), 
            speech = self.get_audio(fid),
            pitch  = self.get_prosody(fid, "pitch"),
            energy = self.get_prosody(fid, "energy")
        )

    def __len__(self):
        return len(self.inputs)


class JETSCollate():
    """ Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, n_speakers: int):
        
        self.n_speakers      = n_speakers

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

    def pad_feat(self, feats, ids_sorted_decreasing):
        # Right zero-pad mel-spec
        n_feats      = feats[0].size(0)
        max_feat_len = max([x.size(1) for x in feats])
        # mel padded
        feat_padded  = torch.zeros(len(feats), n_feats, max_feat_len)
        feat_lengths = torch.LongTensor(len(feats))

        for i in range(len(ids_sorted_decreasing)):
            feat_padded[i, :, :feats[ids_sorted_decreasing[i]].size(1)] = feats[ids_sorted_decreasing[i]]
            feat_lengths[i] = feats[ids_sorted_decreasing[i]].size(1)

        return feat_padded, feat_lengths, max(feat_lengths)

    def pad_prosody(self, prosodies, max_len, ids_sorted_decreasing):
        prosody_padded = torch.zeros(len(prosodies), max_len)

        for i in range(len(ids_sorted_decreasing)):
            prosody_padded[i, :prosodies[ids_sorted_decreasing[i]].size(0)] = prosodies[ids_sorted_decreasing[i]]

        return prosody_padded
        
    def __call__(self, batch):
        """ Collate"s training batch from normalized pid and mel-spectrograms """
        texts = [x["pid"] for x in batch]
        text_padded, text_lengths, ids_sorted_decreasing, max_text_lengths = self.pad_text(texts)
        
        if self.n_speakers == -1: 
            sids   = None
            spembs = torch.stack([batch[idx]["sid"] for idx in ids_sorted_decreasing])
        else:
            sids   = None
            spembs = torch.LongTensor([batch[idx]["sid"] for idx in ids_sorted_decreasing])

        feats    = [x["feat"] for x in batch]
        feats_padded, feat_lengths, max_feat_lengths = self.pad_feat(feats, ids_sorted_decreasing)
        
        speechs  = [x["speech"] for x in batch]
        speechs_padded, speech_lengths, max_speech_lengths = self.pad_feat(speechs, ids_sorted_decreasing)
        
        pitches       = [x["pitch"] for x in batch]
        pitch_padded  = self.pad_prosody(pitches, max_feat_lengths, ids_sorted_decreasing)
        energies      = [x["energy"] for x in batch]
        energy_padded = self.pad_prosody(energies, max_feat_lengths, ids_sorted_decreasing)

        return dict(
            text           = text_padded,
            text_lengths   = text_lengths,
            feats          = feats_padded.transpose(1, 2),
            feats_lengths  = feat_lengths,
            pitch          = pitch_padded.unsqueeze(-1),
            pitch_lengths  = feat_lengths,
            energy         = energy_padded.unsqueeze(-1),
            energy_lengths = feat_lengths,
            sids           = sids,
            spembs         = spembs
        ), speechs_padded
