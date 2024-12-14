import os
import random
from typing import Tuple, List

import torch
from torch.utils.data import Dataset

from modules.g2p import _symbols_to_sequence
from src.tools.tools_for_data import load_wav_to_torch, build_feat_extractor


class VITS2Loader(Dataset):
    """ Dataloader for training vocodear model"""
    def __init__(self, dataset_list: list, config: dict, speakers: dict):
        
        self.inputs   = dataset_list
        random.shuffle(self.inputs)
        self.config   = config
        self.speakers = speakers
        
        # NOTE(by deanng):
        #   + Using mel posterior encoder for VITS2 (fbank)
        #   + Using lin posterior encoder for VITS1 (linear_spectrogram)
        self.spec_extractor = build_feat_extractor(choice="fbank", config=self.config)
        self.feats_odim     = self.spec_extractor.output_size()
        self.max_wav_value  = self.config["signal"]["max_wav_value"]

    def get_speaker(self, spk):

        return torch.IntTensor([self.speakers[spk]])

    def get_text(self, pid: str) -> torch.Tensor:
        text_norm = intersperse(_symbols_to_sequence(pid))

        return torch.IntTensor(text_norm)

    def get_audio(self, fname: str) -> Tuple[torch.Tensor, torch.Tensor]:
        audio, sr = load_wav_to_torch(fname)
        if sr != self.config["signal"]["sampling_rate"]:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sr, self.config["signal"]["sampling_rate"]))
        audio = audio.unsqueeze(0)

        spec_path = fname.replace("/wavs/", "/feats/").replace(".wav", ".spec.pt")
        if os.path.exists(spec_path):
            spec = torch.load(spec_path)
        else:
            spec, _ = self.spec_extractor(audio)
            spec = spec.transpose(1, 2).squeeze(0)
            torch.save(spec, spec_path)

        return (spec, audio)

    def __getitem__(self, index):
        fid, sid, txt, _ = self.inputs[index]
        txt = self.get_text(txt)
        spk = self.get_speaker(sid)
        spec, audio = self.get_audio(fid)
        
        return {
            "pid": txt,
            "speaker": spk, 
            "spectrogram": spec,
            "audio": audio
        }

    def __len__(self):
        return len(self.inputs)


class VITS2Collate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, n_frames_per_step=1):

        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized pid and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot pid sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x["spectrogram"].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x["pid"]) for x in batch])
        max_spec_len = max([x["spectrogram"].size(1) for x in batch])
        max_wav_len  = max([x["audio"].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths  = torch.LongTensor(len(batch))
        speaker_ids  = torch.LongTensor(len(batch))

        text_padded = torch.zeros(len(batch), max_text_len).long()
        spec_padded = torch.zeros(len(batch), batch[0]["spectrogram"].size(0), max_spec_len)
        wav_padded  = torch.zeros(len(batch), 1, max_wav_len)

        for i in range(len(ids_sorted_decreasing)):
            data = batch[ids_sorted_decreasing[i]]
            
            pid = data["pid"]
            text_padded[i, :pid.size(0)] = pid
            text_lengths[i] = pid.size(0)

            spec = data["spectrogram"]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = data["audio"]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            speaker_ids[i] = data["speaker"] 

        return dict(
            text          = text_padded,
            text_lengths  = text_lengths,
            feats         = spec_padded,
            feats_lengths = spec_lengths,
            sids          = speaker_ids
        ), wav_padded


def intersperse(lst: List, item: int=0) -> List:
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst

    return result

