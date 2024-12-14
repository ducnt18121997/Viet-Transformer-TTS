import os
import sys

sys.path.append(".")
import tqdm
import random
import numpy as np
import soundfile as sf
from typing import Dict
import torch
import torch.autograd
from modules.g2p import G2p_vi
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet2.tts.feats_extract.linear_spectrogram import LinearSpectrogram
from espnet2.tts.feats_extract.log_spectrogram import LogSpectrogram
from espnet2.tts.feats_extract.log_mel_fbank import LogMelFbank
from espnet2.tts.feats_extract.dio import Dio
from espnet2.tts.feats_extract.energy import Energy
from src.tools.utils import load_wav_to_torch, extract_embedding, beta_binomial_prior_distribution


def create_filelist(data_dir: str, speakers: list, out_dir: str) -> None:
    train_list, test_list = [], []
    g2p_vi    = G2p_vi()
    for spk in speakers:
        with open(os.path.join(data_dir, spk, "metadata.csv"), "r", encoding="utf8") as f:
            metadata = [line.split("|") for line in f.read().split("\n") if line]

        print(f"[==] speaker no.{speakers[spk]}: {spk}")
        file_list = []
        for line in tqdm.tqdm(metadata, position=0, leave=False):
            file_name, text = line
            file_name = os.path.join(data_dir, spk, "wavs", file_name)
            if not os.path.exists(file_name): 
                continue
            if any(t not in g2p_vi.vn_words + [",", "."] for t in text.split()): 
                continue # remove sample with english words

            waveform = sf.read(file_name)[0]
            if int(len(waveform) / 256 - 1) > 1000: 
                continue # remove too long waveforms
            if int(len(waveform) / 256 - 1) < 64: 
                continue # remove too short waveforms

            # initilize phonemes
            phonemes, boundaries = g2p_vi(text.replace("-", " "))
            file_list.append("|".join(
                [file_name, spk, " ".join(phonemes),  ", ".join([str(b) for b in boundaries])]
            ))
        random.shuffle(file_list)
        train_list.extend(file_list[: int(0.95 * len(file_list))])
        test_list.extend(file_list[int(0.95 * len(file_list)): ])

    with open(os.path.join(out_dir, "train.txt"), "w", encoding="utf8") as f:
        f.write("\n".join(train_list))

    with open(os.path.join(out_dir, "test.txt"), "w", encoding="utf8") as f:
        f.write("\n".join(test_list))


def create_unknown_filelist(data_dir: str, out_dir: str, *kwags) -> None:
    train_list, test_list = [], []
    g2p_vi    = G2p_vi()
    with open(os.path.join(data_dir, "metadata.csv"), "r", encoding="utf8") as f:
        metadata = [line.split("|") for line in f.read().split("\n") if line]

    file_list = []
    for line in tqdm.tqdm(metadata, position=0, leave=False):
        file_name, text = line
        file_name = os.path.join(data_dir, "wavs", file_name)

        waveform = sf.read(file_name)[0]
        if int(len(waveform) / 256 - 1) > 1000: 
            continue # remove too long waveforms
        if int(len(waveform) / 256 - 1) < 64: 
            continue # remove too short waveforms

        # initilize phonemes
        phonemes, boundaries = g2p_vi(text)
        file_list.append("|".join(
            [file_name, file_name," ".join(phonemes),  ", ".join([str(b) for b in boundaries])]
        ))

    random.shuffle(file_list)
    train_list.extend(file_list[: int(0.95 * len(file_list))])
    test_list.extend(file_list[int(0.95 * len(file_list)): ])

    print(f"Total dataset {len(train_list)} train samples!!!")
    with open(os.path.join(out_dir, "train.txt"), "w", encoding="utf8") as f:
        f.write("\n".join(file_list))

    print(f"Total dataset {len(test_list)} test samples!!!")
    with open(os.path.join(out_dir, "test.txt"), "w", encoding="utf8") as f:
        f.write("\n".join(file_list))


def build_feat_extractor(choice: str, config: Dict) -> AbsFeatsExtract:
    if choice == "linear_spectrogram":
        feats_extractor  = LinearSpectrogram(
            n_fft      = config["stft"]["filter_length"],
            win_length = config["stft"]["win_length"],
            hop_length = config["stft"]["hop_length"],
            window     = config["stft"]["window"]
        )
    elif choice == "spectrogram": 
        feats_extractor  = LogSpectrogram(
            n_fft      = config["stft"]["filter_length"],
            win_length = config["stft"]["win_length"],
            hop_length = config["stft"]["hop_length"],
            window     = config["stft"]["window"]
        )
    elif choice == "fbank":
        feats_extractor  = LogMelFbank(
            fs         = config["signal"]["sampling_rate"],
            n_fft      = config["stft"]["filter_length"],
            win_length = config["stft"]["win_length"],
            hop_length = config["stft"]["hop_length"],
            window     = config["stft"]["window"],
            n_mels     = config["mel"]["channels"],
            fmin       = config["mel"]["fmin"],
            fmax       = config["mel"]["fmax"],
            log_base   = config["mel"]["log_base"]
        )
    else:
        raise NotImplementedError(f"Not implement for feat_extractor_choice=`{choice}` ")

    return feats_extractor


def prepare_inputs(list_segments: list, hparams: dict, feat_extractor_choice: str="linear_spectrogram") -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats_extractor  = build_feat_extractor(feat_extractor_choice, config=hparams).to(device)
    feats_odim       = feats_extractor.output_size()
    pitch_extractor  = Dio(
        fs         = hparams["signal"]["sampling_rate"],
        n_fft      = hparams["stft"]["filter_length"],
        hop_length = hparams["stft"]["hop_length"],
        reduction_factor=1, use_token_averaged_f0=False
    ).to(device)
    energy_extractor = Energy(
        fs         = hparams["signal"]["sampling_rate"],
        n_fft      = hparams["stft"]["filter_length"],
        win_length = hparams["stft"]["win_length"],
        hop_length = hparams["stft"]["hop_length"],
        window     = hparams["stft"]["window"],
        reduction_factor=1, use_token_averaged_energy=False
    ).to(device)

    prosody_dict  = {f"{segment[1]}_{os.path.basename(segment[0])}": {} for segment in list_segments}
    for line in tqdm.tqdm(list_segments, desc="Creating prosody"):
        fid, sid, pid, duration = line
        src_path = os.path.dirname(os.path.dirname(fid))
        prosody_dict[f"{sid}_{os.path.basename(fid)}"]["wav"] = fid

        os.makedirs(os.path.join(src_path, "embedding"), exist_ok=True)
        embedding_location = os.path.join(src_path, "embedding", f"{'.'.join(os.path.basename(fid).split('.')[: -1])}.npy")
        prosody_dict[f"{sid}_{os.path.basename(fid)}"]["embedding"] = embedding_location

        os.makedirs(os.path.join(src_path, "duration"), exist_ok=True)
        duration_location = os.path.join(src_path, "duration", f"{'.'.join(os.path.basename(fid).split('.')[: -1])}.txt")
        prosody_dict[f"{sid}_{os.path.basename(fid)}"]["duration"] = duration_location
        if len(duration.split(", ")) == len(pid.split()):
            with open(duration_location, "w", encoding="utf8") as f:
                f.write(duration.strip())
        elif sum([int(d) for d in duration.split(", ")]) == len(pid.split()):
            pass
        else:
            raise ValueError(f"Last values of input must be boundaries or duration...")

        os.makedirs(os.path.join(src_path, "alignment"), exist_ok=True)
        aligment_location = os.path.join(src_path, "alignment", f"{'.'.join(os.path.basename(fid).split('.')[: -1])}.npy")
        prosody_dict[f"{sid}_{os.path.basename(fid)}"]["alignment"] = aligment_location

        os.makedirs(os.path.join(src_path, "feats"), exist_ok=True)
        feat_location = os.path.join(src_path, "feats", f"{'.'.join(os.path.basename(fid).split('.')[: -1])}.pt")
        prosody_dict[f"{sid}_{os.path.basename(fid)}"]["feat"] = feat_location

        os.makedirs(os.path.join(src_path, "pitch"), exist_ok=True)
        pitch_location = os.path.join(src_path, "pitch", f"{'.'.join(os.path.basename(fid).split('.')[: -1])}.pt")
        prosody_dict[f"{sid}_{os.path.basename(fid)}"]["pitch"] = pitch_location

        os.makedirs(os.path.join(src_path, "energy"), exist_ok=True)
        energy_location = os.path.join(src_path, "energy", f"{'.'.join(os.path.basename(fid).split('.')[: -1])}.pt")
        prosody_dict[f"{sid}_{os.path.basename(fid)}"]["energy"] = energy_location

        if not all(os.path.exists(v) for k, v in prosody_dict[f"{sid}_{os.path.basename(fid)}"].items() if k != "duration"):
            audio, sampling_rate = load_wav_to_torch(fid) 
            assert sampling_rate == hparams["signal"]["sampling_rate"], (
                f"Audio sample rate missmatch: given {sampling_rate} Hz, expected {hparams['signal']['sampling_rate']} Hz"
            )
            audio = audio.unsqueeze(0).to(device)
            feats, feats_lens = feats_extractor(audio)

            # initialize embeddings
            if not os.path.exists(embedding_location):
                embed = extract_embedding(fid)
                torch.save(embed, embedding_location)

            # initialize features
            if not os.path.exists(feat_location):
                torch.save(feats.squeeze().detach().cpu(), feat_location)

            # initialize pitches
            if not os.path.exists(pitch_location):
                pitch, _ = pitch_extractor(audio, feats_lens)
                torch.save(pitch.squeeze().detach().cpu(), pitch_location)

            # initialize energies
            if not os.path.exists(energy_location):
                energy, _ = energy_extractor(audio)
                torch.save(energy.squeeze().detach().cpu(), energy_location)

            # initialize alignment map
            if not os.path.exists(aligment_location):
                alignment = beta_binomial_prior_distribution(len(pid.split()), feats.size(1))
                np.save(aligment_location, alignment)

    return prosody_dict, feats_odim
