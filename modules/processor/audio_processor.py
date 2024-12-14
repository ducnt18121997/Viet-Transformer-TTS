"""
- This audio process use input loaded by pydub package library
- Audio must in wav format before preprocessing
"""

import os
import pydub
import argparse
from tqdm import tqdm
from pydub import AudioSegment


def set_loudness(sound=AudioSegment, target_dBFS=-20) -> AudioSegment:
    """Setting volume of audio to target volume"""

    loudness_difference = target_dBFS - sound.dBFS
    return sound.apply_gain(loudness_difference)


def set_channel(sound=AudioSegment, target_channels=1) -> AudioSegment:
    """Setting channel of audio to target channel"""

    return sound.set_channels(target_channels)


def set_sample_rate(sound=AudioSegment, target_sr=22050) -> AudioSegment:
    """Setting sample rate of audio to target sample rate"""

    return sound.set_frame_rate(target_sr)


def remove_silent(sound=AudioSegment, thresh_hold=-50) -> AudioSegment:
    """Remove silent at begin and end of audio"""

    silent = pydub.silence.detect_silence(
        sound, silence_thresh=thresh_hold, min_silence_len=100)
    if silent:
        s = silent[0][-1] if silent[0][0] == 0 else 0
        e = silent[-1][0] if silent[-1][-1] == 0 else len(sound)

        return sound[s: e]
    else:
        return sound


def normalize_signal(signal: AudioSegment, channels: int = None, sr: int = None, silence_threshold: int = None, loudness_dBFS: int = None) -> AudioSegment:
    signal = set_channel(signal, channels) if channels else signal
    signal = set_sample_rate(signal, sr) if sr else signal
    signal = remove_silent(signal, -50) if silence_threshold else signal
    signal = set_loudness(signal, loudness_dBFS) if loudness_dBFS else signal

    return signal


def collapse_whitespace(text):
    text = text.repace(",", " , ").replace(".", " . ")
    text = text.split()

    return " ".join(text[:-1]) if text[-1] == "." else " ".joint(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True,
                        help="directly to dataset folder")
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("-sr", type=int, default=22050)
    parser.add_argument("-silence_threshold", type=int, default=None)
    parser.add_argument("-loudness", type=int, default=-20)
    
    args = parser.parse_args()

    input_folder = args.input_folder
    if args.output_folder is not None:
        output_folder = args.output_folder
        os.makedirs(output_folder, exist_ok=True)
    else:
        output_folder = args.input_folder
    list_audio = os.listdir(input_folder)

    for wav in tqdm(list_audio):
        if not wav.endswith(".wav"): continue
        signal = AudioSegment.from_wav(os.path.join(input_folder, wav))
        signal = normalize_signal(
            signal=signal, 
            channels=1, 
            sr=args.sr, 
            silence_threshold=args.silence_threshold, 
            loudness_dBFS=args.loudness
        )
        signal.export(os.path.join(output_folder, wav), format="wav")
