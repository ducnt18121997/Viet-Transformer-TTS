import torch
from modules.g2p import symbols
from models.utils import show_model, show_params
from src.tools.tools_for_model import build_config, apply_weight


if __name__ == "__main__":
    config = build_config("config/")
    stats  = {"pitch": {"mean": 5.173152923583984, "std": 0.3412020493798016, "max": 2.912043333053589, "min": -2.9330103397369385}, "energy": {"mean": 36.22599411010742, "std": 29.15140760631787, "max": 7.434078216552734, "min": -1.2422232627868652}}
    n_spks = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpi"

    print("Choose model to display:")
    print("1. fastspeech2")
    print("2. hifigan")
    print("3. joint")
    print("4. jets")
    model_display = input("Your choice: ")
    assert model_display in ["1", "2", "3", "4"]
    if model_display == "1": # acoustic
        from models.tts.fastspeech2 import FastSpeech2
        model_conf = {
            "idim": len(symbols),
            "odim": config["audio"]["mel"]["channels"],
            "conf": {
                "n_speakers": n_spks,
                "hparams": config["models"]["fastspeech2"],
            },
        }
        nnet = FastSpeech2(
            n_symbols  = len(symbols),
            n_channels = config["audio"]["mel"]["channels"],
            stats      = stats,
            **model_conf["conf"]
        )
    elif model_display == "2":
        from models.gan_tts.hifigan import HiFiGAN
        nnet = HiFiGAN(config = config["models"]["hifigan"])
    elif model_display == "3":
        from models.gan_tts.text2wav.model import Text2Wav
        model_conf = {
            "idim": len(symbols),
            "odim": config["audio"]["mel"]["channels"],
            "text2mel_conf": {
                "n_speakers": n_spks,
                "hparams": config["models"]["fastspeech2"],
            },
            "mel2wav_conf": config["models"]["hifigan"]
        }
        nnet = Text2Wav(
            vocabs          = len(symbols),
            aux_channels    = config["audio"]["mel"]["channels"],
            text2mel_params = {**model_conf["text2mel_conf"], "stats": stats},
            mel2wav_params  = model_conf["mel2wav_conf"]
        )
    elif model_display == "4":
        model_conf = config["models"]["jets"]
        model_conf["idim"] = len(symbols)
        model_conf["odim"] = config["audio"]["mel"]["channels"]
        model_conf["generator_params"]["spks"] = n_spks
        from models.gan_tts.jets import JETS
        nnet = JETS(
            idim=len(symbols),
            odim=513,
            **model_conf["generator_params"]
        )
    else:
        raise NotImplementedError

    show_model(nnet)
    show_params(nnet)

    checkpoint_path = input("Nhập pre-trained weight path: ")
    if checkpoint_path:
        nnet = apply_weight(checkpoint_path, nnet)
        print(f"Sucessful loaded from {checkpoint_path}...")