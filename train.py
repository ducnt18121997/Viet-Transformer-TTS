from src.trainer.utils import build_arguments, set_seed
from src.tools.tools_for_model import build_config


if __name__ == "__main__":
    args = build_arguments()
    conf = build_config("config/")
    set_seed(conf["train"]["seed"])

    if args.is_finetune is True:
        assert args.task in ["text2wav"], \
            f"Fine-tuning model is not supported for `{args.task}` yet!"

    if args.task == "text2wav":
        if args.is_finetune is True:
            from src.trainer import JointFinetuner as TrainerModule
        else:
            from src.trainer import JointTrainer as TrainerModule
    elif args.task == "fastspeech2":
        from src.trainer import FastSpeech2Trainer as TrainerModule
    elif args.task == "jets":
        from src.trainer import JETSTrainer as TrainerModule
    elif args.task == "vits2":
        from src.trainer import VITS2Trainer as TrainerModule
    elif args.task == "matcha":
        from src.trainer import MatchaTrainer as TrainerModule
    elif args.task == "hifigan":
        from src.trainer import HiFiGANTrainer as TrainerModule
    elif args.task == "adaspeech":
        from src.trainer import AdaSpeechTrainer as TrainerModule
    else:
        raise NotImplementedError(f"Not supported to training `{args.task}`...")

    trainer = TrainerModule(args, conf)
    trainer.run()
