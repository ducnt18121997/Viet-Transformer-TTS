
import torch
import numpy
import random
import argparse


def build_arguments():
    parser = argparse.ArgumentParser()
    ### TASK CONFIGURATION ###
    parser.add_argument("--task", default="text2wav", 
                        choices=["text2wav", "fastspeech2", "adaspeech", "jets", "vits2", "matcha", "hifigan"],
                        help="task trainer, (default: %(default)s)")
    ### DATA CONFIGURATION ###
    parser.add_argument("-i", "--input_folder", type=str, required=True,
                        help="directly to egs folder")
    parser.add_argument("-d", "--data_folder", type=str, required=True,
                        help="directly to dataset folder")
    parser.add_argument("-o", "--output_folder", type=str, required=True,
                        help="directory to saved model folder")
    ### MODEL CONFIGURATION ###
    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="directly to joint pretrained checkpoint")
    parser.add_argument("-a", "--acoustic_checkpoint", type=str, default=None,
                        help="directly to acoustic pretrained checkpoint")
    parser.add_argument("-v", "--vocoder_checkpoint", type=str, default=None,
                        help="directly to vocoder pretrained checkpoint")
    ### TRAINING CONFIGURATION ###
    parser.add_argument("--version", type=str, default="fastspeech2", 
                        choices=["fastspeech2", "matcha", "adaspeech"],
                        help="model type when use acoustic model, (default: %(default)s)")
    parser.add_argument("--max_iter", type=int, default=2000000)
    parser.add_argument("--is_finetune", action="store_true")

    return parser.parse_args()


def set_seed(seed: int):
    r"""
    Sets the seed for generating random numbers

    Args:
        seed: seed value.

    Returns:
        None
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
