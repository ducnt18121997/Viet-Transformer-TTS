import os
import yaml
import json
import glob
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim


def clear_disk(_path: str, prefix: str="generator") -> str:
    pattern = os.path.join(_path, f"{prefix}*.pt")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    cp_list = sorted(cp_list, key=lambda x: int(os.path.basename(x).split("_")[-2].replace("iteration", "")))
    if len(cp_list) > 3:
        os.remove(cp_list[0])
        print(f"Free up space by deleting ckpt {cp_list[0]}")

    return


def to_device(datas: Any, device: torch.device):
    if datas is None:
        pass
    elif isinstance(datas, dict):
        datas = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in datas.items()}
    elif isinstance(datas, (tuple, list)):
        datas = tuple([x.to(device) if isinstance(x, torch.Tensor) else x for x in datas])
    else:
        raise NotImplementedError

    return datas


def scan_checkpoint(_path: str, prefix: str="generator") -> str:
    pattern = os.path.join(_path, f"{prefix}*.pt")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    cp_list = sorted(cp_list, key=lambda x: int(os.path.basename(x).split("_")[-2].replace("iteration", "")))

    return cp_list[-1]


def apply_weight(_path: str, model: nn.Module, optim: optim.Optimizer=None) -> nn.Module:
    if isinstance(_path, str):
        checkpoint_dict = torch.load(_path)
    else:
        checkpoint_dict = _path

    loaded_state = checkpoint_dict["state_dict"]
    for name, param in loaded_state.items():
        if name not in model.state_dict():
            if name not in model.state_dict():
                print("%s is not in the model."%name)
                continue
        if model.state_dict()[name].size() != loaded_state[name].size():
            print("Wrong parameter length: %s, model: %s, loaded: %s"%(name, model.state_dict()[name].size(), loaded_state[name].size()))
            continue
        model.state_dict()[name].copy_(param)

    if isinstance(_path, str): print(f"Loaded checkpoint from {_path}...")        
    if optim is not None: # and "optimizer" in checkpoint_dict:
        try:
            optim.load_state_dict(checkpoint_dict["optimizer"])
        except:
            print(f"There're something changed in model! Skip optimizer checkpoint")
        
        return model, optim

    return model
    

def build_config(config_path: str)-> dict:

    return {
        "audio": yaml.load(open(os.path.join(config_path, "preprocessing_config.yaml"), "r"), Loader=yaml.FullLoader),
        "models": yaml.load(open(os.path.join(config_path, "model_config.yaml"), "r"), Loader=yaml.FullLoader),
        "train": yaml.load(open(os.path.join(config_path, "train_config.yaml"), "r"), Loader=yaml.FullLoader)
    }
