import os
import sys

sys.path.append(".")
import json
import tqdm
from argparse import ArgumentParser
from loguru import logger as system_logger

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.tts.fastspeech2 import FastSpeech2, FastSpeech2Loss
from log.logger import text_colors
from src.trainer.base import BaseTrainer
from src.tools.dataloader import Fastspeech2Loader, Fastspeech2Collate
from src.tools.tools_for_model import apply_weight, scan_checkpoint, to_device


class FastSpeech2Trainer(BaseTrainer):

    def __init__(self, args: ArgumentParser, conf: dict) -> None:
        super().__init__(args, conf)

    def __init_dataset__(self):
        self.self_learning = self.conf["models"]["fastspeech2"]["variance"]["learn_alignment"]

        # 0. Load pre-calculated stats
        self.stats  = None
        if os.path.exists(os.path.join(self.args.output_folder, "stats.json")):
            self.stats = json.load(open(os.path.join(self.args.output_folder, "stats.json"), "r"))

        # 1. Initialize dataloader
        self.conf["audio"]["self_learning"] = self.self_learning
        self.train_loader = DataLoader(
            Fastspeech2Loader(
                self.train_set, 
                config   = self.conf["audio"], 
                speakers = self.speakers, 
                accents  = self.accents, 
                stats    = self.stats
            ), 
            shuffle    = True, 
            batch_size = self.conf["train"]["batch_size"], 
            pin_memory = True, 
            drop_last  = True,
            collate_fn=Fastspeech2Collate(            
                n_speakers      = len(self.speakers) if self.speakers is not None else -1, 
                use_accent      = self.conf["models"]["fastspeech2"]["use_cvae"],
                return_waveform = False
            )
        )
        self.stats = self.train_loader.dataset.stats
        print(json.dumps(self.stats, ensure_ascii=False, indent=4))
        
        self.valid_loader = DataLoader(
            Fastspeech2Loader(
                self.test_set, 
                config   = self.conf["audio"], 
                speakers = self.speakers, 
                accents  = self.accents, 
                stats    = self.stats
            ), 
            batch_size = self.conf["train"]["batch_size"], 
            collate_fn = Fastspeech2Collate(
                n_speakers      = len(self.speakers) if self.speakers is not None else -1, 
                use_accent      = self.conf["models"]["fastspeech2"]["use_cvae"],
                return_waveform = False
            )
        )

    def __init_model__(self):
        # Initialize model
        self.model_conf = {
            "idim": len(self.symbols),
            "odim": self.train_loader.dataset.feats_odim,
            "conf": {
                "n_speakers": len(self.speakers) if self.speakers is not None else -1,
                "n_accents": len(self.accents) if self.accents is not None else -1,
                "hparams": self.conf["models"]["fastspeech2"],
            },
        }

        self.model = FastSpeech2(
            n_symbols  = self.model_conf["idim"],
            n_channels = self.model_conf["odim"],
            stats      = self.stats,
            **self.model_conf["conf"]
        ).to(self.device)
        self.discriminator = None
        
    def __init_loss__(self):
        # Initialize loss function
        self.loss_conf = self.conf["train"]["fastspeech2"]["loss"]
        self.loss_conf.update(dict(
            pitch_feature_level  = self.conf["models"]["fastspeech2"]["variance"]["variance_embedding"]["pitch_feature"], 
            energy_feature_level = self.conf["models"]["fastspeech2"]["variance"]["variance_embedding"]["energy_feature"],
        ))

        self.criterion = FastSpeech2Loss(self.args.max_iter, self.loss_conf)

    def __init_optimizer__(self):
        # 1. Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), **self.conf["train"]["fastspeech2"]["optimizer"]
        )

        # 2. Initialize parameter from checkpoint / resume training
        self.iter, self.epoch = 0, 0
        if self.args.checkpoint is not None:
            # apply weight & ignore new or other state_dict
            self.model, self.optimizer = apply_weight(self.args.checkpoint, self.model, self.optimizer)
        elif os.path.exists(self.args.output_folder):
            system_logger.info(f"Resuming training from last checkpoint...")
            model_ckpt = scan_checkpoint(self.args.output_folder)
            if model_ckpt is not None:
                self.model, self.optimizer = apply_weight(self.args.checkpoint, self.model, self.optimizer)
                self.epoch = int(os.path.basename(model_ckpt).split("_")[-3].replace("epoch", ""))
                self.iter  = int(os.path.basename(model_ckpt).split("_")[-2].replace("iteration", ""))

        # 3. Initialize scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999875)

    def train_one_epoch(self):
        self.model.train()
        pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {self.epoch}", position=0, leave=False)
        for batch in pbar:
            x, y = [to_device(_, device=self.device) for _ in batch]
            # forward
            y_pred, y_extra = self.model(**x, step=self.iter)

            # calculate loss
            losses = self.criterion(y_pred, y[1: ] + y_extra, step=self.iter)
            total_loss = sum([sum([x for x in _loss.values() if x != 0]) if isinstance(_loss, dict) else _loss for _loss in losses.values()])

            # backward
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # logging
            msg = [" - ".join([f"{k}-loss: {round(v.item(), 2)}" for k, v in losses.items() if v != 0])]
            if self.iter % self.conf["train"]["log_step"] == 0:
                self.train_logger.log(losses, self.iter, lr=self.optimizer.param_groups[0]["lr"])
            pbar.set_postfix_str(f"with {' - '.join(msg)}")
            pbar.set_description_str(f"Epoch {self.epoch} [{round(self.iter / self.args.max_iter * 100, 2)}%]")
            self.iter += 1
    
    def valid_one_epoch(self):
        self.scheduler.step()
        self.model.eval()
        val_loss = {"feat": 0.0, "feat_postnet": 0.0}
        for batch in self.valid_loader:
            with torch.no_grad():
                x, y = [to_device(_, device=self.device) for _ in batch]
                y_pred, y_extra = self.model(**x, step=self.iter)

                loss = self.criterion(y_pred, y[1: ] + y_extra)
            val_loss["feat"] += loss["feat"].item()
            if "feat_postnet" in loss: val_loss["feat_postnet"] += loss["feat_postnet"].item()

        val_loss = {k: v / self.valid_loader.__len__() for k, v in val_loss.items()}
        self.valid_logger.log(val_loss, self.iter, state_dict=self.model)

        count, msg = 0, []
        for _loss in val_loss:
            if val_loss[_loss] == 0:
                count += 1
                continue
            diff = round(((val_loss[_loss] - self.best_val_loss[_loss]) / self.best_val_loss[_loss]), 4) \
                if self.best_val_loss is not None else -1
            if diff <= 0:
                msg.append(f"{_loss}-loss {text_colors.OKGREEN}{round(val_loss[_loss], 4)} (↓ {abs(diff) * 100}%){text_colors.ENDC}")
                count += 1
            else:
                msg.append(f"{_loss}-loss {text_colors.FAIL}{round(val_loss[_loss], 4)} (↑ {abs(diff) * 100}%){text_colors.ENDC}")
        self.best_val_loss = val_loss.copy() if count == 2 else self.best_val_loss
        system_logger.info(f"Epoch {self.epoch}: {' - '.join(msg)}")
