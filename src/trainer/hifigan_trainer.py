import os
import sys

sys.path.append(".")
import tqdm
from argparse import ArgumentParser
from loguru import logger as system_logger

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.gan_tts.hifigan import HiFiGAN
from models.gan_tts.hifigan import (
    HiFiGANMultiScaleMultiPeriodDiscriminator, 
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
    MelSpectrogramLoss
)
from log.logger import text_colors
from src.trainer.base import BaseTrainer
from src.tools.dataloader import HifiGanLoader, HifiGanCollate
from src.tools.tools_for_model import apply_weight, scan_checkpoint, to_device


class HiFiGANTrainer(BaseTrainer):

    def __init__(self, args: ArgumentParser, conf: dict) -> None:
        super().__init__(args, conf)    

    def __init_dataset__(self):
        # 0. Load pre-calculated stats
        self.stats = None

        # 1. Initialize dataloader
        self.train_loader = DataLoader(
            HifiGanLoader(
                self.train_set, 
                sampling_rate = self.conf["audio"]["signal"]["sampling_rate"], 
                use_speaker   = self.conf["models"]["hifigan"]["use_spk"]
            ), 
            shuffle    = True, 
            batch_size = self.conf["train"]["batch_size"], 
            drop_last  = True,
            collate_fn = HifiGanCollate(self.conf["audio"], self.conf["models"]["hifigan"]["segment_size"])
        )

        self.valid_loader = DataLoader(
            HifiGanLoader(
                self.test_set, 
                sampling_rate = self.conf["audio"]["signal"]["sampling_rate"], 
                use_speaker   = self.conf["models"]["hifigan"]["use_spk"]
            ), 
            batch_size = self.conf["train"]["batch_size"],
            collate_fn = HifiGanCollate(self.conf["audio"], self.conf["models"]["hifigan"]["segment_size"])
        )

    def __init_model__(self):
        # Initialize model
        self.model_conf = {"idim": 80, **self.conf["models"]["hifigan"]}
        self.model = HiFiGAN(
            in_channels     = self.model_conf["idim"], 
            out_channels    = 1,
            global_channels = 192 if self.conf["models"]["hifigan"]["use_spk"] else -1
        ).to(self.device)
        self.discriminator = HiFiGANMultiScaleMultiPeriodDiscriminator().to(self.device)

    def __init_loss__(self):
        # Initialize loss function
        self.melspectrogram_loss = MelSpectrogramLoss(**{
            "fs": self.conf["audio"]["signal"]["sampling_rate"],
            "n_fft": self.conf["audio"]["stft"]["filter_length"],
            "hop_length": self.conf["audio"]["stft"]["hop_length"],
            "win_length": self.conf["audio"]["stft"]["win_length"],
            "window": self.conf["audio"]["stft"]["window"],
            "n_mels": self.conf["audio"]["mel"]["channels"],
            "fmin": self.conf["audio"]["mel"]["fmin"],
            "fmax": self.conf["audio"]["mel"]["fmax"],
            # "log_base": None
        }).to(self.device)

        self.generator_adv_loss = GeneratorAdversarialLoss(
            average_by_discriminators = False,
            loss_type                 = "mse"
        )
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(
            average_by_discriminators = False,
            loss_type                 = "mse"
        )
        self.feat_match_loss = FeatureMatchLoss(
            average_by_discriminators = False,   # whether to average loss value by #discriminators
            average_by_layers         = False,   # whether to average loss value by #layers of each discriminator
            include_final_outputs     = True     # whether to include final outputs for loss calculation
        )

    def __init_optimizer__(self):
        # 1. Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), **self.conf["train"]["hifigan"]["optimizer"]
        )
        self.disc_optimizer = optim.AdamW(
            self.discriminator.parameters(), **self.conf["train"]["hifigan"]["optimizer"]
        )

        # 2. Initialize parameter from checkpoint / resume training
        self.iter, self.epoch = 0, 0
        dis_ckpt = ""
        if self.args.checkpoint is not None:
            gen_ckpt = self.args.checkpoint
            self.model, self.optimizer = apply_weight(gen_ckpt, self.model, self.optimizer)
            self.epoch = int(self.args.checkpoint.split('/')[-1].split('_')[1][5:]) + 1
            self.iter  = int(self.args.checkpoint.split('/')[-1].split('_')[2][9:])

            (f"Loaded checkpoint from {self.args.checkpoint}...")
            dis_ckpt   = os.path.join(os.path.dirname(gen_ckpt), "last_dicriminator_statedict.pt")
        elif os.path.exists(self.args.output_folder):
            gen_ckpt = scan_checkpoint(self.args.output_folder)
            if gen_ckpt is not None:
                self.model, self.optimizer = apply_weight(gen_ckpt, self.model, self.optimizer)
                self.epoch = int(os.path.basename(gen_ckpt).split("_")[-3].replace("epoch", ""))
                self.iter  = int(os.path.basename(gen_ckpt).split("_")[-2].replace("iteration", ""))

                dis_ckpt   = os.path.join(os.path.dirname(gen_ckpt), "last_dicriminator_statedict.pt")
            system_logger.info(f"Resumed training from last checkpoint at epoch {self.epoch} & iter {self.iter}...")

        if os.path.exists(dis_ckpt):
            self.discriminator, self.disc_optimizer = apply_weight(dis_ckpt, self.discriminator, self.disc_optimizer)
        else:
            system_logger.info(f"Can't detect discriminator last checkpoint, using new discriminator can make model harder to learn...")

        # 3. Initialize scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999875)
        self.disc_scheduler = optim.lr_scheduler.ExponentialLR(self.disc_optimizer, gamma=0.999875)

    def train_one_epoch(self):
        self.model.train()
        self.discriminator.train()
        pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {self.epoch} [0%]", position=0, leave=False)
        for batch in pbar:
            # forward
            x, y, e = to_device(batch, self.device)
            y_g_hat = self.model(x, g=e)

            # discriminator backward
            p_hat = self.discriminator(y_g_hat.detach())
            p     = self.discriminator(y)
            loss_disc_real, loss_disc_fake = self.discriminator_adv_loss(p_hat, p)

            self.disc_optimizer.zero_grad()
            loss_disc_all = loss_disc_real + loss_disc_fake
            loss_disc_all.backward()
            self.disc_optimizer.step()

            # self.model backward
            loss_mel = self.melspectrogram_loss(y_g_hat, y) * 45
            p_hat = self.discriminator(y_g_hat)
            with torch.no_grad():
                p = self.discriminator(y)

            loss_fm  = self.feat_match_loss(p_hat, p) * 2
            loss_gen = self.generator_adv_loss(p_hat)

            self.optimizer.zero_grad()
            loss_gen_all = loss_gen + loss_fm + loss_mel
            loss_gen_all.backward()
            self.optimizer.step()

            pbar.set_postfix_str("with self.model loss - {}, Discriminator loss - {}, mel loss - {}"\
                .format(round(loss_gen_all.item(), 4), round(loss_disc_all.item(), 4), round(loss_mel.item() / 45, 4)))
            
            # summary logging
            self.train_logger.add_scalar("training/self.model_loss", loss_gen_all, self.iter)
            self.train_logger.add_scalar("training/msd_loss", loss_disc_real, self.iter)
            self.train_logger.add_scalar("training/mpd_loss", loss_disc_fake, self.iter)
            self.train_logger.add_scalar("training/mel_loss", loss_mel, self.iter)
            self.iter += 1

    def valid_one_epoch(self):
        self.scheduler.step()
        self.disc_scheduler.step()

        self.model.eval()
        val_loss = 0.0
        for i, batch in enumerate(self.valid_loader):
            x, y, e = to_device(batch, self.device)
            with torch.no_grad():
                y_g_hat = self.model(x, g=e)
            val_loss   += self.melspectrogram_loss(y_g_hat, y).item()

        val_loss = val_loss / (i + 1)
        diff = round(((val_loss - self.best_val_loss) / self.best_val_loss), 4) \
            if self.best_val_loss is not None else -1
        if diff <= 0:
            msg = f"{text_colors.OKGREEN}{round(val_loss, 4)} (↓ {abs(diff) * 100}%){text_colors.ENDC}"
            self.best_val_loss = val_loss
        else:
            msg = f"{text_colors.FAIL}{round(val_loss, 4)} (↑ {abs(diff) * 100}%){text_colors.ENDC}"
        system_logger.info(f"Epoch {self.epoch} - iters {self.iter}: mel-loss {msg}")
