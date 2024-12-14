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
from espnet2.gan_tts.utils import get_segments

from models.gan_tts.jets import JETS, JETSLoss
from models.gan_tts.hifigan import HiFiGANMultiScaleMultiPeriodDiscriminator
from log.logger import text_colors
from src.trainer.base import BaseTrainer
from src.tools.dataloader import JETSLoader, JETSCollate
from src.tools.tools_for_model import apply_weight, scan_checkpoint, to_device


class JETSTrainer(BaseTrainer):

    def __init__(self, args: ArgumentParser, conf: dict) -> None:
        super().__init__(args, conf)    
    
    def __init_dataset__(self):
        # 0. Load pre-calculated stats
        self.stats = None
        if self.args.checkpoint is not None:
            self.stats = json.load(open(os.path.join(os.path.dirname(self.args.checkpoint), "stats.json"), "r"))
        elif os.path.exists(self.args.output_folder):
            self.stats = json.load(open(os.path.join(self.args.output_folder, "stats.json"), "r"))

        # 1. Initialize dataloader
        self.train_loader = DataLoader(
            JETSLoader(
                self.train_set, 
                config   = self.conf["audio"], 
                speakers = self.speakers, 
                stats    = self.stats, 
                feat_extractor_choice="fbank"
            ), 
            shuffle    = True, 
            batch_size = self.conf["train"]["batch_size"], 
            pin_memory = True, 
            drop_last  = True,
            collate_fn = JETSCollate(n_speakers=len(self.speakers) if self.speakers is not None else -1),
        )
        self.stats = self.train_loader.dataset.stats
        print(json.dumps(self.stats, ensure_ascii=False, indent=4))

        self.valid_loader = DataLoader(
            JETSLoader(
                self.test_set, 
                config   = self.conf["audio"], 
                speakers = self.speakers, 
                stats    = self.stats, 
                feat_extractor_choice="fbank"
            ), 
            batch_size = self.conf["train"]["batch_size"], 
            collate_fn = JETSCollate(n_speakers=len(self.speakers) if self.speakers is not None else -1),
        )

    def __init_model__(self):
        # Initialize model
        self.model_conf = {
            "idim": len(self.symbols),
            "odim": self.train_loader.dataset.feats_odim,
            **self.conf["models"]["jets"]
        }
        
        self.model = JETS(
            idim = self.model_conf["idim"],
            odim = self.model_conf["odim"],
            **self.model_conf["generator_params"]
        ).to(self.device)
        self.discriminator = HiFiGANMultiScaleMultiPeriodDiscriminator(
            **self.model_conf["discriminator_params"]
        ).to(self.device)
        
    def __init_loss__(self):
        # Initialize loss function
        self.loss_conf = self.conf["train"]["jets"]["loss"]
        self.loss_conf["mel_loss_params"] = {
            "fs": self.conf["audio"]["signal"]["sampling_rate"],
            "n_fft": self.conf["audio"]["stft"]["filter_length"],
            "hop_length": self.conf["audio"]["stft"]["hop_length"],
            "win_length": self.conf["audio"]["stft"]["win_length"],
            "window": self.conf["audio"]["stft"]["window"],
            "n_mels": self.conf["audio"]["mel"]["channels"],
            "fmin": self.conf["audio"]["mel"]["fmin"],
            "fmax": self.conf["audio"]["mel"]["fmax"],
            "log_base": None
        } # build config based on pre-processing
        self.criterion = JETSLoss(**self.loss_conf).to(self.device)
   
    def __init_optimizer__(self):
        # 1. Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), **self.conf["train"]["jets"]["optimizer"]["gen_optim_conf"]
        )
        self.disc_optimizer = optim.AdamW(
            self.discriminator.parameters(), **self.conf["train"]["jets"]["optimizer"]["gen_optim_conf"]
        )
    
        # 2. Initialize parameter from checkpoint / resume training
        self.iter, self.epoch = 0, 0
        dis_ckpt = ""
        if self.args.checkpoint is not None:
            gen_ckpt = self.args.checkpoint
            dis_ckpt = os.path.join(os.path.dirname(gen_ckpt), "last_dicriminator_statedict.pt")
            self.model, self.optimizer = apply_weight(gen_ckpt, self.model, self.optimizer)
        elif os.path.exists(self.args.output_folder):
            system_logger.info(f"Resuming training from last checkpoint...")
            gen_ckpt = scan_checkpoint(self.args.output_folder)
            if gen_ckpt is not None:
                dis_ckpt = os.path.join(os.path.dirname(gen_ckpt), "last_dicriminator_statedict.pt")
                self.model, self.optimizer = apply_weight(gen_ckpt, self.model, self.optimizer)
                self.epoch = int(os.path.basename(gen_ckpt).split("_")[-3].replace("epoch", ""))
                self.iter  = int(os.path.basename(gen_ckpt).split("_")[-2].replace("iteration", ""))
        if os.path.exists(dis_ckpt):
            self.discriminator, self.optimizer = apply_weight(dis_ckpt, self.discriminator, self.disc_optimizer)
        else:
            system_logger.info(f"Can't detect discriminator last checkpoint, using new discriminator can make model harder to learn...")

        # 3. Initialize scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, **self.conf["train"]["jets"]["optimizer"]["gen_scheduler_conf"]
        )
        self.disc_scheduler = optim.lr_scheduler.ExponentialLR(
            self.disc_optimizer, **self.conf["train"]["jets"]["optimizer"]["dis_scheduler_conf"]
        )

    def train_one_epoch(self):
        self.model.train()
        self.discriminator.train()
        pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {self.epoch} [0%]", position=0, leave=False)
        for batch in pbar:
            # forward
            x, speech = batch
            x       = to_device(x, device=self.device)
            speech  = speech.to(self.device)
            outs    = self.model(**x)

            speech_hat_, bin_loss, log_p_attn, start_idxs, d_outs, ds, p_outs, ps, e_outs, es = outs
            speech_ = get_segments(
                speech,
                start_idxs   = start_idxs * self.model.upsample_factor,
                segment_size = self.model.segment_size * self.model.upsample_factor,
            )

            # discriminator backward
            p_hat = self.discriminator(speech_hat_.detach())
            p     = self.discriminator(speech_)

            self.disc_optimizer.zero_grad()
            loss_disc_real, loss_disc_fake = self.criterion.discriminator_adv_loss(p_hat, p)
            loss_disc_all = loss_disc_real + loss_disc_fake
            loss_disc_all.backward()
            self.disc_optimizer.step()

            # generator backward
            p_hat = self.discriminator(speech_hat_)
            with torch.no_grad():
                p = self.discriminator(speech_)

            self.optimizer.zero_grad()
            mel_loss = self.criterion.mel_loss(speech_hat_, speech_)
            adv_loss = self.criterion.generator_adv_loss(p_hat)
            fm_loss  = self.criterion.feat_match_loss(p_hat, p)
            dur_loss, pitch_loss, energy_loss = self.criterion.variance_loss(d_outs, ds, p_outs, ps, e_outs, es, x["text_lengths"])
            forwardsum_loss = self.criterion.forwardsum_loss(log_p_attn, x["text_lengths"], x["feats_lengths"])
            
            gen_loss   = mel_loss * self.criterion.lambda_mel \
                + adv_loss * self.criterion.lambda_adv \
                + fm_loss * self.criterion.lambda_feat_match
            var_loss   = (dur_loss + pitch_loss + energy_loss) * self.criterion.lambda_var
            align_loss = (forwardsum_loss + bin_loss) * self.criterion.lambda_align

            loss_gen_all = gen_loss + var_loss + align_loss
            loss_gen_all.backward()
            self.optimizer.step()

            # logging 
            losses = dict(
                loss_gen_all  = loss_gen_all.item(),
                loss_disc_all = loss_disc_all.item(),
                loss_mel      = mel_loss.item(),
                loss_align    = align_loss.item(),
                loss_duration = dur_loss.item(),
                loss_var      = var_loss.item()
            )

            msg = " - ".join([f"{k}: {round(v, 2)}" for k, v in losses.items() if v != 0])
            if self.iter % self.conf["train"]["log_step"] == 0:
                self.train_logger.log(losses, self.iter, lr=self.optimizer.param_groups[0]["lr"])
            pbar.set_postfix_str(f"with {msg}")
            pbar.set_description_str(f"Epoch {self.epoch} [{round(self.iter / self.args.max_iter * 100, 2)}%]")
            self.iter += 1

    def valid_one_epoch(self):
        self.scheduler.step()
        self.disc_scheduler.step()

        self.model.eval()
        val_loss = 0.0
        for i, batch in enumerate(self.valid_loader):
            x, speech = batch
            x         = to_device(x, device=self.device)
            outs      = self.model(**x)

            speech_hat_, _, _, start_idxs, *_ = outs
            speech_ = get_segments(
                speech.to(self.device),
                start_idxs   = start_idxs * self.model.upsample_factor,
                segment_size = self.model.segment_size * self.model.upsample_factor,
            )
            val_loss += self.criterion.mel_loss(speech_hat_, speech_).item()

        val_loss = val_loss / (i + 1)
        diff = round(((val_loss - self.best_val_loss) / self.best_val_loss), 4) if self.best_val_loss is not None else -1
        if diff <= 0:
            self.best_val_loss = val_loss
            msg = f"{text_colors.OKGREEN}{round(val_loss, 4)} (↓ {abs(diff) * 100}%){text_colors.ENDC}"
        else:
            msg = f"{text_colors.FAIL}{round(val_loss, 4)} (↑ {abs(diff) * 100}%){text_colors.ENDC}"
        system_logger.info(f"Epoch {self.epoch} - iters {self.iter}: mel-loss {msg}")
