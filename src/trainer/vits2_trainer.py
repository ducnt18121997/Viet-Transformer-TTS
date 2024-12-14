import os
import sys

sys.path.append(".")
import tqdm
from argparse import ArgumentParser

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from espnet2.gan_tts.utils import get_segments

from models.gan_tts.vits2 import VITS2, MultiPeriodDiscriminator
from models.gan_tts.vits2.loss import generator_loss, discriminator_loss, kl_loss, feature_loss
from log.logger import text_colors
from src.trainer.base import BaseTrainer
from src.tools.dataloader import VITS2Loader, VITS2Collate
from src.tools.tools_for_model import apply_weight, scan_checkpoint, to_device
from src.tools.tools_for_data import build_feat_extractor


class VITS2Trainer(BaseTrainer):

    def __init__(self, args: ArgumentParser, conf: dict) -> None:
        super().__init__(args, conf)    
    
    def __init_dataset__(self):

        # 1. Initialize dataloader
        self.train_loader = DataLoader(
            VITS2Loader(
                self.train_set, 
                config   = self.conf["audio"], 
                speakers = self.speakers
            ), 
            shuffle    = True, 
            batch_size = self.conf["train"]["batch_size"], 
            pin_memory = True, 
            drop_last  = True,
            collate_fn = VITS2Collate(),
        )
        self.valid_loader = DataLoader(
            VITS2Loader(
                self.test_set, 
                config   = self.conf["audio"], 
                speakers = self.speakers
            ),
            batch_size = self.conf["train"]["batch_size"], 
            collate_fn = VITS2Collate()
        )

    def __init_model__(self):
        # 0. Load pre-calculated stats
        self.stats  = None

        # 1. Initialize model
        self.model_conf = {
            "n_vocab": len(self.symbols),
            "spec_channels": self.train_loader.dataset.feats_odim,
            "n_speakers": len(self.speakers),
            **self.conf["models"]["vits2"]
        }

        self.spec_extractor = build_feat_extractor(
            choice = "fbank", 
            config = self.conf["audio"]
        ).to(self.device)
        
        self.model = VITS2(**self.model_conf).to(self.device)
        self.discriminator = MultiPeriodDiscriminator(
            self.model_conf["use_spectral_norm"]
        ).to(self.device) 
        
        duration_discriminator_type = self.model_conf.get("duration_discriminator_type", "dur_disc_1")
        if duration_discriminator_type == "dur_disc_1":
            from models.gan_tts.vits2.discriminator import DurationDiscriminatorV1 as DurationDiscriminator
        elif duration_discriminator_type == "dur_disc_2":
            from models.gan_tts.vits2.discriminator import DurationDiscriminatorV2 as DurationDiscriminator
        self.duration_discriminator = DurationDiscriminator(
            in_channels     = self.model_conf["hidden_channels"],
            filter_channels = self.model_conf["hidden_channels"],
            kernel_size     = 3,
            p_dropout       = 0.1,
            gin_channels    = self.model_conf["gin_channels"]
        ).to(self.device)
        
    def __init_loss__(self):
        # Initialize loss function
        self.loss_conf = self.conf["train"]["vits2"]["loss"]
        self.criterion = None
   
    def __init_optimizer__(self):
        # 1. Initialize optimizer
        self.optimizer  = optim.AdamW(
            self.model.parameters(), **self.conf["train"]["vits2"]["optimizer"])
        self.disc_optimizer = optim.AdamW(
            self.discriminator.parameters(), **self.conf["train"]["vits2"]["optimizer"]
        )
        self.dur_disc_optimizer = optim.AdamW(
            self.duration_discriminator.parameters(), **self.conf["train"]["vits2"]["optimizer"]
        )

        # 2. Initialize parameter from checkpoint / resume training
        self.iter, self.epoch = 0, 0
        dis_ckpt = ""
        if self.args.checkpoint is not None:
            raise NotImplementedError()
        elif os.path.exists(self.args.output_folder):
            print(f"Resuming training from last checkpoint...")
            gen_ckpt = scan_checkpoint(self.args.output_folder)
            if gen_ckpt is not None:
                dis_ckpt = os.path.join(os.path.dirname(gen_ckpt), "last_dicriminator_statedict.pt")
                
                self.model, self.optimizer = apply_weight(gen_ckpt, self.model, self.optimizer)
                self.epoch = int(os.path.basename(gen_ckpt).split("_")[-3].replace("epoch", ""))
                self.iter  = int(os.path.basename(gen_ckpt).split("_")[-2].replace("iteration", ""))

        if os.path.exists(dis_ckpt):
            print(f"Loaded checkpoint from {dis_ckpt}...")
            checkpoint_dict = torch.load(dis_ckpt)
            self.discriminator.load_state_dict(checkpoint_dict["state_dict"])
            self.disc_optimizer.load_state_dict(checkpoint_dict["optimizer"])
            try:
                self.duration_discriminator.load_state_dict(checkpoint_dict["dur_state_dict"])
                self.dur_disc_optimizer.load_state_dict(checkpoint_dict["durs_optimizer"])
            except:
                pass
        else:
            print(f"Can't detect discriminator last checkpoint, using new discriminator can make model harder to learn...")

        # 3. Initialize scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, **self.conf["train"]["vits2"]["scheduler"]
        )
        self.disc_scheduler = optim.lr_scheduler.ExponentialLR(
            self.disc_optimizer, **self.conf["train"]["vits2"]["scheduler"]
        )
        self.dur_disc_scheduler = optim.lr_scheduler.ExponentialLR(
            self.dur_disc_optimizer, **self.conf["train"]["vits2"]["scheduler"]
        )

    def train_one_epoch(self):
        self.model.train()
        self.discriminator.train()
        self.duration_discriminator.train()
        
        pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {self.epoch} [0%]", position=0, leave=False)
        for batch in pbar:
            # Update noise scalse
            if self.model_conf["use_noise_scaled_mas"] == True:
                current_mas_noise_scale = (
                    self.model.noise_scale_delta
                    - self.model.noise_scale_delta * self.iter
                )
                self.model.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)

            # Initialize input datas
            x, y = batch
            x = to_device(x, device=self.device)
            y = y.to(self.device)

            # Forward
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
            ) = self.model(**x)
            y = get_segments(y, ids_slice * self.model.dec.upsample_factor, self.model.segment_size)
            y_mel = self.spec_extractor(y.squeeze(1))[0].transpose(1, 2)
            y_hat_mel = self.spec_extractor(y_hat.squeeze(1))[0].transpose(1, 2)

            y_d_hat_r, y_d_hat_g, _, _ = self.discriminator(y, y_hat.detach())
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g
            )
            y_dur_hat_r, y_dur_hat_g = self.duration_discriminator(
                hidden_x.detach(), x_mask.detach(), logw_.detach(), logw.detach()
            )
            loss_dur_disc, losses_dur_disc_r, losses_dur_disc_g = discriminator_loss(
                y_dur_hat_r, y_dur_hat_g
            )
            self.dur_disc_optimizer.zero_grad()
            loss_dur_disc.backward()
            self.dur_disc_optimizer.step()

            self.disc_optimizer.zero_grad()
            loss_disc.backward()
            self.disc_optimizer.step()

            # Generator forward
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(y, y_hat)
            y_dur_hat_r, y_dur_hat_g = self.duration_discriminator(hidden_x, x_mask, logw_, logw)

            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.loss_conf["c_mel"]
            loss_kl  = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.loss_conf["c_kl"]

            loss_fm  = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

            loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
            loss_gen_all += loss_dur_gen

            self.optimizer.zero_grad()
            loss_gen_all.backward()
            self.optimizer.step()

            # logging 
            losses = dict(
                loss_gen_all  = loss_gen_all.item(),
                loss_disc_all = loss_disc.item(),
                loss_dur_disc_all = loss_dur_disc.item(),
                loss_mel      = loss_mel.item() / self.loss_conf["c_mel"],
                loss_duration = loss_dur.item(),
                loss_kl       = loss_kl.item() / self.loss_conf["c_kl"],
                loss_fm       = loss_fm.item()
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
        self.dur_disc_scheduler.step()
        
        self.model.eval()
        val_loss = 0.0
        for i, batch in enumerate(self.valid_loader):
            x, y = batch
            x = to_device(x, device=self.device)
            y = y.to(self.device)

            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
            ) = self.model(**x)
            y = get_segments(y, ids_slice * self.model.dec.upsample_factor, self.model.segment_size)
            # y_mel = slice_segments(x["feats"], ids_slice, self.model.segment_size // self.model.dec.upsample_factor)
            y_mel = self.spec_extractor(y.squeeze(1))[0].transpose(1, 2)
            y_hat_mel = self.spec_extractor(y_hat.squeeze(1))[0].transpose(1, 2)
            val_loss += F.l1_loss(y_mel, y_hat_mel).item()

        val_loss = val_loss / (i + 1)
        diff = round(((val_loss - self.best_val_loss) / self.best_val_loss), 4) if self.best_val_loss is not None else -1
        if diff <= 0:
            self.best_val_loss = val_loss
            msg = f"{text_colors.OKGREEN}{round(val_loss, 4)} (↓ {abs(diff) * 100}%){text_colors.ENDC}"
        else:
            msg = f"{text_colors.FAIL}{round(val_loss, 4)} (↑ {abs(diff) * 100}%){text_colors.ENDC}"
        print(f"Epoch {self.epoch} - iters {self.iter}: mel-loss {msg}")



