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

from models.gan_tts.text2wav import Text2Wav
from models.gan_tts.hifigan import (
    HiFiGANMultiScaleMultiPeriodDiscriminator, 
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
    MelSpectrogramLoss
)
from log.logger import text_colors
from src.trainer.base import BaseTrainer
from src.tools.tools_for_model import apply_weight, scan_checkpoint, to_device


class JointTrainer(BaseTrainer):

    def __init__(self, args: ArgumentParser, conf: dict) -> None:
        super().__init__(args, conf)
    
    def __init_dataset__(self):
        self.version = self.args.version

        # 0. Load pre-calculated stats
        self.stats  = None
        if not self.args.is_finetune: 
            if self.args.acoustic_checkpoint is not None and os.path.exists(os.path.join(os.path.dirname(self.args.acoustic_checkpoint), "stats.json")):
                self.stats = json.load(open(os.path.join(os.path.dirname(self.args.acoustic_checkpoint), "stats.json"), "r"))
            elif self.args.checkpoint is not None and os.path.exists(os.path.join(os.path.dirname(self.args.checkpoint), "stats.json")):
                self.stats = json.load(open(os.path.join(os.path.dirname(self.args.checkpoint), "stats.json"), "r"))
            if os.path.exists(os.path.join(self.args.output_folder, "stats.json")):
                self.stats = json.load(open(os.path.join(self.args.output_folder, "stats.json"), "r"))

        # 1. Initialize dataloader
        if self.version in ["fastspeech2", "adaspeech"]:
            self.conf["audio"]["self_learning"] = self.conf["models"][self.version]["variance"]["learn_alignment"]
            from src.tools.dataloader import Fastspeech2Loader as CustomLoader
            from src.tools.dataloader import Fastspeech2Collate as CustomCollate
        else:
            self.conf["audio"]["self_learning"] = True
            from src.tools.dataloader import MatchaLoader as CustomLoader
            from src.tools.dataloader import MatchaCollate as CustomCollate
        self.train_loader = DataLoader(
            CustomLoader(
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
            collate_fn=CustomCollate(
                n_speakers      = len(self.speakers) if self.speakers is not None else -1, 
                use_accent      = self.conf["models"][self.version]["use_cvae"],
                return_waveform = True
            )
        )
        self.stats = self.train_loader.dataset.stats
        print(json.dumps(self.stats, ensure_ascii=False, indent=4))
        
        self.valid_loader = DataLoader(
            CustomLoader(
                self.test_set, 
                config   = self.conf["audio"], 
                speakers = self.speakers, 
                accents  = self.accents, 
                stats    = self.stats
            ), 
            batch_size = self.conf["train"]["batch_size"], 
            collate_fn = CustomCollate(
                n_speakers      = len(self.speakers) if self.speakers is not None else -1, 
                use_accent      = self.conf["models"][self.version].get("use_cvae", False),
                return_waveform = True
            )
        )

    def __init_model__(self):
        # Initialize model
        self.model_conf = {
            "idim": len(self.symbols),
            "odim": self.train_loader.dataset.feats_odim,
            "version": self.version,
            "text2mel_conf": {
                "n_speakers": len(self.speakers) if self.speakers is not None else -1,
                "n_accents": len(self.accents) if self.accents is not None else -1,
                "hparams": self.conf["models"][self.version],
            },
            "mel2wav_conf": self.conf["models"]["hifigan"]
        }
        
        self.model = Text2Wav(
            vocabs          = self.model_conf["idim"],
            aux_channels    = self.model_conf["odim"],
            version         = self.version,
            text2mel_params = {**self.model_conf["text2mel_conf"], "stats": self.stats},
            mel2wav_params  = self.model_conf["mel2wav_conf"],
        ).to(self.device)
        self.discriminator = HiFiGANMultiScaleMultiPeriodDiscriminator().to(self.device)

    def __init_loss__(self):
        # Initialize loss function
        if self.version in ["fastspeech2", "adaspeech"]:
            self.loss_conf = self.conf["train"][self.version]["loss"]
            self.loss_conf.update(dict(
                pitch_feature_level  = self.conf["models"][self.version]["variance"]["variance_embedding"]["pitch_feature"], 
                energy_feature_level = self.conf["models"][self.version]["variance"]["variance_embedding"]["energy_feature"],
            ))
            if self.version == "fastspeech2":
                from models.tts.fastspeech2 import FastSpeech2Loss as ModelLoss
            else:
                from models.tts.adaspeech import AdaSpeechLoss as ModelLoss
            self.criterion = ModelLoss(self.args.max_iter, self.loss_conf, is_finetune=True if self.args.acoustic_checkpoint or self.args.checkpoint else False)
        else:
            self.loss_conf = {"n_channels": self.conf["audio"]["mel"]["channels"], **self.conf["train"]["matcha"]["loss"]}
            from models.tts.matcha import MatchaTTSLoss
            self.criterion = MatchaTTSLoss(self.args.max_iter, self.loss_conf, is_finetune=True if self.args.acoustic_checkpoint or self.args.checkpoint else False)

        self.melspectrogram_loss = MelSpectrogramLoss(**{
            "fs": self.conf["audio"]["signal"]["sampling_rate"],
            "n_fft": self.conf["audio"]["stft"]["filter_length"],
            "hop_length": self.conf["audio"]["stft"]["hop_length"],
            "win_length": self.conf["audio"]["stft"]["win_length"],
            "window": self.conf["audio"]["stft"]["window"],
            "n_mels": self.conf["audio"]["mel"]["channels"],
            "fmin": self.conf["audio"]["mel"]["fmin"],
            "fmax": self.conf["audio"]["mel"]["fmax"],
            "log_base": self.conf["audio"]["mel"]["log_base"]
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
            filter(lambda p: p.requires_grad, self.model.parameters()), **self.conf["train"][self.version]["optimizer"]
        )
        self.disc_optimizer = optim.AdamW(
            self.discriminator.parameters(), **self.conf["train"]["hifigan"]["optimizer"]
        )
        
        # 2. Initialize parameter from checkpoint / resume training
        self.iter, self.epoch = 0, 0
        dis_ckpt = ""
        if self.args.checkpoint is not None:
            dis_ckpt = os.path.join(os.path.dirname(self.args.checkpoint), "last_dicriminator_statedict.pt")
            self.model, self.optimizer = apply_weight(self.args.checkpoint, self.model, self.optimizer)
            system_logger.info(f"Initial training from checkpoint...")
        elif self.args.acoustic_checkpoint is not None:
            self.model.text2mel = apply_weight(self.args.acoustic_checkpoint, self.model.text2mel)
            system_logger.info(f"Loaded checkpoint for text2mel from {self.args.acoustic_checkpoint}...")
            if self.args.vocoder_checkpoint is not None:
                dis_ckpt = os.path.join(os.path.dirname(self.args.vocoder_checkpoint), "last_dicriminator_statedict.pt")
                self.model.mel2wav = apply_weight(self.args.vocoder_checkpoint, self.model.mel2wav)
                system_logger.info(f"Loaded checkpoint for mel2wav from {self.args.vocoder_checkpoint}...")
            else:
                system_logger.info(f"Training without pretrained vocoder can make model harder/slower to converge...")
        elif os.path.exists(self.args.output_folder):
            gen_ckpt = scan_checkpoint(self.args.output_folder)
            if gen_ckpt is not None:
                dis_ckpt = os.path.join(os.path.dirname(gen_ckpt), "last_dicriminator_statedict.pt")
                self.model, self.optimizer = apply_weight(gen_ckpt, self.model, self.optimizer)
                self.epoch = int(os.path.basename(gen_ckpt).split("_")[-3].replace("epoch", "")) + 1
                self.iter  = int(os.path.basename(gen_ckpt).split("_")[-2].replace("iteration", ""))
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
            x, y = batch
            x    = to_device(x, device=self.device)
            y    = to_device(y, device=self.device)
            y_pred, y_extra, start_idxs = self.model(**x, step=self.iter)

            speech_hat_ = y_pred[0]
            speech_     = get_segments(
                y[0], 
                start_idxs   = start_idxs * self.model.upsample_factor, 
                segment_size = self.model.segment_size * self.model.upsample_factor
            )

            # discriminator backward
            p_hat = self.discriminator(speech_hat_.detach())
            p     = self.discriminator(speech_)
            loss_disc_real, loss_disc_fake = self.discriminator_adv_loss(p_hat, p)

            self.disc_optimizer.zero_grad()
            loss_disc_all = loss_disc_real + loss_disc_fake
            loss_disc_all.backward()
            self.disc_optimizer.step()

            # self.model backward
            p_hat = self.discriminator(speech_hat_)
            with torch.no_grad():
                p = self.discriminator(speech_)

            loss_gen = self.generator_adv_loss(p_hat)
            loss_fm  = self.feat_match_loss(p_hat, p) * 2
            loss_mel = self.melspectrogram_loss(speech_hat_, speech_) * 45
            if self.version in ["fastspeech2", "adaspeech"]:
                feat_ = get_segments(y[1].transpose(1, 2), start_idxs, self.model.segment_size)
                loss_var = self.criterion(
                    predictions = y_pred[1: ], 
                    targets     = (feat_, ) + y[2: ] + y_extra, 
                    is_joint    = True, 
                    step        = self.iter
                )
            else:
                feat_ = get_segments(y[1], start_idxs, self.model.segment_size)
                u_    = get_segments(y_pred[2], start_idxs, self.model.segment_size)
                mu_y_ = get_segments(y_pred[3], start_idxs, self.model.segment_size)
                loss_var = self.criterion(
                    predictions = (y_pred[1], u_, mu_y_) + y_pred[4: ], 
                    targets     = (feat_, ) + y[2: ], 
                    is_joint    = True,
                    step        = self.iter
                )
            self.optimizer.zero_grad()
            loss_gen_all = loss_gen + loss_fm + loss_mel + sum([_loss for _loss in loss_var.values()])
            loss_gen_all.backward()
            self.optimizer.step()

            # logging 
            losses = dict(
                loss_gen_all  = loss_gen_all.item(),
                loss_disc_all = loss_disc_all.item(),
                loss_mel      = loss_mel.item(),
                **{f"loss_{k}": v.item() for k, v in loss_var.items()}
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
        val_loss = {"loss_mel": 0.0, "loss_feat/diff": 0.0}
        for i, batch in enumerate(self.valid_loader):
            x, y = batch
            x    = to_device(x, device=self.device)
            y    = to_device(y, device=self.device)
            y_pred, y_extra, start_idxs = self.model(**x)

            speech_hat_ = y_pred[0]
            speech_     = get_segments(
                x            = y[0],
                start_idxs   = start_idxs * self.model.upsample_factor,
                segment_size = self.model.segment_size * self.model.upsample_factor,
            )
            if self.version in ["fastspeech2", "adaspeech"]:
                feat_ = get_segments(y[1].transpose(1, 2), start_idxs, self.model.segment_size)
                val_loss["loss_feat/diff"] += self.criterion(
                    predictions = y_pred[1: ], 
                    targets     = (feat_, ) + y[2: ] + y_extra, 
                    is_joint    = True, 
                )["feat"].item()
            else:
                feat_ = get_segments(y[1], start_idxs, self.model.segment_size)
                u_    = get_segments(y_pred[2], start_idxs=start_idxs, segment_size=self.model.segment_size)
                mu_y_ = get_segments(y_pred[3], start_idxs=start_idxs, segment_size=self.model.segment_size)
                val_loss["loss_feat/diff"] += self.criterion(
                    predictions =  (y_pred[1], u_, mu_y_) + y_pred[4: ], 
                    targets     =  (feat_, ) + y[2: ], 
                    is_joint    = True
                )["diff"].item()
            val_loss["loss_mel"] += self.melspectrogram_loss(speech_hat_, speech_).item()
        val_loss = {k: v / (i + 1) for k, v in val_loss.items()}

        self.valid_logger.log(val_loss, self.iter, state_dict=self.model)
        count, msg = 0, []
        for _loss in val_loss:
            if val_loss[_loss] == 0:
                count += 1
                continue
            diff = round(((val_loss[_loss] - self.best_val_loss[_loss]) / self.best_val_loss[_loss]), 4) \
                if self.best_val_loss is not None else -1
            if diff <= 0:
                msg.append(f"{_loss} {text_colors.OKGREEN}{round(val_loss[_loss], 4)} (↓ {abs(diff) * 100}%){text_colors.ENDC}")
                count += 1
            else:
                msg.append(f"{_loss} {text_colors.FAIL}{round(val_loss[_loss], 4)} (↑ {abs(diff) * 100}%){text_colors.ENDC}")
        self.best_val_loss = val_loss.copy() if count == len(val_loss) else self.best_val_loss
        system_logger.info(f"Epoch {self.epoch}: {' - '.join(msg)}")
