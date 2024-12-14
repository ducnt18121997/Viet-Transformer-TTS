import os
import sys

sys.path.append(".")
import json
import shutil
from argparse import ArgumentParser
from loguru import logger as system_logger

import torch
from torch.utils.data import DataLoader

from src.trainer.joint_trainer import JointTrainer


class JointFinetuner(JointTrainer):

    def __init__(self, args: ArgumentParser, conf: dict) -> None:
        super().__init__(args, conf)
        self.threshold_epoch = 50

    def __init_parameter__(self):
        assert self.args.checkpoint is not None, \
            f"Adapter VoiceAI Task must have pre-trained model, give `{self.args.checkpoint}`"

        # 0. Verify speaker
        src_speakers = json.load(open(os.path.join(os.path.dirname(self.args.checkpoint), "speakers.json"), "r", encoding="utf8"))
        tgt_speakers = json.load(open(os.path.join(self.args.input_folder, "speakers.json"), "r", encoding="utf8"))
        if src_speakers == tgt_speakers:
            raise NotImplementedError(f"=> Speakers is still same! Training is not effective..")
        self.adapter_speakers = [spk for spk in tgt_speakers if spk not in src_speakers]

        checkpoint = torch.load(self.args.checkpoint, map_location="cpu")
        # 1. Re-initialize speakers embedding
        system_logger.success(f"Start initialize for new-speakers: {', '.join(self.adapter_speakers)}")
        old_embeds = checkpoint["state_dict"]["text2mel.speaker_emb.cvae.embedding_layer_spk.weight"]
        new_embeds = torch.rand((len(tgt_speakers), old_embeds.size(1)))
        for spk in src_speakers.keys():
            system_logger.info(f"- Re-parameterized speaker {spk} from checkpoint")
            new_embeds[tgt_speakers[spk]] = old_embeds[src_speakers[spk]]
        checkpoint["state_dict"]["text2mel.speaker_emb.cvae.embedding_layer_spk.weight"] = new_embeds

        # 2. Re-initialize accents embedding
        src_accents = json.load(open(os.path.join(os.path.dirname(self.args.checkpoint), "accents.json"), "r", encoding="utf8"))
        tgt_accents = json.load(open(os.path.join(self.args.input_folder, "accents.json"), "r", encoding="utf8"))
        self.adapter_accents = [spk for spk in tgt_accents if spk not in src_accents]
        if src_accents != tgt_accents:
            system_logger.success(f"Start initialize for new-accents: {', '.join(self.adapter_accents)}")
            old_embeds = checkpoint["state_dict"]["text2mel.speaker_emb.cvae.embedding_layer_acc.weight"]
            new_embeds = torch.rand((len(tgt_accents), old_embeds.size(1)))
            for acc in src_accents.keys():
                system_logger.info(f"- Re-parameterized accent {acc} from checkpoint")
                new_embeds[tgt_accents[acc]] = old_embeds[src_accents[acc]]
                # else:
                #     system_logger.info(f"- Re-parameterized accent {k}")
                #     new_embeds.append(old_embeds[0])
            checkpoint["state_dict"]["text2mel.speaker_emb.cvae.embedding_layer_acc.weight"] = new_embeds

        # 2. Save parameter
        os.makedirs(self.args.output_folder, exist_ok=True)
        torch.save({"state_dict": checkpoint["state_dict"]}, os.path.join(self.args.output_folder, "generator_epoch0_iteration0_statedict.pt"))
        shutil.copy(os.path.join(os.path.dirname(self.args.checkpoint), "last_dicriminator_statedict.pt"), self.args.output_folder)
        shutil.copy(os.path.join(os.path.dirname(self.args.checkpoint), "stats.json"), self.args.output_folder)
        self.args.checkpoint = None

    def __init_dataset__(self):
        super().__init_dataset__()
        self.all_train_loader = self.train_loader
        if self.version in ["fastspeech2", "adaspeech"]:
            from src.tools.dataloader import Fastspeech2Loader as CustomLoader
            from src.tools.dataloader import Fastspeech2Collate as CustomCollate
        else:
            from src.tools.dataloader import MatchaLoader as CustomLoader
            from src.tools.dataloader import MatchaCollate as CustomCollate
        self.adapter_train_loader = DataLoader(
            CustomLoader(
                [x.split("|") for x in open(os.path.join(self.args.input_folder, "train.txt"), "r", encoding="utf8") \
                    if x and x.split("|")[1] in self.adapter_speakers], 
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

    def train_one_epoch(self):
        self.train_loader = self.all_train_loader if self.epoch >= self.threshold_epoch else self.adapter_train_loader
        super().train_one_epoch()

    def valid_one_epoch(self):
        if self.epoch >= self.threshold_epoch:
            super().valid_one_epoch()


""" REPLACING BY PRE-TRAINED EMBEDDINGS
# 1. Initialize speakers embedding
info_path = os.path.join(self.args.input_folder, "info.json")
if not os.path.exists(info_path):
    raise NotImplementedError(f"Can't automatic initialize, please create `info.json` file")
info_datas = json.load(open(info_path, "r", encoding="utf8"))
system_logger.success(f"Start initialize for new-speakers: {', '.join(self.adapter_speakers)}")
old_embeds = checkpoint["state_dict"]["text2mel.speaker_emb.cvae.embedding_layer_spk.weight"]
new_embeds = []
for k, v in tgt_speakers.items():
    if k in src_speakers:
        system_logger.info(f"- Copying base speaker {k} from checkpoint")
        new_embeds.append(old_embeds[v])
    else:
        map_spk = info_datas[k]
        system_logger.info(f"- Replacing speaker {k} by {map_spk}")
        new_embeds.append(old_embeds[src_speakers[map_spk]])
new_embeds = torch.stack(new_embeds, dim=0)
checkpoint["state_dict"]["text2mel.speaker_emb.cvae.embedding_layer_spk.weight"] = new_embeds

# 2. Initialize accents embedding
src_accents = json.load(open(os.path.join(os.path.dirname(self.args.checkpoint), "accents.json"), "r", encoding="utf8"))
tgt_accents = json.load(open(os.path.join(self.args.input_folder, "accents.json"), "r", encoding="utf8"))
self.adapter_accents = [spk for spk in tgt_accents if spk not in src_accents]
if src_accents != tgt_accents:
    system_logger.success(f"Start initialize for new-accents: {', '.join(self.adapter_accents)}")
    old_embeds = checkpoint["state_dict"]["text2mel.speaker_emb.cvae.embedding_layer_acc.weight"]
    new_embeds = []
    for k, v in tgt_accents.items():
        if k in src_accents:
            system_logger.info(f"- Copying base accent {k} from checkpoint")
            new_embeds.append(old_embeds[v])
        else:
            system_logger.info(f"- Re-parameterized accent {k}")
            new_embeds.append(old_embeds[0])
    new_embeds = torch.stack(new_embeds, dim=0)
    checkpoint["state_dict"]["text2mel.speaker_emb.cvae.embedding_layer_acc.weight"] = new_embeds
"""