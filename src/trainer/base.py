import os
import yaml
import json
import torch
from abc import ABC
from datetime import datetime
from argparse import ArgumentParser
from modules.g2p.symbols import symbols
from log.logger import TensorLogger
from loguru import logger as system_logger
from src.tools.tools_for_model import clear_disk


class BaseTrainer(ABC):
    """ Customize TTS Trainer """
    def __init__(self, args: ArgumentParser, conf: dict) -> None:
        super().__init__()

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.symbols = symbols
        self.args = args
        self.conf = conf

        # 0. Initialize speakers/accents
        self.speakers = None
        if os.path.exists(os.path.join(self.args.input_folder, "speakers.json")):
            self.speakers = json.load(open(os.path.join(self.args.input_folder, "speakers.json"), "r"))

        self.accents = None
        if os.path.exists(os.path.join(self.args.input_folder, "accents.json")):
            self.accents = json.load(open(os.path.join(self.args.input_folder, "accents.json"), "r"))

        # 1. Initialize dataset
        if not os.path.isfile(os.path.join(self.args.input_folder, "train.txt")):
            if self.speakers is None:
                from src.tools.tools_for_data import create_unknown_filelist as create_filelist
            else:
                from src.tools.tools_for_data import create_filelist
            create_filelist(self.args.data_folder, self.speakers, self.args.input_folder)
        self.train_set = [x.split("|") for x in open(os.path.join(self.args.input_folder, "train.txt"), "r", encoding="utf8") if x]
        self.test_set  = [x.split("|") for x in open(os.path.join(self.args.input_folder, "test.txt"), "r", encoding="utf8") if x]
        
        # 2. Initialize base configuration
        self.__init_parameter__()
        self.__init_dataset__()
        self.__init_model__()
        self.__init_loss__()
        self.__init_optimizer__()
        system_logger.info("==================================== Training Configuration =====================================")
        system_logger.info(f" ---> Number of Model Parameters: {get_param_num(self.model)}")
        if self.discriminator is not None:
            system_logger.info(f" ---> Number of Multi Discriminator Parameters: {get_param_num(self.discriminator)}")
        system_logger.info(f" ---> Number of used GPU: {1}")
        system_logger.info(f" ---> Total train samples: {self.train_loader.dataset.__len__()}")
        system_logger.info(f" ---> Total valid samples: {self.valid_loader.dataset.__len__()}")
        system_logger.info(f" ---> Batch size in total: {self.conf['train']['batch_size']}")
        system_logger.info("=================================================================================================")

        os.makedirs(self.args.output_folder, exist_ok=True)
        self.train_logger = TensorLogger(os.path.join(self.args.output_folder, "log/train"))
        self.valid_logger = TensorLogger(os.path.join(self.args.output_folder, "log/valid"))

        self.save_information()
        self.best_val_loss = None

    def __init_parameter__(self):
        # NOTE(by deanng): This function will be implemented when fine-tune/adapt new VoiceAI
        pass
    
    def __init_dataset__(self):
        """ Initialize dataloader with dataset """
        self.train_loader = None
        self.valid_loader = None

        raise NotImplementedError()

    def __init_model__(self):
        """ Initialize model """
        self.model = None
        self.discriminator = None
        
        raise NotImplementedError()
    
    def __init_loss__(self):
        """ Initialize loss function """
        self.criterion = None

        raise NotImplementedError()

    def __init_optimizer__(self):
        """ Initialize optimizer and scheduler with model """
        self.optimizer = None
        self.scheduler = None

        self.disc_optimizer = None
        self.disc_scheduler = None

        raise NotImplementedError()

    def train_one_epoch(self):
        """ Training model through one epoch """

        return NotImplementedError()

    def valid_one_epoch(self):
        """ Valid model at end of epoch """

        return NotImplementedError()

    def save_model(self):
        prefix = "generator" if self.discriminator is not None else "model"
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            },
            os.path.join(self.args.output_folder, f"{prefix}_epoch{self.epoch}_iteration{self.iter}_statedict.pt")
        )
        if self.discriminator is not None:
            torch.save(
                {
                    "state_dict": self.discriminator.state_dict(),
                    "optimizer": self.disc_optimizer.state_dict()
                },
                os.path.join(self.args.output_folder, "last_dicriminator_statedict.pt")
            )
        clear_disk(self.args.output_folder, prefix)

    def save_information(self):
        os.makedirs(self.args.output_folder, exist_ok=True)
        with open(os.path.join(self.args.output_folder, "config.yaml"), "w", encoding="utf8") as f:
            yaml.dump({"audio": self.conf["audio"], "model": self.model_conf}, f, default_flow_style=False)

        if self.speakers is not None:
            with open(os.path.join(self.args.output_folder, "speakers.json"), "w", encoding="utf8") as f:
                json.dump(self.speakers, f, ensure_ascii=False, indent=4)

        if self.accents is not None:
            with open(os.path.join(self.args.output_folder, "accents.json"), "w", encoding="utf8") as f:
                json.dump(self.accents, f, ensure_ascii=False, indent=4)

        if self.stats is not None:
            with open(os.path.join(self.args.output_folder, "stats.json"), "w", encoding="utf8") as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=4)

    def run(self):
        system_logger.success(f"Training started at {datetime.today().strftime('%Y/%m/%d %H:%M:%S')}...")
        while True:
            self.train_one_epoch()
            self.valid_one_epoch()

            self.save_model()

            # End training model
            if self.iter >= self.args.max_iter:
                break
            self.epoch += 1

        system_logger.success(f"Training finished at {datetime.today().strftime('%Y/%m/%d %H:%M:%S')}...")
        

def show_params(nnet):
    system_logger.info("=" * 40, "Model Parameters", "=" * 40)
    num_params = 0
    for module_name, m in nnet.named_modules():
        if module_name == '':
            for name, params in m.named_parameters():
                print(name, params.size())
                i = 1
                for j in params.size():
                    i = i * j
                num_params += i

    system_logger.info('[*] Parameter Size: {}'.format(num_params))
    system_logger.info("=" * 100)


def show_model(nnet):
    system_logger.info("=" * 40, "Model Structures", "=" * 40)
    for module_name, m in nnet.named_modules():
        if module_name == '':
            print(m)

    system_logger.info("=" * 100)


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())

    return num_param
