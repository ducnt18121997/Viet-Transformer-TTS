from torch.utils.tensorboard import SummaryWriter


class text_colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class TensorLogger(SummaryWriter):
    def __init__(self, logdir):
        super(TensorLogger, self).__init__(logdir)

    def log(self, losses, step, state_dict=None, lr=None):
        if isinstance(losses, tuple):
            for _loss in losses:
                for k, v in _loss.items():
                    self.add_scalar(f"train/{k}", v, step)
        else:
            for k, v in losses.items():
                self.add_scalar(f"valid/{k}", v, step)

        if lr is not None:
            self.add_scalar(f"train/learning_rate", lr, step)

        if state_dict is not None:
            # plot distribution of parameters
            for tag, value in state_dict.named_parameters():
                tag = tag.replace('.', '/')
                self.add_histogram(tag, value.data.cpu().numpy(), step)
