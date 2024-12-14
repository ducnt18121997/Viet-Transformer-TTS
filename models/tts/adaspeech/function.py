import torch
from models.tts.fastspeech2.function import pad_2D, frame2phoneme


def get_mel_phoneme_level(frame_feature, src_len, duration, max_len=None):
    phoneme_feature = torch.from_numpy(
        pad_2D(
            [frame2phoneme(dur[:len], var) for dur, len, var \
                    in zip(duration.int().cpu().numpy(), src_len.cpu().numpy(), frame_feature.cpu().numpy())]
        )
    ).float()
    
    if max_len != phoneme_feature.size(1):
        _phoneme_feature = torch.zeros(frame_feature.size(0), max_len)
        for i in range(len(_phoneme_feature)):
            _phoneme_feature[i][: phoneme_feature[i].size(0)] = phoneme_feature[i]
        phoneme_feature = _phoneme_feature

    return phoneme_feature.to(frame_feature.device)
