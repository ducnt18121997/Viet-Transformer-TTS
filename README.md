<div align="center">

# End-to-end TTS system - PyTorch Implementation

This is PyTorch Implementation of **A Non-Autoregressive Transformer with unsupervised learning durations** based on Transformer & Conformer blocks, supporting a family of supervised and unsupervised duration modelings, **aiming to achieve the ultimate Text2Speech** with Vietnamese dataset, researched and developed by Dean Ng.

![Static Badge](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)
![Static Badge](https://img.shields.io/badge/-pytorch_2.0.0-%23ee4c2c?logo=pytorch&logoColor=white)
![Static Badge](https://img.shields.io/badge/-espnet_v202304-green?logo=github&logoColor=white)


</div>

### Phonemes Presentation
- [x] [Text-to-Speech Synthesis using Phoneme Concatenation](https://www.researchgate.net/publication/262638379_Text-to-Speech_Synthesis_using_Phoneme_Concatenation) (Mahwash & Shibli, 2014)

- [x] [The Effect of Tone Modeling in Vietnamese LVCSR System](https://www.sciencedirect.com/science/article/pii/S1877050916300606) (Quoc Bao et al., 2016)

- [x] [HMM-Based Vietnamese Speech Synthesis](https://ieeexplore.ieee.org/abstract/document/8919326) (Thu Trang et al, 2015)

### Architecture Design
- [x] [JETS: Jointly Training FastSpeech2 and HiFi-GAN for End to End Text to Speech](https://arxiv.org/abs/2203.16852) (Lim et al., 2022)

- [x] [VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103) (Jaehyeon Kim et al., 2021)

### Phonemes Acoustic
- [x] [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558) (Ren et al., 2020)

- [x] [Matcha-TTS: A fast TTS architecture with conditional flow matching](https://arxiv.org/abs/2309.03199) (Shivam et al., 2023)

### Audio Upsampler
- [x] [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646) (Kong et al., 2020)

### Linguistic Encoder
- [x] [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)

- [x] [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) (Gulati et al., 2020)

### Duration Modeling
- [x] [Differentiable Duration Modeling for End-to-End Text-to-Speech](https://arxiv.org/abs/2203.11049) (Nguyen et al., 2022)

- [x] [One TTS Alignment To Rule Them All](https://arxiv.org/abs/2108.10447) (Badlani et al., 2021)

### Speaker Embeddings
- [x] [ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification](https://arxiv.org/abs/2005.07143) (Desplanques et al., 2020)


<h1> 1. Installation </h1>

```bash
conda create --name venv python=3.10
conda install conda-forge::ffmpeg
# GPU
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 -c pytorch
pip install -r requirements.txt
```

<h1> 2. Train & Test </h1>

### Model Configuration
```bash
    ├── config/
    │   └── model_config.yaml 
    │   └── preprocessing_config.yaml
    │   └── train_config.yaml 

```

### Dataset Format
```bash
    /path_to_dataset[/w `speaker.json`]: 
        ├── dataset/
        │   └── speakers.json (JSON Object {`speaker_id`: `index`})
        │   └── accents.json (JSON Object {`accent_id`: `index`})
        │   └── speaker no.1/
        │       └── wavs/
        │       └── metadata.csv
        │   └── speaker no.2/
        │       └── wavs/
        │       └── metadata.csv
        |   ...
        │   └── speaker no.{n}/
        │       └── wavs/
        │       └── metadata.csv

    /path_to_dataset[/wo `speaker.json`]
        ├── dataset/
        │   └── wavs/
        |       └── ...
        │   └── metadata.csv
```

### Train & Test
``` bash
[*] python train.py 
    --task /task_name {text2wav, fastspeech2, adaspeech, jets, vits2, matcha, hifigan}
    --input_folder /path_to_input_dataset 
    --data_folder /path_to_data_save_folder 
    # use for continue training model
    --checkpoint /path_to_pretrained_checkpoint
    # use for joint-train from disconnect pre-trained
    --acoustic_checkpoint /path_to_pretrained_acoustic
    --vocoder_checkpoint /path_to_pretrained_vocoder
    # config for joint-train
    --version {fastspeech2, matcha, adaspeech}
    # use for training new speaker from base model
    --is_finetune
```

``` bash
[*] python test.py \
    --new_id /path_to_test_file 
    --acoustic_path /path_to_acoustic_checkpoint 
    # use when inference 2 model
    --vocoder_path /path_to_vocoder_checkpoint 
    --model_type {JOINT, JETS} 
    --output_folder /path_to_output_folder
```

### Experiments
During train phase, we found some experiments:

    i) Encoder and decoder block with 6 (from 4) blocks and 386 (from 256) hidden dims give better result for 22050 sample rate input audios
    
    ii) Unsupervised model after long-term training is better than supervised model (for experiment, please use supervised for faster training)
    
    iii) Comformer block take a lot of GPU when training but give better result than Transformer block

### References
- [ESPnet](https://github.com/espnet/espnet)
- [jik876's HiFi-GAN](https://github.com/jik876/hifi-gan)
- [ming024's FastSpeech2](https://github.com/ming024/FastSpeech2)
- [jaywalnut310's VITS](https://github.com/jaywalnut310/vits)
- [shivammehta25's Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)
- [TaoRuijie's ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)
- [keonlee9420's Transformer-TTS](https://github.com/keonlee9420/Comprehensive-Transformer-TTS)


### Citation
If you use this code in your research, please cite:
```bibtex
@misc{deanng_2024,
    author = {Dean Nguyen},
    title = {End-to-end TTS system - PyTorch Implementation},
    year = {2023},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/deanng/end-to-end-tts}}
}
```