import os
import time
import torch
import requests
import argparse
from src.api.modules.tts import OneStageTTS, TwoStageTTS
from modules.upload.api import save_to_local


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_id", type=str, default=None,
                        help="directly to dataset folder")
    parser.add_argument("--list_id", type=str, default=None,
                        help="path to list of news id")
    parser.add_argument("--acoustic_path", type=str, default=None,
                        help="directory to acoustic checkpoint")
    parser.add_argument("--vocoder_path", type=str, default=None,
                        help="directory to vocoder checkpoint")
    parser.add_argument("--model_type", type=str, default="JOINT", 
                        choices=["JOINT", "JETS", "VITS2" ,"fastspeech2", "matcha"])
    parser.add_argument("--output_folder", type=str, default="data/simulated/",
                        help="path to save output audio")
    parser.add_argument("--format", type=str, default="m4a")
    args = parser.parse_args()

    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # model initialization
    if args.acoustic_path is None:
        raise NotImplementedError(
            f"Expect acoustic_path, given `args.acoustic_path = {args.acoustic_path}`"
        )
    elif args.vocoder_path is not None:
        nnet = TwoStageTTS(args.acoustic_path, args.vocoder_path, model_type=args.model_type.lower())
    else:
        nnet = OneStageTTS(args.acoustic_path, model_type=args.model_type.upper())

    # input/output initialization
    print("[*] Start processing:")
    if args.new_id is None and args.list_id is None:
        raise ValueError("Expect `new_id` or `list_id`, both is None ")
    
    if args.list_id is not None:
        list_ids = [x for x in open(args.list_id, "r", encoding="utf8").read().split("\n") if x]
    else:
        list_ids = [args.new_id]
    for nid in list_ids:
        os.makedirs(f"{os.path.join(output_folder, os.path.basename(nid))}", exist_ok=True)
        print(f"[*] Processing for {nid}")
        input_texts = requests.get(f"https://speech.aiservice.vn/tts/bot/check?filename={nid}").json()["content"]
        with open(os.path.join(f"{os.path.join(output_folder, os.path.basename(nid))}/transcipt.txt"), "w", encoding="utf8") as f:
            f.write(input_texts)
        
        input_sequences = requests.post("https://speech.aiservice.vn//tts-normalization", json={"text": input_texts}).json()["result"]
        print("\n".join(input_sequences))
        if nnet.accents is None:
            for spk in nnet.speakers:
                print(f"***Initialize for {spk}***")

                tik = time.time()
                audio_generated = nnet(texts=input_sequences, speaker_id=spk)
                print(f"- Inference time: {time.time() - tik} seconds")

                tik = time.time()
                torch.cuda.empty_cache()
                print(f"- Remove cache time: {time.time() - tik} seconds")

                tik = time.time()
                save_path = f"{os.path.join(output_folder, os.path.basename(nid))}/{spk}.{args.format}"
                # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_to_local(
                    datas=audio_generated,
                    save_path=save_path,
                    audio_format=args.format
                )
                print(f"- Save audio time: {time.time() - tik} seconds")
        else:
            for spk in nnet.accents:
                for acc in nnet.accents[spk]:
                    print(f"***Initialize for {spk} with {acc}***")

                    tik = time.time()
                    audio_generated = nnet(texts=input_sequences, speaker_id=spk, accent_id=acc)
                    print(f"- Inference time: {time.time() - tik} seconds")

                    tik = time.time()
                    torch.cuda.empty_cache()
                    print(f"- Remove cache time: {time.time() - tik} seconds")

                    tik = time.time()
                    save_path = f"{os.path.join(output_folder, os.path.basename(nid))}/{spk}.{acc}.{args.format}"
                    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    save_to_local(
                        datas=audio_generated,
                        save_path=save_path,
                        audio_format=args.format
                    )
                    print(f"- Save audio time: {time.time() - tik} seconds")
