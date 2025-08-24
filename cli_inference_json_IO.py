import argparse
import json

import torch
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

from config import Config
from models.salmonn import SALMONN
from utils import prepare_one_sample


parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", type=str, required=True, help="path to configuration file")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
parser.add_argument("--json-path", type=str, required=True, help="input json file path")
parser.add_argument("--output-json", type=str, required=True, help="output json file path")

args = parser.parse_args()
cfg = Config(args)

model, ckpt = SALMONN.from_config(cfg.config.model)
model.to(args.device)
model.eval()

wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

with open(args.json_path, "r") as f:
    data = json.load(f)

annotations = data.get("annotation", [])

for idx, item in enumerate(tqdm(annotations, desc="Processing audio files"), 1):
    try:
        wav_path = item["path"]
        prompt = "Recognize the speech and give me the transcription."
        samples = prepare_one_sample(wav_path, wav_processor)
        prompt = [
            cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt.strip())
        ]
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = model.generate(samples, cfg.config.generate, prompts=prompt)[0]
        item["llama_output"] = output
    except Exception as e:
        item["llama_output"] = f"Error: {e}"

    # 每处理150条打印一次
    if idx % 150 == 0:
        tqdm.write(f"[{idx}] path={item.get('path','')}\n{item['llama_output']}\n")

with open(args.output_json, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
