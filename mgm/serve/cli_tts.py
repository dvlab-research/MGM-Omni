import argparse
import os
import copy
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import TextStreamer

from PIL import Image
import librosa
import whisper
import soundfile as sf

from mgm.constants import DEFAULT_SPEECH_TOKEN, AUDIO_START, AUDIO_END, AUDIO_SEP
from mgm.conversation import conv_templates
from mgm.model.builder import load_pretrained_model
from mgm.utils import disable_torch_init
from mgm.mm_utils import tokenizer_speech_token
from mgm.serve.utils import whispers_asr


def main(args):
    # Model
    disable_torch_init()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    # prepare model
    model_path = os.path.expanduser(args.model)
    tokenizer, model = load_pretrained_model(
        model_path, args.load_8bit, args.load_4bit, device=args.device)

    # prepare refer audio file
    if args.ref_audio is None:
        args.ref_audio = 'refer/default_en.wav'
    audio_refer, _ = librosa.load(args.ref_audio, sr=16000)
    audio_refer = torch.tensor(audio_refer).unsqueeze(0).to(model.device)
    if args.ref_audio_text:
        text_refer = args.ref_audio_text
    else:
        whispers_model = whisper.load_model("large-v3")
        text_refer = whispers_asr(whispers_model, args.ref_audio)
    print('Refer audio text:', text_refer)
    input_ids_refer = tokenizer(text_refer)['input_ids']
    input_ids_refer = torch.tensor(input_ids_refer).unsqueeze(0).to(model.device)

    pre_prompt_cn = '使用参考音频中听到的语气回答。'
    pre_prompt_en = 'Respond with the tone of the reference audio clip.'
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text_refer)
    pre_prompt = (pre_prompt_cn if has_chinese else pre_prompt_en)

    conv = conv_templates['qwen2vl'].copy()
    roles = conv.roles

    while True:

        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
             inp = ""
        if not inp:
            print("exit...")
            break
        print(f"{roles[1]}: ", end="")

        oup = AUDIO_START + copy.deepcopy(inp)
        inp = pre_prompt + AUDIO_START + DEFAULT_SPEECH_TOKEN + AUDIO_END + "\n"
        
        conv = conv_templates['qwen2vl'].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], oup)
        prompt = conv.get_prompt()
        input_ids = tokenizer_speech_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(model.device)

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.inference_mode():
            speech_ids, audio = model.generate(
                input_ids.clone(),
                input_ids_refer=input_ids_refer,
                audio_refer=audio_refer,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                bos_token_id=tokenizer.pad_token_id,
                eos_token_id=[tokenizer.eos_token_id],
                pad_token_id=tokenizer.pad_token_id,
                streamer=streamer,
                tokenizer=tokenizer,
                use_cache=True)

            sf.write(args.out_path, audio, 24000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="wcy1122/MGM-Omni-TTS-2B")
    parser.add_argument("--ref-audio", type=str, default="assets/ref_audio/Man_EN.wav")
    parser.add_argument("--ref-audio-text", type=str, default=None)
    parser.add_argument("--out-path", type=str, default="outputs/clone.wav")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    main(args)