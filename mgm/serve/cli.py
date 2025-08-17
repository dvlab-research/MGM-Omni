import argparse
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import TextStreamer

from PIL import Image
import librosa
import whisper
import soundfile as sf

from mgm.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_SPEECH_TOKEN, AUDIO_START, AUDIO_END, AUDIO_SEP
from mgm.conversation import conv_templates
from mgm.model import *
from mgm.model.builder import load_pretrained_model
from mgm.utils import disable_torch_init
from mgm.mm_utils import tokenizer_image_speech_token, tokenizer_speech_token
from mgm.serve.utils import whispers_asr, preprocess_image_qwen2vl, process_visual_input, process_audio_input


def main(args):
    # Model
    disable_torch_init()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    
    # prepare model
    tokenizer, tokenizer_speech, model, image_processor, speech_processor = \
        load_pretrained_model(
            args.model, args.load_8bit, args.load_4bit,
            speechlm_path=args.speechlm, use_flash_attn=True, device=args.device,
        )

    conv = conv_templates['qwen2vl'].copy()
    roles = conv.roles
    
    # prepare image & speech file
    image_aspect_ratio = getattr(model.config, 'image_aspect_ratio', 'qwen2vl')
    image_tensor = process_visual_input(args.image_file, args.video_file, image_processor, image_aspect_ratio)
    speech_tensor = process_audio_input(args.audio_file, speech_processor)

    if image_tensor is not None:
        if isinstance(image_tensor, dict):
            for key in image_tensor.keys():
                image_tensor[key] = image_tensor[key].to(dtype=model.dtype, device=model.device, non_blocking=True)
        else:
            image_tensor = image_tensor.to(dtype=model.dtype, device=model.device, non_blocking=True)
        image_tensor = [image_tensor]
        images = image_tensor
    else:
        images = None
    
    if speech_tensor is not None:
        speech_tensor = [speech_tensor.to(dtype=model.dtype, device=model.device, non_blocking=True)]
        speeches = speech_tensor
    else:
        speeches = None

    # prepare refer audio file
    audio_refer, _ = librosa.load(args.ref_audio, sr=16000)
    audio_refer = torch.tensor(audio_refer).unsqueeze(0).to(model.device)
    if args.ref_audio_text:
        text_refer = args.ref_audio_text
    else:
        whispers_model = whisper.load_model("large-v3")
        text_refer = whispers_asr(whispers_model, args.ref_audio)
    print('Refer audio text:', text_refer)
    input_ids_refer = tokenizer_speech(text_refer)['input_ids']
    input_ids_refer = torch.tensor(input_ids_refer).unsqueeze(0).to(model.device)

    # prepare input for SpeechLM
    pre_prompt_cn = '使用参考音频中听到的语气回答。'
    pre_prompt_en = 'Respond with the tone of the reference audio clip.'
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text_refer)
    pre_prompt_speech = (pre_prompt_cn if has_chinese else pre_prompt_en)
    inp_speech = pre_prompt_speech + AUDIO_START + DEFAULT_SPEECH_TOKEN + AUDIO_END + "\n"
    conv_speech = conv_templates['qwen2vl'].copy()
    conv_speech.append_message(conv_speech.roles[0], inp_speech)
    conv_speech.append_message(conv_speech.roles[1], AUDIO_START)
    prompt_speech = conv_speech.get_prompt()
    input_ids_speech = tokenizer_speech_token(prompt_speech, tokenizer_speech, return_tensors='pt').unsqueeze(0).to(model.device)
    input_ids_speech = input_ids_speech[:, :-2]

    while True:

        if speeches is not None and images is not None:
            inp = DEFAULT_SPEECH_TOKEN
        else:
            try:
                inp = input(f"{roles[0]}: ")
            except EOFError:
                inp = ""
            if not inp:
                print("exit...")
                break

        print(f"{roles[1]}: ", end="")

        if images is not None:
            if model.config.image_aspect_ratio == "qwen2vl":
                inp = '<|vision_start|>' + DEFAULT_IMAGE_TOKEN + '<|vision_end|>' + "\n" + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
        elif speeches is not None:
            inp = DEFAULT_SPEECH_TOKEN + "\n" + inp

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if images is not None:
            input_ids = tokenizer_image_speech_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(model.device)
        else:
            input_ids = tokenizer_speech_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(model.device)
        # prompt for base model

        streamer = TextStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        
        with torch.inference_mode():
            output_ids, speech_ids, audio = model.generate(
                input_ids,
                inputs_speech=input_ids_speech,
                images=image_tensor,
                speeches=speech_tensor,
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
                assistant_tokenizer=tokenizer_speech,
                use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            conv.messages[-1][-1] = outputs

            sf.write(args.out_path, audio, 24000)

        images = None
        speeches = None
        # prepare for second turn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="wcy1122/MGM-Omni-7B")
    parser.add_argument("--speechlm", type=str, default="wcy1122/MGM-Omni-TTS-2B")
    parser.add_argument("--ref-audio", type=str, default="assets/ref_audio/Man_EN.wav")
    parser.add_argument("--ref-audio-text", type=str, default=None)
    parser.add_argument("--out-path", type=str, default="outputs/chat.wav")
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--video-file", type=str, default=None)
    parser.add_argument("--audio-file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")

    args = parser.parse_args()
    main(args)