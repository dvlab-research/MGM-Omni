import requests
from PIL import Image, ImageDraw
from io import BytesIO
from qwen_vl_utils import process_vision_info, fetch_video
import torch

import torchaudio
from torchaudio.transforms import Resample
import opencc


def whispers_asr(whispers_model, ref_speech_file):
    audio_text = whispers_model.transcribe(ref_speech_file)['text']
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in audio_text)
    if audio_text[0] == ' ': audio_text = audio_text[1:]
    if has_chinese:
        if audio_text[-1] not in ['。', '！', '？']:
            audio_text += '。'
        audio_text = opencc.OpenCC('t2s').convert(audio_text)
    else:
        if audio_text[-1] not in ['.', '!', '?']:
            audio_text += '.'
        if audio_text[0].islower():
            audio_text = audio_text[0].upper() + audio_text[1:]
    return audio_text


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def preprocess_image_qwen2vl(image, image_resolution):
    if max(image.width, image.height) > image_resolution:
        resize_factor = image_resolution / max(image.width, image.height)
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.NEAREST)
    if image.mode != "RGB":
        image = image.convert("RGB")
    if min(image.width, image.height) < 28:
        width, height = max(image.width, 28), max(image.height, 28)
        image = image.resize((width, height), resample=Image.NEAREST)
    if image.width / image.height > 200:
        width, height = image.height * 180, image.height
        image = image.resize((width, height), resample=Image.NEAREST)
    if image.height / image.width > 200:
        width, height = image.width, image.width * 180
        image = image.resize((width, height), resample=Image.NEAREST)
    return image


def process_visual_input(image_file, video_file, image_processor, image_aspect_ratio="qwen2vl"):
    # prepare visual file
    if image_file is not None:
        images = []
        if ',' in image_file:
            images = image_file.split(',')
        else:
            images = [image_file]
        image_convert = []
        for _image in images:
            image_convert.append(load_image(_image))
        if len(image_convert) == 1:
            image_convert = image_convert[0]
        if image_aspect_ratio == "qwen2vl":
            image_tensor = preprocess_image_qwen2vl(image_convert, image_resolution=2880)
            image_tensor = image_processor(images=image_tensor, return_tensors="pt")
        else:
            raise NotImplementedError
    elif video_file is not None:
        if image_aspect_ratio == "qwen2vl":
            total_pixels = 28 * 28 * 2048 * 32
            print('Visual tokens:', int(total_pixels / 28 / 28))
            min_pixels = 28 * 28 * 4
            max_frames = 16
            messages = [{
                "role": "user",
                "content": [{
                    "type": "video",
                    "video": video_file,
                    "total_pixels": total_pixels,
                    "min_pixels": min_pixels,
                    "max_frames": max_frames
                }]
            }]
            image_inputs, video_inputs = process_vision_info(messages)
            image_tensor = image_processor(images=None, videos=video_inputs[0], return_tensors="pt")
        else:
            raise NotImplementedError
    else:
        image_tensor = None

    return image_tensor


def process_audio_input(audio_file, speech_processor):
    # prepare speech file
    if audio_file is not None:
        target_sample_rate = 16000
        wav, sample_rate = torchaudio.load(audio_file)
        resample_transform = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        wav = resample_transform(wav)
        if wav.ndim != 1: # convert to mono
            wav = wav[0]
        
        speech_tensor = []
        whipser_len = target_sample_rate * 30
        speech_num = wav.shape[0] // whipser_len + 1
        for i in range(speech_num):
            temp_wav = wav[i*whipser_len:(i+1)*whipser_len]
            _speech_tensor = speech_processor(raw_speech=temp_wav, 
                                sampling_rate=target_sample_rate, 
                                return_tensors="pt", 
                                return_attention_mask=True)["input_features"].squeeze() # (128, 3000)
            speech_tensor.append(_speech_tensor)
        speech_tensor = torch.stack(speech_tensor, dim=0).squeeze()

    else:
        speech_tensor = None

    return speech_tensor

