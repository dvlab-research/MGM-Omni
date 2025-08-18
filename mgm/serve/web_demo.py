import io
import os
import ffmpeg
import copy
import uuid
import requests
from PIL import Image
from io import BytesIO
import gradio as gr

import torch
import numpy as np
import random

import soundfile as sf 
import librosa
import whisper
import opencc
import torchaudio
from torchaudio.transforms import Resample

import modelscope_studio.components.base as ms
import modelscope_studio.components.antd as antd
import gradio.processing_utils as processing_utils

from gradio_client import utils as client_utils
from argparse import ArgumentParser

from mgm.conversation import conv_templates
from mgm.model import *
from mgm.model.builder import load_pretrained_model
from mgm.mm_utils import tokenizer_image_speech_token, tokenizer_speech_token
from mgm.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_SPEECH_TOKEN, AUDIO_START, AUDIO_END, AUDIO_SEP
from mgm.model.multimodal_generator.mgm_omni_streamer import MGMOmniStreamer
from mgm.serve.utils import whispers_asr, preprocess_image_qwen2vl, process_visual_input, process_audio_input
from transformers import TextStreamer, TextIteratorStreamer
from threading import Thread


def _load_model_processor(args):
    tokenizer, tokenizer_speech, model, image_processor, audio_processor = \
        load_pretrained_model(
            args.model, args.load_8bit, args.load_4bit,
            speechlm_path=args.speechlm, use_flash_attn=True, device=args.device
        )
    return tokenizer, tokenizer_speech, model, image_processor, audio_processor


def _launch_demo(args, tokenizer, tokenizer_speech, model, image_processor, audio_processor):
    # Voice settings

    default_system_prompt = 'You are MGM Omni, a virtual human developed by the Von Neumann Institute, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
    pre_prompt_cn = '使用参考音频中听到的语气回答。'
    pre_prompt_en = 'Respond with the tone of the reference audio clip.'
    ref_chinese = [
        ('assets/ref_audio/Man_ZH.wav', '他疯狂寻找到能够让自己升级的办法终于有所收获，那就是炼体。'),
        ('assets/ref_audio/Woman_ZH.wav', '语音合成技术其实早已悄悄地走进了我们的生活。从智能语音助手到有声读物再到个性化语音复刻，这项技术正在改变我们获取信息，与世界互动的方式，而且他的进步速度远超我们的想象。')
    ]
    ref_english = [
        ('assets/ref_audio/Man_EN.wav', '\"Incredible!\" Dr. Chen exclaimed, unable to contain her enthusiasm. \"The quantum fluctuations we have observed in these superconducting materials exhibit completely unexpected characteristics.\"'),
        ('assets/ref_audio/Woman_EN.wav', 'The device would work during the day as well, if you took steps to either block direct sunlight or point it away from the sun.')
    ]
    previous_turn_is_tts = False
    language = args.ui_language
    whispers_model = whisper.load_model("large-v3")

    def get_text(text: str, cn_text: str):
        if language == 'en':
            return text
        if language == 'zh':
            return cn_text
        return text

    def format_history(history: list, system_prompt: str):
        messages = []
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        for item in history:
            if isinstance(item["content"], str):
                messages.append({"role": item['role'], "content": item['content']})
            elif item["role"] == "user" and (isinstance(item["content"], list) or
                                            isinstance(item["content"], tuple)):
                file_path = item["content"][0]

                mime_type = client_utils.get_mimetype(file_path)
                if mime_type.startswith("image"):
                    messages.append({
                        "role":
                        item['role'],
                        "content": [{
                            "type": "image",
                            "image": file_path
                        }]
                    })
                elif mime_type.startswith("video"):
                    messages.append({
                        "role":
                        item['role'],
                        "content": [{
                            "type": "video",
                            "video": file_path
                        }]
                    })
                elif mime_type.startswith("audio"):
                    if len(item["content"]) == 1:
                        messages.append({
                            "role":
                            item['role'],
                            "content": [{
                                "type": "audio",
                                "audio": file_path,
                            }]
                        })
                    elif len(item["content"]) == 2:
                        messages.append({
                            "role":
                            item['role'],
                            "content": [{
                                "type": "refer_speech",
                                "refer_speech": file_path,
                                "ref_speech_text": item["content"][1],
                            }]
                        })
                    else:
                        raise ValueError(f"Invalid content length: {len(item['content'])}")
        return messages
    
    def process_messages(messages, conv):
        inp = ''
        image_files = []
        audio_files = []
        ref_speech_file = None
        ref_speech_text = None

        user_inp = ''
        last_text_inp = ''
        for message in messages:
            if message['role'] == 'system':
                conv.system = '<|im_start|>system\n' + message['content'][0]['text']
            elif message['role'] == 'user':
                if isinstance(message['content'], str):
                    user_inp += message['content']
                    last_text_inp = message['content']
                    conv.append_message(conv.roles[0], user_inp)
                    user_inp = ''
                else:
                    for item in message['content']:
                        if item['type'] == 'image':
                            image_files.append((item['image'], None))
                            user_inp += '<|vision_start|>' + DEFAULT_IMAGE_TOKEN + '<|vision_end|>' + "\n"
                        if item['type'] == 'video':
                            image_files.append((None, item['video']))
                            user_inp += '<|vision_start|>' + DEFAULT_IMAGE_TOKEN + '<|vision_end|>' + "\n"
                        elif item['type'] == 'audio':
                            audio_files.append(item['audio'])
                            user_inp += DEFAULT_SPEECH_TOKEN
                        elif item['type'] == 'refer_speech':
                            ref_speech_file = item['refer_speech']
                            ref_speech_text = item['ref_speech_text']
            elif message['role'] == 'assistant':
                if user_inp != '':
                    conv.append_message(conv.roles[0], user_inp)
                    user_inp = ''
                conv.append_message(conv.roles[1], message['content'])
        if user_inp != '':
            conv.append_message(conv.roles[0], user_inp)
            user_inp = ''
        conv.append_message(conv.roles[1], None)

        if ref_speech_file is None:
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in last_text_inp)
            if has_chinese:
                ref_item = random.choice(ref_chinese)
            else:
                ref_item = random.choice(ref_english)
            ref_speech_file, ref_speech_text = ref_item

        return conv, image_files, audio_files, ref_speech_file, ref_speech_text


    def predict(messages):
        conv = conv_templates['qwen2vl'].copy()
        conv_speech = conv_templates['qwen2vl'].copy()
        conv, image_files, audio_files, ref_speech_file, ref_speech_text = process_messages(messages, conv)

        # prepare image & speech file
        image_aspect_ratio = getattr(model.config, 'image_aspect_ratio', 'qwen2vl')
        image_tensor = [process_visual_input(image_file[0], image_file[1], image_processor, image_aspect_ratio) for image_file in image_files]
        speech_tensor = [process_audio_input(audio_file, audio_processor) for audio_file in audio_files]

        if len(image_tensor) > 0:
            if isinstance(image_tensor[0], dict):
                for image in image_tensor:
                    for key in image.keys():
                        image[key] = image[key].to(dtype=model.dtype, device=model.device, non_blocking=True)
            else:
                image_tensor = [image.to(dtype=model.dtype, device=model.device, non_blocking=True) for image in image_tensor]
        else:
            image_tensor = None

        if len(speech_tensor) > 0:
            speech_tensor = [speech.to(dtype=model.dtype, device=model.device, non_blocking=True) for speech in speech_tensor]
        else:
            speech_tensor = None

        # process refer speech
        audio_refer, _ = librosa.load(ref_speech_file, sr=16000)
        audio_refer = torch.tensor(audio_refer).unsqueeze(0).to(model.device)
        text_refer = ref_speech_text
        input_ids_refer = tokenizer_speech(text_refer)['input_ids']
        input_ids_refer = torch.tensor(input_ids_refer).unsqueeze(0).to(model.device)

        prompt = conv.get_prompt()
        if image_tensor is not None:
            input_ids = tokenizer_image_speech_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(model.device)
        else:
            input_ids = tokenizer_speech_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(model.device)
        print("************MLM prompt: ", prompt)
        # prompt for base model

        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text_refer)
        pre_prompt_speech = (pre_prompt_cn if has_chinese else pre_prompt_en)
        inp_speech = pre_prompt_speech + AUDIO_START + DEFAULT_SPEECH_TOKEN + AUDIO_END + "\n" # + inp_speech
        conv_speech.append_message(conv_speech.roles[0], inp_speech)
        conv_speech.append_message(conv_speech.roles[1], AUDIO_START)
        prompt_speech = conv_speech.get_prompt().replace('<|im_end|>\n', '')
        input_ids_speech = tokenizer_speech_token(prompt_speech, tokenizer_speech, return_tensors='pt').unsqueeze(0).to(model.device)
        print("************SLM prompt: ", prompt_speech)
        # prompt for speech generator

        streamer = MGMOmniStreamer(
            tokenizer,
            cosyvoice=model.speechlm.cosyvoice.model,
            max_audio_token=model.config.speechlm.tokenizer_speech_size,
            skip_prompt=True, skip_special_tokens=True, timeout=15
        )
        thread = Thread(
            target=model.generate,
            kwargs=dict(
                inputs=input_ids,
                inputs_speech=input_ids_speech,
                images=image_tensor,
                speeches=speech_tensor,
                input_ids_refer=input_ids_refer,
                audio_refer=audio_refer,
                streamer=streamer,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=4096,
                bos_token_id=tokenizer.pad_token_id,
                eos_token_id=[tokenizer.eos_token_id],
                pad_token_id=tokenizer.pad_token_id,
                tokenizer=tokenizer,
                assistant_tokenizer=tokenizer_speech,
                use_cache=True
            ),
        )
        thread.start()

        response = ''
        audio = []
        stop_str = '<|im_end|>'
        for item in streamer:
            item_type, content = item
            if item_type == 'text':
                response += content
                if response.endswith(stop_str):
                    response = response[: -len(stop_str)]
                yield {"type": "text", "data": response}
            else:
                yield {"type": "audio", "data": content}

        thread.join()


    def chat_predict(text, refer_speech, audio, talk_inp, image, video, history, system_prompt, autoplay):
        # Clean TTS history
        global previous_turn_is_tts
        try:
            if previous_turn_is_tts:
                history = []
                previous_turn_is_tts = False
        except:
            previous_turn_is_tts = False

        # Process text input
        if text:
            history.append({"role": "user", "content": text})
        else:
            text = ''
        
        # Process refer_speech input
        if refer_speech:
            refer_speech_text = whispers_asr(whispers_model, refer_speech)
            history.append({"role": "user", "content": (refer_speech, refer_speech_text)})

        # Process talk input
        if talk_inp:
            history.append({"role": "user", "content": (talk_inp, )})

        # assign refer_speech
        has_refer_speech = False
        for item in history:
            if isinstance(item['content'], tuple):
                has_refer_speech |= (len(item['content']) == 2)
        if has_refer_speech == False:
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
            if has_chinese:
                ref_item = random.choice(ref_chinese)
            else:
                ref_item = random.choice(ref_english)
            refer_speech, refer_speech_text = ref_item
            history.append({"role": "user", "content": (refer_speech, refer_speech_text)})

        formatted_history = format_history(history=history,
                                           system_prompt=system_prompt)

        yield None, None, None, None, None, None, None, history

        history.append({"role": "assistant", "content": ""})
        sample_rate = 24000
        audio = []
        for chunk in predict(formatted_history):
            if chunk["type"] == "text":
                history[-1]["content"] = chunk["data"]
                yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(
                ), None, history
            elif chunk["type"] == "audio":
                audio.append(chunk["data"])
                audio_output = (sample_rate, chunk["data"]) if autoplay else None
                yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), audio_output, history
        audio = np.concatenate(audio)
        history.append({"role": "assistant", "content": gr.Audio((sample_rate, audio))})
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), None, history
    

    def tts_run(messages):
        sample_rate = 24000
        target_text = messages[1]['content']
        if len(messages) < 3:
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in target_text)
            if has_chinese:
                ref_item = random.choice(ref_chinese)
            else:
                ref_item = random.choice(ref_english)
            ref_speech_file, ref_speech_text = ref_item
        else:
            ref_speech_file = messages[2]['content'][0]['refer_speech']
            ref_speech_text = messages[2]['content'][0]['ref_speech_text']

        # process refer audio
        audio_refer, _ = librosa.load(ref_speech_file, sr=16000)
        audio_refer = torch.tensor(audio_refer).unsqueeze(0).to(model.device)
        text_refer = ref_speech_text
        input_ids_refer = tokenizer_speech(text_refer)['input_ids']
        input_ids_refer = torch.tensor(input_ids_refer).unsqueeze(0).to(model.device)

        conv = conv_templates['qwen2vl'].copy()
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text_refer)
        pre_prompt = (pre_prompt_cn if has_chinese else pre_prompt_en)
        inp = pre_prompt + AUDIO_START + DEFAULT_SPEECH_TOKEN + AUDIO_END + "\n"
        oup = AUDIO_START + target_text
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], oup)
        prompt = conv.get_prompt()
        input_ids = tokenizer_speech_token(prompt, tokenizer_speech, return_tensors='pt').unsqueeze(0).to(model.device)
        print("************SLM prompt: ", prompt)
        # prompt for SpeechLM

        streamer = MGMOmniStreamer(
            tokenizer_speech,
            cosyvoice=model.speechlm.cosyvoice.model,
            max_audio_token=model.config.speechlm.tokenizer_speech_size,
            skip_prompt=True, skip_special_tokens=True, timeout=15
        )
        thread = Thread(
            target=model.speechlm.generate,
            kwargs=dict(
                inputs=input_ids,
                input_ids_refer=input_ids_refer,
                audio_refer=audio_refer,
                streamer=streamer,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=4096,
                bos_token_id=tokenizer_speech.pad_token_id,
                eos_token_id=[tokenizer_speech.eos_token_id],
                pad_token_id=tokenizer_speech.pad_token_id,
                tokenizer=tokenizer_speech,
                use_cache=True
            ),
        )
        thread.start()

        response = ''
        audio = []
        stop_str = '<|im_end|>'
        for item in streamer:
            item_type, content = item
            if item_type == 'text':
                response += content
                if response.endswith(stop_str):
                    response = response[: -len(stop_str)]
                yield {"type": "text", "data": response}
            else:
                yield {"type": "audio", "data": content}

        thread.join()


    def tts_predict(text, refer_speech, system_prompt, history, autoplay):
        # Process refer_speech input
        if refer_speech:
            refer_speech_text = whispers_asr(whispers_model, refer_speech)
        else:
            refer_speech = None
            refer_speech_text = None
            for item in history:
                if item["role"] == "user" and len(item["content"]) == 2:
                    refer_speech = item["content"][0]
                    refer_speech_text = item["content"][1]
        history = []
        global previous_turn_is_tts
        previous_turn_is_tts = True

        # Process text input
        if text:
            history.append({"role": "user", "content": text})
        else:
            history.append({"role": "assistant", "content": "Don't forget to input text for text to speech synthesis."})
            yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), None, history
            return

        if refer_speech is not None:
            history.append({"role": "user", "content": (refer_speech, refer_speech_text)})

        formatted_history = format_history(history=history,
                                           system_prompt=system_prompt)

        yield None, None, None, None, None, None, None, history

        history.append({"role": "assistant", "content": ""})
        sample_rate = 24000
        audio = []
        for chunk in tts_run(formatted_history):
            if chunk["type"] == "text":
                history[-1]["content"] = chunk["data"]
                yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(
                ), None, history
            elif chunk["type"] == "audio":
                audio.append(chunk["data"])
                audio_output = (sample_rate, chunk["data"]) if autoplay else None
                yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), audio_output, history
        audio = np.concatenate(audio)
        history.append({"role": "assistant", "content": gr.Audio((sample_rate, audio))})
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), None, history

    with gr.Blocks(title="MGM-Omni", theme=gr.themes.Soft()) as demo:  # Using a clean theme similar to ChatGPT
        with gr.Sidebar(open=False):
            system_prompt_textbox = gr.Textbox(label="System Prompt",
                                            value=default_system_prompt)
        gr.HTML(
                    """
                    <style>
                    .grid-wrap.fixed-height {
                        min-height: 0 !important;
                        max-height: 55vh;
                    }
                    .container-display { 
                        display: none; 
                    }
                    .gallery_reference_example .caption-label {
                        font-size: 12px !important;
                    }
                    .small-radio {font-size: 14px !important;}
                    .right-align { display: flex; justify-content: flex-end; }
                    </style>
                    """
                )
        gr.Markdown(
            """
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="margin-bottom: 8px;">MGM-Omni: An Open-source Omni Model</h1>
                <div style="display: inline-flex; white-space: nowrap; gap: 8px; align-items: center; justify-content: center; overflow-x: auto;">
                    <a href="https://github.com/dvlab-research/MGM-Omni" target="_blank" rel="noopener noreferrer">
                        <img alt="Github" src="https://img.shields.io/badge/Github-000000?style=for-the-badge&logo=github&logoColor=white">
                    </a>
                    <a href="https://mgm-omni.notion.site/MGM-Omni-An-Open-source-Omni-Chatbot-2395728e0b0180149ac9f24683fc9907?source=copy_link" target="_blank" rel="noopener noreferrer">
                        <img alt="Blog" src="https://img.shields.io/badge/Blog-000000.svg?style=for-the-badge&logo=notion&logoColor=white">
                    </a>
                    <a href="https://huggingface.co/collections/wcy1122/mgm-omni-6896075e97317a88825032e1" target="_blank" rel="noopener noreferrer">
                        <img alt="Models" src="https://img.shields.io/badge/Models-000000?style=for-the-badge&logo=huggingface&logoColor=white">
                    </a>
                </div>
                <p style="margin-top: 0; margin-bottom: 12px;">Chat with the model. Upload audio, images, or videos as needed.</p>
            </div>
            """
        )

        # Hidden components for handling uploads and outputs
        audio_input = gr.Audio(visible=True, type="filepath", elem_classes="container-display" )
        image_input = gr.Image(visible=True, type="filepath", elem_classes="container-display" )
        video_input = gr.Video(visible=True, elem_classes="container-display" )
        audio_output = gr.Audio(
            label="Generated Audio",
            autoplay=True,
            streaming=True,
            visible=True,
            elem_classes="container-display" 
        )
        placeholder = placeholder = """
**Welcome to MGM-Omni!** 🎉  

Start chatting or generate voice responses with these options:  

- 🎙️ **Reference Voice**: Choose, upload or record an audio clip for voice clone.
- 📤 **Upload**: Upload video, image, or audio files.   
- ✍️ **Input Mode**:  
  - **Text**: Type your message to chat.
  - **Talk**: Record or upload audio to chat.  
- 🚀 **Generate Mode**:  
  - **Chat**: Engage in a conversation with MGM-Omni.  
  - **TTS**: Text to speech generation with reference voice.

**Get started by typing or uploading below!** 😊
"""
        with gr.Row(equal_height=True):
            with gr.Column(scale=7, min_width="70%"):
                # Chatbot as the main component
                chatbot = gr.Chatbot(
                    type="messages",
                    height=600,
                    placeholder=placeholder,
                    show_label=False
                )
            with gr.Column(scale=3):
                refer_speech = gr.Audio(sources=["upload", "microphone"],
                                    type="filepath",
                                    label="Upload Reference Voice",
                                    elem_classes="media-upload",
                                    value=None,
                                    scale=0
                                    )
                # Restore reference speech gallery in sidebar for better layout
                gr.Markdown("### Voice Clone Examples")
                refer_items = [
                    ("assets/ref_img/Man_ZH.jpg", "assets/ref_audio/Man_ZH.wav", "Man-ZH"),
                    ("assets/ref_img/Man_EN.jpg", "assets/ref_audio/Man_EN.wav", "Man-EN"),
                    ("assets/ref_img/Woman_ZH.jpg", "assets/ref_audio/Woman_ZH.wav", "Woman-ZH"),
                    ("assets/ref_img/Woman_EN.jpg", "assets/ref_audio/Woman_EN.wav", "Woman-EN"),
                    ("assets/ref_img/Old_Woman_ZH.jpg", "assets/ref_audio/Old_Woman_ZH.wav", "Old-Woman-ZH"),
                    ("assets/ref_img/Musk.jpg", "assets/ref_audio/Musk.wav", "Elon Musk"),
                    ("assets/ref_img/Trump.jpg", "assets/ref_audio/Trump.wav", "Donald Trump"),
                    ("assets/ref_img/Jensen.jpg", "assets/ref_audio/Jensen.wav", "Jensen Huang"),
                    ("assets/ref_img/Lebron.jpg", "assets/ref_audio/Lebron.wav", "LeBron James"),
                    ("assets/ref_img/jay.jpg", "assets/ref_audio/Jay.wav", "Jay Chou(周杰伦)"),
                    ("assets/ref_img/GEM.jpg", "assets/ref_audio/GEM.wav", "G.E.M.(邓紫棋)"),
                    ("assets/ref_img/Zhiling.jpg", "assets/ref_audio/Zhiling.wav", "Lin Chi-Ling(林志玲)"),
                    ("assets/ref_img/mabaoguo.jpg", "assets/ref_audio/mabaoguo.wav", "Ma Baoguo(马保国)"),
                    ("assets/ref_img/Taiyi.jpg", "assets/ref_audio/Taiyi.wav", "Taiyi(太乙真人)"),
                    ("assets/ref_img/StarRail_Firefly.jpg", "assets/ref_audio/StarRail_Firefly.wav", "崩铁-流萤"),
                    ("assets/ref_img/genshin_Kokomi.jpg", "assets/ref_audio/genshin_Kokomi.wav", "原神-珊瑚宫心海"),
                    ("assets/ref_img/genshin_Raiden.jpg", "assets/ref_audio/genshin_Raiden.wav", "原神-雷电将军"),
                    ("assets/ref_img/genshin_ZhongLi.jpg", "assets/ref_audio/genshin_ZhongLi.wav", "原神-钟离"),
                    ("assets/ref_img/Wave_Jinhsi.jpg", "assets/ref_audio/Wave_Jinhsi.wav", "鸣潮-今汐"),
                    ("assets/ref_img/Wave_Carlotta.jpg", "assets/ref_audio/Wave_Carlotta.wav", "鸣潮-珂莱塔"),
                ]
                gallery_items = [(img, label) for img, _, label in refer_items]
                
                
                gallery = gr.Gallery(
                    value=gallery_items,
                    label=None,
                    show_label=False,
                    allow_preview=False,
                    columns=3,  # Adjusted for sidebar width
                    # rows=5,
                    height="auto",
                    object_fit="cover",
                    elem_classes="gallery_reference_example" 
                )
                

                def on_image_click(evt: gr.SelectData):
                    index = evt.index
                    if index is not None and 0 <= index < len(refer_items):
                        audio_path = refer_items[index][1]
                        return gr.update(value=audio_path)
                    return gr.update()

                gallery.select(
                    fn=on_image_click,
                    inputs=None,
                    outputs=refer_speech
                )
                clear_btn = gr.Button("Clear")
                autoplay_checkbox = gr.Checkbox(
                    label="Autoplay",
                    value=True
                )

        text_input = gr.Textbox(
            show_label=False,
            placeholder="Type your message here...",
            container=False
        )
        talk_input = gr.Audio(sources=["microphone", ], visible=False, type="filepath", label="Audio Message" )

        with gr.Row(equal_height=True):
            upload_btn = gr.UploadButton(
                label="Upload",
                file_types=["image", "video", "audio"],
                file_count="single",
                size="md",
                scale=1,
                visible=True
            )
            chat_mode_selector = gr.Radio(
                choices=["Text", "Talk"],
                value="Text",
                show_label=False,
                interactive=True,
                elem_classes="small-radio",
                scale=2,
            )
            submit_mode_selector = gr.Radio(
                choices=["Chat", "TTS"],
                value="Chat",
                show_label=False,
                interactive=True,
                elem_classes="small-radio",
                scale=2,
            )
            gr.Column(scale=3, min_width=0)  
            submit_btn = gr.Button(
                "Send",
                variant="primary",
                min_width=0,
                size="md",
                scale=1,
                visible=True
            )
            tts_submit_btn = gr.Button(
                "TTS Submit",
                variant="primary",
                min_width=0,
                size="md",
                scale=1,
                visible=False
            )

        # State to hold history
        state = gr.State([])

        def handle_upload(file, history):
            if file:
                mime = client_utils.get_mimetype(file.name)
                if mime.startswith("image"):
                    history.append({"role": "user", "content": (file, )})
                    return file, None, None, history
                elif mime.startswith("video"):
                    history.append({"role": "user", "content": (file, )})
                    return None, file, None, history
                elif mime.startswith("audio"):
                    history.append({"role": "user", "content": (file, )})
                    return None, None, file, history
            return None, None, None, history

        upload_btn.upload(
            handle_upload,
            inputs=[upload_btn, chatbot],
            outputs=[image_input, video_input, audio_input, chatbot]
        )

        def clear_chat_history():
            return [], gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value="Text"), gr.update(value="Chat")

        submit_event = gr.on(
                triggers=[submit_btn.click, text_input.submit],
                fn=chat_predict,
                inputs=[
                    text_input, refer_speech, audio_input, talk_input, image_input, video_input, chatbot,
                    system_prompt_textbox, autoplay_checkbox
                ],
                outputs=[
                    text_input, refer_speech, audio_input, talk_input, image_input, video_input, audio_output, chatbot
                ])
        tts_submit_event = gr.on(
                triggers=[tts_submit_btn.click],
                fn=tts_predict,
                inputs=[
                    text_input, refer_speech, system_prompt_textbox, chatbot, autoplay_checkbox
                ],
                outputs=[
                    text_input, refer_speech, audio_input, talk_input, image_input, video_input, audio_output, chatbot
                ])

        def chat_switch_mode(mode):
            if mode == "Text":
                return gr.update(visible=True), gr.update(visible=False)
            else: 
                return gr.update(visible=False), gr.update(visible=True)

        def submit_switch_mode(mode):
            if mode == "Chat":
                return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
            else: 
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

        chat_mode_selector.change(
            fn=chat_switch_mode,
            inputs=[chat_mode_selector],
            outputs=[text_input, talk_input]
        )
        
        submit_mode_selector.change(
            fn=submit_switch_mode,
            inputs=[submit_mode_selector],
            outputs=[upload_btn, submit_btn, tts_submit_btn]
        )

        clear_btn.click(fn=clear_chat_history,
                        inputs=None,
                        outputs=[
                            chatbot, text_input, refer_speech, audio_input, talk_input, image_input,
                            video_input, audio_output, chat_mode_selector, submit_mode_selector  
                        ])



        # Custom CSS for ChatGPT-like styling
        demo.css = """
            .gradio-container {
                max-width: 90vw !important;
                margin: auto;
                padding: 20px;
            }
            .chatbot .message {
                border-radius: 10px;
                padding: 10px;
            }
            .chatbot .user {
                background-color: #f0f0f0;
            }
            .chatbot .assistant {
                background-color: #e6e6e6;
            }
            footer {display:none !important}
        """

    demo.queue(default_concurrency_limit=100, max_size=100).launch(max_threads=100,
                                                                share=True,
                                                                show_error=True,
                                                                ssl_certfile=None,
                                                                ssl_keyfile=None,
                                                                ssl_verify=False,
                                                                inbrowser=args.inbrowser,
                                                                server_port=args.server_port,
                                                                server_name=args.server_name,)


def _get_args():
    parser = ArgumentParser()
    parser.add_argument('--cpu-only', action='store_true', help='Run demo with CPU only')
    parser.add_argument('--flash-attn2',
                        action='store_true',
                        default=False,
                        help='Enable flash_attention_2 when loading the model.')
    parser.add_argument('--share',
                        action='store_true',
                        default=False,
                        help='Create a publicly shareable link for the interface.')
    parser.add_argument('--inbrowser',
                        action='store_true',
                        default=False,
                        help='Automatically launch the interface in a new tab on the default browser.')
    parser.add_argument('--server-port', type=int, default=8888, help='Demo server port.')
    parser.add_argument('--server-name', type=str, default='0.0.0.0', help='Demo server name.')
    parser.add_argument('--ui-language', type=str, choices=['en', 'zh'], default='en', help='Display language for the UI.')
    parser.add_argument("--model", type=str, default="wcy1122/MGM-Omni-7B")
    parser.add_argument("--speechlm", type=str, default="wcy1122/MGM-Omni-TTS-2B")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _get_args()
    tokenizer, tokenizer_speech, model, image_processor, audio_processor = _load_model_processor(args)
    _launch_demo(args, tokenizer, tokenizer_speech, model, image_processor, audio_processor)