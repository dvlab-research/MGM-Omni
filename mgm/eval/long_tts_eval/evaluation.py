import os
import argparse
import glob
import json
import string
import re
from num2words import num2words
from tqdm import tqdm
import multiprocessing as mp
import torch
from datasets import load_dataset

import soundfile as sf
import scipy

from funasr import AutoModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration 
from jiwer import compute_measures
from zhon.hanzi import punctuation


def load_en_model(device):
    model_id = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    return processor, model

def load_zh_model(device):
    model = AutoModel(model="paraformer-zh")
    model.model.to(device)
    return model


def calc_wer(hypo, truth):
    measures = compute_measures(truth, hypo)
    ref_list = truth.split(" ")
    wer = measures["wer"]
    subs = measures["substitutions"] / len(ref_list)
    dele = measures["deletions"] / len(ref_list)
    inse = measures["insertions"] / len(ref_list)
    return (wer, subs, dele, inse)


def normalize_text(text, language='en'):
    try:
        text = re.sub(r'\d+', lambda m: f" {num2words(int(m.group(0)), lang=('zh' if language == 'zh' else 'en'))} ", text)
    except:
        text = text
    punctuation_all = punctuation + string.punctuation
    pattern = f'[{re.escape(punctuation_all)}]'
    text = re.sub(pattern, ' ', text.lower())
    text = ' '.join(text.split())
    if language == 'en':
        text = ' '.join(text.split())
    else:
        text = re.sub(r'(?<=[^\s])([\u4e00-\u9fff])', r' \1', text)
        text = re.sub(r'([\u4e00-\u9fff])(?=[^\s])', r'\1 ', text)
    return text


def process_chunk(args):
    audio_paths, chunk_id, language, text_dict = args
    device = f"cuda:{chunk_id % torch.cuda.device_count()}"
    if language == 'en':
        processor, model = load_en_model(device)
    else:
        model = load_zh_model(device)
    
    process_results = []
    translator = str.maketrans('', '', string.punctuation)
    for audio_path in tqdm(audio_paths):
        audio, sr = sf.read(audio_path)
        if sr != 16000:
            audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr))
            sr = 16000
        chunk_size = 28 * sr
        # load audio

        if language == "en":
            texts = []
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
            for start in range(0, len(audio), chunk_size):
                chunk = audio[start:start + chunk_size]
                input_features = processor(chunk, sampling_rate=sr, return_tensors="pt").input_features
                input_features = input_features.to(device)
                with torch.no_grad():
                    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
                text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                texts.append(text.strip())
            transcription = " ".join(t for t in texts if t)
        elif language == "zh":
            texts = []
            for start in range(0, len(audio), chunk_size):
                chunk = audio[start:start + chunk_size]
                res = model.generate(input=chunk, batch_size_s=300, device=device)
                if len(res) > 0:
                    text = res[0]["text"]
                    texts.append(text.strip())
            transcription = "".join(t for t in texts if t)
        # get transcription

        qid = os.path.basename(audio_path).replace('.wav', '')
        if text_dict.get(qid, None) is None:
            continue
        ground_truth1 = normalize_text(text_dict[qid][0], language)
        ground_truth2 = normalize_text(text_dict[qid][1], language)
        transcription = normalize_text(transcription, language)
        wer1, subs1, dele1, inse1 = calc_wer(transcription, ground_truth1)
        wer2, subs2, dele2, inse2 = calc_wer(transcription, ground_truth2)
        if wer1 < wer2:
            wer, subs, dele, inse = wer1, subs1, dele1, inse1
            word_count = len(ground_truth1.split(' '))
        else:
            wer, subs, dele, inse = wer2, subs2, dele2, inse2
            word_count = len(ground_truth2.split(' '))
        result = dict(
            audio_path=audio_path,
            ground_truth1=ground_truth1,
            ground_truth2=ground_truth2,
            transcription=transcription,
            wer=wer,
            insertions=inse,
            deletions=dele,
            substitutions=subs,
            wer1=wer1,
            wer2=wer2,
            word_count=word_count
        )
        process_results.append(result)
        # calculate wer
    
    return process_results


def load_split(
    data_path = "wcy1122/Long-TTS-Eval",
    split = "long_tts_eval_en"
):
    ds = load_dataset(data_path, split=split)
    data = ds.to_list()
    text_dict = dict()
    for item in data:
        text_dict[item['id']] = (item['text'], item['text_norm'])
    return text_dict


def main():
    parser = argparse.ArgumentParser(description="Evaliation on long TTS")
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--split", type=str, default=None)
    args = parser.parse_args()

    data_path = args.data_path
    output_dir = args.output_dir
    split = args.split

    if split is None:
        splits = ['long_tts_eval_en', 'long_tts_eval_zh', 'hard_tts_eval_en', 'hard_tts_eval_zh']
    else:
        splits = [split]
    
    results_all = dict()
    
    for split in splits:

        audio_dir = os.path.join(output_dir, 'samples', split)
        if not os.path.exists(audio_dir):
            print(f"Speech file for [{split}] split does not exist!")
            continue
        audio_paths = glob.glob(os.path.join(audio_dir, '*.wav')) 
        audio_paths.sort()
        print('Evaluate', split, 'split with', len(audio_paths), 'audio files')

        language = 'en' if 'en' in split else 'zh'
        text_dict = load_split(data_path, split)

        chunk_num = min(16, torch.cuda.device_count() * 2)
        chunks = [audio_paths[i::chunk_num] for i in range(chunk_num)]
        args = [(chunk, i, language, text_dict) for i, chunk in enumerate(chunks)]
        with mp.Pool(processes=chunk_num) as pool:
            results_from_pool = pool.map(process_chunk, args)
    
        total_wer = 0
        total_words = 0
        result_all = []
        for process_result_list in results_from_pool:
            for result in process_result_list:
                result_all.append(result)
                total_wer += result['wer'] * result['word_count']
                total_words += result['word_count']

        wer = total_wer / total_words
        print('WER:', wer)
        results_all[split] = wer
        result_all.append(dict(
            size=len(result_all),
            wer=wer,
        ))
        output_path = os.path.join(output_dir, f"wer_{split}.json")
        json.dump(result_all, open(output_path, "w"), indent=2, ensure_ascii=False)
    
    print("Final Results:")
    for split in splits:
        print(f"{split}: {results_all[split]}")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
