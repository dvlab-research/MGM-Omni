# Long-TTS-Eval

## Dataset Preparation
You can download the dataset for all tasks from [here](https://huggingface.co/datasets/wcy1122/Long-TTS-Eval).

## Metrics
We evaluate English TTS with word error rate (WER) and Chinese TTS with character error rate (CER).
Consistent with [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval), we use [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) as the ASR backend for English and [Paraformer-zh](https://huggingface.co/funasr/paraformer-zh) for Chinese.

## Structure
Please generate all speech for the model yourself, and organize all generated speech as follows before evaluation.
```
Your_Output_Dir
├── samples
│   ├── long_tts_eval_en
│   ├── long_tts_eval_zh
│   ├── hard_tts_eval_en
│   ├── hard_tts_eval_zh
```

## Evaluation
To evaluate all 4 splits
```
python -m mgm.eval.long_tts_eval.evaluation \
    --data-path wcy1122/Long-TTS-Eval \
    --output-dir $Your_Output_Dir
```

To evaluate a specific split
```
python -m mgm.eval.long_tts_eval.evaluation \
    --data-path wcy1122/Long-TTS-Eval \
    --output-dir $Your_Output_Dir
    --split long_tts_eval_en
```
