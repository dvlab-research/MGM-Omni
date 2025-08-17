import torch
import torch.nn as nn
import os

from transformers import Qwen2AudioEncoder, WhisperFeatureExtractor, Qwen2AudioEncoderConfig
from transformers import WhisperModel, WhisperFeatureExtractor, WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder

class DualAudioTower(nn.Module):
    def __init__(self, speech_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.speech_tower_name = speech_tower
        self.speech_tower_name_aux = getattr(args, 'speech_tower_aux')
        self.is_optimize = getattr(args, 'optimize_speech_tower', False)
        
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_speech_tower', False):
            self.load_model()
        else:
            self.cfg_only = Qwen2AudioEncoderConfig.from_pretrained(self.speech_tower_name)

    def load_model(self):
        self.speech_processor = WhisperFeatureExtractor.from_pretrained(self.speech_tower_name)
        self.speech_tower = Qwen2AudioEncoder.from_pretrained(self.speech_tower_name)
        self.speech_tower.requires_grad_(False)
        self.speech_tower_aux = WhisperModel.from_pretrained(self.speech_tower_name_aux).encoder
        print("Load auxiliary speech tower model: {}".format(self.speech_tower_name_aux))
        self.speech_tower_aux.requires_grad_(False)
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, speeches):
        speech_features_main = self.speech_tower(speeches.to(device=self.speech_tower.device), return_dict=True).last_hidden_state
        speech_features_aux = self.speech_tower_aux(speeches.to(device=self.speech_tower_aux.device), return_dict=True).last_hidden_state
        return (speech_features_main, speech_features_aux)


    @property
    def dtype(self):
        return self.speech_tower.dtype

    @property
    def device(self):
        return self.speech_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.speech_tower.config
        else:
            return self.cfg_only