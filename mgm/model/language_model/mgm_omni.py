#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
#    Copyright 2025 Chengyao Wang
# ------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import os
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils import logging
from transformers.generation.utils import GenerateOutput
from transformers.generation.streamers import BaseStreamer

from mgm.model import *
from mgm.model.language_model.mgm_tts import MGMTTSConfig
from mgm.model.multimodal_generator.mgm_omni_generation import MGMOmniGeneration
from mgm.model.multimodal_generator.mgm_omni_streamer import MGMOmniStreamer
from mgm.model.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from mgm.constants import SPEECH_TOKEN_INDEX


class MGMOmniConfig(Qwen2_5_VLConfig):
    model_type = "MGMOmni"

class MGMOmniForCausalLM(MGMQwen25VLForCausalLM, MGMOmniGeneration):
    config_class = MGMOmniConfig

    def __init__(self, config):
        super(MGMOmniForCausalLM, self).__init__(config)
    
    def load_speechlm(self, speechlm_path, cosyvoice_path=None):
        kwargs = dict(
            device_map="auto",
            torch_dtype=self.dtype
        )
        self.speechlm = MGMTTSForCausalLM.from_pretrained(speechlm_path, low_cpu_mem_usage=True, **kwargs)
        self.config.speechlm = self.speechlm.config
        self.speechlm.load_cosyvoice(cosyvoice_path)
        # build speech generator

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        inputs_speech: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        speeches: Optional[torch.Tensor] = None,
        input_ids_refer: Optional[torch.Tensor] = None,
        audio_refer: Optional[torch.Tensor] = None,
        rope_deltas: Optional[torch.Tensor] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        inputs_speech, refer_speech = self.speechlm.prepare_inputs_for_speech_generation(
            inputs_speech, input_ids_refer, audio_refer
        )
        if streamer is not None and isinstance(streamer, MGMOmniStreamer):
            streamer.set_refer_speech(refer_speech)
        
        (
            inputs_speech,
            position_ids_speech,
            attention_mask_speech,
            _,
            inputs_embeds_speech,
            _,
        ) = self.speechlm.prepare_inputs_labels_for_speech_generation(
            inputs_speech,
            position_ids,
            attention_mask,
            None,
            None
        )

        output_ids, speech_ids = super().generate(
            inputs=inputs,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_speech=inputs_speech,
            images=images,
            speeches=speeches,
            inputs_embeds_speech=inputs_embeds_speech,
            position_ids_speech=position_ids_speech,
            attention_mask_speech=attention_mask_speech,
            rope_deltas=rope_deltas,
            streamer=streamer,
            **kwargs
        )
        speech_ids = speech_ids.reshape(1, -1, self.speechlm.speech_token_count + 1)
        if streamer is None or not isinstance(streamer, MGMOmniStreamer):
            audio = self.speechlm.token2audio(speech_ids, refer_speech)
        else:
            audio = None
        return output_ids, speech_ids, audio


AutoConfig.register("MGMOmni", MGMOmniConfig)
AutoModelForCausalLM.register(MGMOmniConfig, MGMOmniForCausalLM)