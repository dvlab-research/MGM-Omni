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

import os
import warnings

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, GenerationMixin
import torch
from mgm.model import *
from mgm.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from mgm.mm_utils import get_model_name_from_path
import pdb


def initialize_input_ids_for_generation(
    self,
    inputs: Optional[torch.Tensor] = None,
    bos_token_id: Optional[torch.Tensor] = None,
    model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.LongTensor:
    """Initializes input ids for generation, if necessary."""
    if inputs is not None:
        return inputs

    encoder_outputs = model_kwargs.get("encoder_outputs")
    if self.config.is_encoder_decoder and encoder_outputs is not None:
        # make dummy input_ids with value -100, as a sanity check ensuring that they won't be used for encoding
        shape = encoder_outputs.last_hidden_state.size()[:-1]
        return torch.ones(shape, dtype=torch.long, device=self.device) * -100

    # If there is some tensor in `model_kwargs`, we can infer the batch size from it. This is helpful with
    # soft-prompting or in multimodal implementations built on top of decoder-only language models.
    batch_size = 1

    if "inputs_embeds" in model_kwargs:
        return torch.ones((batch_size, 0), dtype=torch.long, device=self.device)

    for value in model_kwargs.values():
        if isinstance(value, torch.Tensor):
            batch_size = value.shape[0]
            break

    if bos_token_id is None:
        raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

    return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id

GenerationMixin._maybe_initialize_input_ids_for_generation = initialize_input_ids_for_generation


def load_pretrained_model(model_path, model_name=None, load_8bit=False, load_4bit=False,
                          device_map="auto", device="cuda", use_flash_attn=False,
                          speechlm_path=None, cosyvoice_path=None, **kwargs):
    model_name = get_model_name_from_path(model_path)
    model_name = model_name.replace('_', '-')
    if 'mgm-omni' not in model_name.lower():
        raise NotImplementedError

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    
    if 'mgm-omni-tts' in model_name.lower():
        model = MGMTTSForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        model.load_cosyvoice(cosyvoice_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        return tokenizer, model

    model = MGMOmniForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    if speechlm_path is not None:
        model.load_speechlm(speechlm_path, cosyvoice_path)
        tokenizer_speech = AutoTokenizer.from_pretrained(speechlm_path, use_fast=False)
    else:
        tokenizer_speech = None
    
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor
    else:
        image_processor = None
    
    speech_tower = model.get_speech_tower()
    if speech_tower is not None:
        if not speech_tower.is_loaded:
            speech_tower.load_model()
        speech_tower.to(device=device, dtype=torch.float16)
        speech_processor = speech_tower.speech_processor
    else:
        speech_processor = None
    
    return tokenizer, tokenizer_speech, model, image_processor, speech_processor
