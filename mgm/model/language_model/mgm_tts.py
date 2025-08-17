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
import copy
import os
import torchaudio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3ForCausalLM, Qwen3Config, Qwen3Model
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.utils import logging
from transformers.generation.utils import GenerateOutput
from transformers.cache_utils import Cache, DynamicCache
from huggingface_hub import snapshot_download

from mgm.constants import IGNORE_INDEX, SPEECH_TOKEN_INDEX
from mgm.model.mgm_arch_tts import MGMTTSMetaModel, MGMTTSMetaForCausalLM
from mgm.model.multimodal_generator.tts_adapter import build_tts_adapter
from mgm.model.multimodal_generator.mgm_tts_generation import MGMTTSGeneration
from mgm.model.multimodal_generator.mgm_omni_streamer import MGMOmniStreamer

import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('third_party')
from cosyvoice.cli.cosyvoice import CosyVoice2
import uuid


@dataclass
class MGMCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_text: Optional[torch.FloatTensor] = None
    loss_speech: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    speech_logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    speech_past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class MGMTTSConfig(Qwen3Config):
    model_type = "MGMTTS"


class MGMTTSModel(MGMTTSMetaModel, Qwen3Model):
    config_class = MGMTTSConfig
    
    def __init__(self, config: Qwen3Config):
        super(MGMTTSModel, self).__init__(config)


class MGMTTSForCausalLM(Qwen3ForCausalLM, MGMTTSMetaForCausalLM, MGMTTSGeneration):
    config_class = MGMTTSConfig

    def __init__(self, config):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = MGMTTSModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # set speech tokenizer parameters
        self.speech_token_count = getattr(config ,'speech_token_count', 4)
        self.speech_token_pre_pad = getattr(config ,'speech_token_pre_pad', 1)
        self.tokenizer_speech_size = getattr(config ,'tokenizer_speech_size', 6563)
        self.tokenizer_text_size = self.vocab_size - self.tokenizer_speech_size * self.speech_token_count
        self.audio_start = self.tokenizer_text_size - 3
        self.audio_blank = 6486

        # build tts adapter
        if hasattr(config, "tts_adapter_type"):
            self.tts_adapter = build_tts_adapter(config)

        # Initialize weights and apply final processing
        self.post_init()
    
    # load cosyvoice2
    def load_cosyvoice(self, cosyvoice_path=None):
        if cosyvoice_path is None:
            model_dir = self.name_or_path
            if not os.path.exists(self.name_or_path):
                model_dir = snapshot_download(
                    repo_id=self.name_or_path,
                    allow_patterns="cosyvoice2/*"
                )
            cosyvoice_path = os.path.join(model_dir, 'cosyvoice2')
        self.cosyvoice = CosyVoice2(cosyvoice_path, fp16=False)

    def get_model(self):
        return self.model

    def initialize_tts_training(self, model_args, tokenizer):
        self.config.tts_adapter_type = getattr(model_args, 'tts_adapter_type', 'qwen3')
        if getattr(self, "tts_adapter", None) is None:
            self.tts_adapter = build_tts_adapter(self.config)

        # expand tokenizer
        self.tokenizer_text_size = len(tokenizer)
        self.speech_token_count = getattr(model_args, 'speech_token_count', 4)
        self.speech_token_pre_pad = getattr(model_args, 'speech_token_pre_pad', 4)
        self.tokenizer_speech_size = getattr(model_args ,'tokenizer_speech_size', 6563)
        if self.tokenizer_text_size > 160000: # speech token is added
            self.tokenizer_text_size -= self.tokenizer_speech_size * self.speech_token_count
        else:
            num_new_tokens = 0

            # add mm_use_speech_start_end tokens
            if self.config.mm_use_speech_start_end:
                tokenizer.add_tokens('<|audio_start|>')
                tokenizer.add_tokens('<|audio_end|>')
                tokenizer.add_tokens('<|audio_sep|>')
                num_new_tokens = 3
                self.tokenizer_text_size += 3
                self.audio_start = self.tokenizer_text_size - 3

            # add speech tokens
            speech_tokens = []
            for i in range(self.speech_token_count):
                for j in range(self.tokenizer_speech_size - 2):
                    speech_token = f'speech_layer_{i}_{j}'
                    speech_tokens.append(speech_token)
                speech_tokens.append(f'speech_layer_{i}_eos') # eos token
                speech_tokens.append(f'speech_layer_{i}_pad') # pad token
            num_new_tokens += tokenizer.add_tokens(speech_tokens)
            self.resize_token_embeddings(len(tokenizer))

            # set requires_grad
            for p in self.get_input_embeddings().parameters():
                p.requires_grad = True
            for p in self.get_output_embeddings().parameters():
                p.requires_grad = True
        
        self.config.tokenizer_text_size = self.tokenizer_text_size
        self.config.tokenizer_speech_size = self.tokenizer_speech_size
    
    def get_tts_adapter(self):
        tts_adapter = getattr(self, 'tts_adapter', None)
        return tts_adapter

    def prepare_cosyvoice_tokens(self, audio):
        audio_24 = torchaudio.transforms.Resample(16000, 24000)(audio.cpu()).to(self.device)
        refer_speech, _ = self.cosyvoice._extract_speech_feat(audio_24)
        embedding = self.cosyvoice._extract_spk_embedding(audio)
        block_size = 16000 * 30
        num_blocks = (audio.shape[1] + block_size - 1) // block_size
        audios = [audio[:, i*block_size:(i+1)*block_size] for i in range(num_blocks)]
        speech_tokens = [self.cosyvoice._extract_speech_token(audio)[0] for audio in audios]
        speech_tokens = torch.cat(speech_tokens, dim=1)
        token_len = min(refer_speech.shape[1] // 2, speech_tokens.shape[1])
        return speech_tokens[:, :token_len], refer_speech[:, :2*token_len], embedding
    
    def prepare_inputs_for_speech_generation(
        self,
        inputs_speech: Optional[torch.Tensor] = None,
        input_ids_refer: Optional[torch.Tensor] = None,
        audio_refer: Optional[torch.Tensor] = None,
    ):
        tokenizer_speech_size = getattr(self.config, 'tokenizer_speech_size', 6563)
        speech_token_count = getattr(self.config, 'speech_token_count', 4)
        speech_token_pre_pad = getattr(self.config, 'speech_token_pre_pad', 4)
        speech_pad, speech_eos = tokenizer_speech_size - 1, tokenizer_speech_size - 2
        device = self.device

        ref_tokens, ref_feat, ref_embedding = self.prepare_cosyvoice_tokens(audio_refer)
        self.speech_pad = speech_pad
        self.speech_eos = speech_eos

        keep_len = (ref_tokens.shape[1] // speech_token_count) * speech_token_count
        speech_ids = torch.cat([
            torch.full((speech_token_pre_pad, speech_token_count), speech_pad, device=device),
            ref_tokens[:, :keep_len].view(-1, speech_token_count).to(device),
            torch.full((1, speech_token_count), speech_eos, device=device)
        ]).unsqueeze(0)

        pad_length = speech_ids.shape[1] - input_ids_refer.shape[1]
        input_ids_refer = F.pad(input_ids_refer, (0, pad_length), value=151643)
        input_ids_refer = input_ids_refer.unsqueeze(2).to(device)
        speech_ids = torch.cat((input_ids_refer, speech_ids), dim=-1)

        inputs_speech = inputs_speech.unsqueeze(-1).expand(-1, -1, speech_token_count + 1).clone().to(device)
        inputs_speech[..., 1:] = speech_pad
        speech_pos = (inputs_speech[..., 0] == SPEECH_TOKEN_INDEX).nonzero()[0][1]
        inputs_speech = torch.cat((
            inputs_speech[:, :speech_pos], speech_ids,
            inputs_speech[:, speech_pos + 1:]), dim=1)

        return inputs_speech, (ref_tokens, ref_feat, ref_embedding)
    
    def token2audio(self, speech_tokens, refer_speech, hop_len = 4096):
        ref_tokens, ref_feat, ref_embedding = refer_speech
        text_tokens = speech_tokens[..., 0]
        speech_tokens = speech_tokens[..., 1:]

        tokenizer_speech_size = getattr(self.config, 'tokenizer_speech_size', 6563)
        speech_token_count = getattr(self.config, 'speech_token_count', 4)
        speech_token_pre_pad = getattr(self.config, 'speech_token_pre_pad', 1)
        speech_pad, speech_eos = tokenizer_speech_size - 1, tokenizer_speech_size - 2
        audio_sep = self.tokenizer_text_size - 1

        if audio_sep in text_tokens:
            sep_idxs = (text_tokens == audio_sep).nonzero()[:, 1]
            start_idx = torch.cat([torch.tensor([-1], device=speech_tokens.device), sep_idxs]) + 1
            end_idx = torch.cat([sep_idxs, torch.tensor([speech_tokens.size(1) - 2], device=speech_tokens.device)])
            splits = [speech_tokens[:, start:end, :] for start, end in zip(start_idx, end_idx)]
        else:
            splits = [speech_tokens[:, :-2]]
        
        speech_ids = [[] for i in range(speech_token_count)]
        # modify out of bound speeck tokens to silance speech
        for i in range(speech_token_count):
            for k, speech_tokens in enumerate(splits):
                speech_id = speech_tokens[:, speech_token_pre_pad:-1, i]
                speech_id[speech_id >= speech_eos] = self.audio_blank
                speech_ids[i].append(speech_id)
        speech_ids = [torch.cat(speech_id, dim=1) for speech_id in speech_ids]
        speech_tokens = torch.cat(speech_ids, dim=0).T.reshape(1, -1)

        audio = []
        hop_len = 6000
        cur_uuid = str(uuid.uuid1())
        speeck_tokens_count = speech_tokens.shape[1]
        for i in range(0, speeck_tokens_count, hop_len):
            self.cosyvoice.model.hift_cache_dict[cur_uuid] = None
            cur_audio = self.cosyvoice.model.token2wav(
                token=speech_tokens[:, i:i+hop_len],
                prompt_token=ref_tokens,
                prompt_feat=ref_feat,
                embedding=ref_embedding,
                uuid=cur_uuid,
                token_offset=0,
                finalize=True
            )
            audio.append(cur_audio)
        audio = torch.cat(audio, dim=1).detach().squeeze().cpu().numpy()
        return audio

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        speech_past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_speech_generation(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels
            )
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        batch_size = hidden_states.shape[0]

        if self.training:
            logits = self.lm_head(hidden_states)
            logits = logits[..., :self.tokenizer_text_size]
        else:
            logits = hidden_states
        logits = logits.float()

        loss, loss_text, loss_speech = None, None, None

        if not self.training and speech_past_key_values is None:
            speech_past_key_values = DynamicCache()
        
        speech_outputs = self.tts_adapter(
            hidden_states=hidden_states,
            attention_mask=attention_mask, 
            position_ids=position_ids,
            use_cache=use_cache,
            past_key_values=speech_past_key_values,
            cache_position=cache_position
        )

        speech_logits = self.lm_head(speech_outputs.hidden_states)
        speech_logits = speech_logits.float()
        speech_logits = speech_logits[..., self.tokenizer_text_size:].reshape(
            batch_size, -1, self.speech_token_count, self.tokenizer_speech_size)        

        if labels is not None:
            # split speech and text labels
            loss_fct = CrossEntropyLoss()
            speech_labels = labels[..., 1:].reshape(batch_size, -1, self.speech_token_count)
            labels = labels[..., 0]
            
            # calc text tokens loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.tokenizer_text_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss_text = loss_fct(shift_logits, shift_labels)

            # calc speech tokens loss
            speech_shift_logits = speech_logits[..., :-1, :, :].contiguous()
            speech_shift_labels = speech_labels[..., 1:, :].contiguous()
            speech_shift_logits = speech_shift_logits.view(-1, self.tokenizer_speech_size)
            speech_shift_labels = speech_shift_labels.view(-1)
            speech_shift_labels = speech_shift_labels.to(speech_shift_logits.device)
            if torch.all(speech_shift_labels == IGNORE_INDEX):
                loss_speech = speech_shift_logits.mean() * 0.0
            else:
                loss_speech = loss_fct(speech_shift_logits, speech_shift_labels)

            loss = loss_text + loss_speech

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MGMCausalLMOutputWithPast(
            loss=loss,
            loss_text=loss_text,
            loss_speech=loss_speech,
            logits=logits,
            speech_logits=speech_logits,
            past_key_values=outputs.past_key_values,
            speech_past_key_values=speech_outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        input_ids_refer: Optional[torch.Tensor] = None,
        audio_refer: Optional[torch.Tensor] = None,
        check_tts_result: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        inputs, refer_speech = self.prepare_inputs_for_speech_generation(
            inputs, input_ids_refer, audio_refer
        )
        if streamer is not None and isinstance(streamer, MGMOmniStreamer):
            streamer.set_refer_speech(refer_speech)
        # prepare refer speech

        index = (inputs[0, :, 0] == self.audio_start).nonzero(as_tuple=True)[0][-1] + 1
        self.text_tokens_buffer = inputs[0, index:-2, 0]
        inputs = inputs[:, :index]
        # put tts text into buffer

        (
            inputs,
            position_ids,
            attention_mask,
            _,
            inputs_embeds,
            _,
        ) = self.prepare_inputs_labels_for_speech_generation(
            inputs,
            position_ids,
            attention_mask,
            None,
            None
        )

        outputs = MGMTTSGeneration.generate(
            self,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict_in_generate=True,
            streamer=streamer,
            **kwargs
        )
        speech_ids = outputs.sequences.reshape(1, -1, self.speech_token_count+1)
        if streamer is None or not isinstance(streamer, MGMOmniStreamer):
            if getattr(self, "speechlm", None) is not None:
                audio = self.speechlm.token2audio(speech_ids, refer_speech)
            else:
                audio = self.token2audio(speech_ids, refer_speech)
        else:
            audio = None

        if check_tts_result:
            tts_error = outputs.tts_error
            return speech_ids, audio, tts_error
        return speech_ids, audio
    
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )
        model_kwargs['position_ids'] = model_kwargs['cache_position'].unsqueeze(0)
        if getattr(outputs, "speech_past_key_values", None) is not None:
            model_kwargs["speech_past_key_values"] = outputs.speech_past_key_values
        return model_kwargs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        speech_past_key_values = kwargs.pop("speech_past_key_values", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if speech_past_key_values is not None:
            _inputs['speech_past_key_values'] = speech_past_key_values
        return _inputs

AutoConfig.register("MGMTTS", MGMTTSConfig)
AutoModelForCausalLM.register(MGMTTSConfig, MGMTTSForCausalLM)