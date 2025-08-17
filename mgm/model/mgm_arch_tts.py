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

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from mgm.constants import IGNORE_INDEX


class MGMTTSMetaModel:

    def __init__(self, config):
        super(MGMTTSMetaModel, self).__init__(config)

    
class MGMTTSMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def prepare_inputs_labels_for_speech_generation(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, **kwargs
    ):
        for input_id in input_ids:
            has_speech = (input_id[:, 0] != IGNORE_INDEX)
            for i in range(self.speech_token_count):
                input_id[has_speech, i+1] += self.tokenizer_text_size + self.tokenizer_speech_size * i
        
        # decoding stage
        if input_ids.shape[1] == 1:
            if past_key_values is not None and input_ids.shape[1] == 1:
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            input_ids = input_ids.to(self.device)
            inputs_embeds = self.get_model().embed_tokens(input_ids[0]).mean(dim=1).unsqueeze(1)
            return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if len(attention_mask.shape) == 2:
            attention_mask = attention_mask.unsqueeze(-1).expand(-1, -1, self.speech_token_count+1)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        labels = [cur_labels.reshape(-1, self.speech_token_count + 1) for cur_labels in labels]

        new_input_embeds = []
        new_labels = []
        new_position_ids = []
        cur_im_idx = 0
        cur_sp_idx = 0
        cur_device = input_ids[0].device
        for batch_idx, cur_input_ids in enumerate(input_ids):

            cur_input_ids = cur_input_ids.reshape(-1, self.speech_token_count+1)

            # text embedding
            text_input_ids = cur_input_ids[:, 0]
            cur_input_embeds = self.get_model().embed_tokens(text_input_ids)

            # speech encoding
            has_speech = cur_input_ids[:, 1] != IGNORE_INDEX
            speech_input_ids = cur_input_ids[has_speech, 1:]
            speech_input_embeds = self.get_model().embed_tokens(speech_input_ids)
            speech_input_embeds = speech_input_embeds.reshape(-1, self.speech_token_count, cur_input_embeds.shape[-1])

            # merge text and speech embedding
            speech_embeds = torch.cat((cur_input_embeds[has_speech].unsqueeze(1), speech_input_embeds), dim=1).mean(dim=1)
            cur_input_embeds[has_speech] = speech_embeds

            cur_position_ids = torch.arange(len(cur_input_ids), dtype=torch.long, device=cur_device)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            new_position_ids.append(cur_position_ids)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_position_ids = [x[..., :tokenizer_model_max_length] for x in new_position_ids]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len, self.speech_token_count+1), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels, cur_position_ids) in enumerate(zip(new_input_embeds, new_labels, new_position_ids)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = cur_position_ids
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = cur_position_ids
                    
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

