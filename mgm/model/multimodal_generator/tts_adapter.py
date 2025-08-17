import copy
import torch.nn as nn
import transformers
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding, Qwen3RMSNorm
from transformers.modeling_outputs import BaseModelOutputWithPast
from mgm.constants import IGNORE_INDEX


def build_tts_adapter(config):
    generator_type = getattr(config, 'tts_adapter_type', 'identity')
    if generator_type == 'qwen3':
        return Qwen3TTSAdapter(config)

    raise ValueError(f'Unknown generator type: {generator_type}')


class Qwen3TTSAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_layers, n_dims, n_heads, n_inter_dims = config.tts_adapter_config[1:-1].split(",")
        if '_' in n_heads:
            n_heads, gqa = n_heads.split('_')
            gqa = int(gqa)
        else:
            gqa = 1
        _config = copy.deepcopy(config)
        _config.hidden_size = int(n_dims)
        _config.num_hidden_layers = int(n_layers)
        _config.num_attention_heads = int(n_heads)
        _config.num_key_value_heads = int(n_heads) // gqa
        _config.intermediate_size = int(n_inter_dims)
        _config._attn_implementation = "flash_attention_2"
        self.config = _config
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(_config, layer_idx) for layer_idx in range(int(n_layers))]
        )
        self.rotary_emb = Qwen3RotaryEmbedding(config=_config)
    
    def forward(self, hidden_states, attention_mask, position_ids, use_cache, past_key_values, cache_position, **kwargs):
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
                past_key_value=past_key_values,
                cache_position=cache_position
            )
            hidden_states = layer_outputs[0]
        
        output = BaseModelOutputWithPast(
            hidden_states=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
        return output

