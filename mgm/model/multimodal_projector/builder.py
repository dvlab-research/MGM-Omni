import torch
import torch.nn as nn
import re
import pdb

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


# Refer to LLaMA-Omni
class MLPSpeechProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.speech_encoder_ds_rate
        self.encoder_dim = config.speech_encoder_hidden_size
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, config.hidden_size)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class DualMLPSpeechProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.speech_encoder_ds_rate
        self.encoder_dim = config.speech_encoder_hidden_size
        
        dim1 = 1280
        dim2 = config.speech_encoder_hidden_size - dim1
        self.dual_query  = nn.Sequential(nn.LayerNorm(dim1), 
                                                      nn.Linear(dim1, dim1))
        self.dual_aux  = nn.Sequential(nn.LayerNorm(dim2),
                                                    nn.Linear(dim2, dim1))
        self.dual_val  = nn.Sequential(nn.LayerNorm(dim2),
                                                    nn.Linear(dim2, dim1))
        
        self.linear1 = nn.Linear(dim1 * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, config.hidden_size)

    def forward(self, x):
        x_main, x_aux = x
        batch_size, seq_len, _ = x_main.size()
        x_aux = x_aux.reshape(batch_size, seq_len, -1, x_aux.shape[-1])
        
        embed_query = self.dual_query(x_main)
        embed_aux = self.dual_aux(x_aux)
        embed_value = self.dual_val(x_aux) 
        embed_att = embed_query[:,:,None] @ (embed_aux.transpose(-1,-2) / (embed_aux.shape[-1]**0.5))
        embed_att = embed_att.nan_to_num()
        embed_feat = (embed_att.softmax(-1) @ embed_value).mean(2)
        x_main = x_main + embed_feat

        batch_size, seq_len, dim = x_main.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x_main = x_main[:, :-num_frames_to_discard, :]
        seq_len = x_main.size(1)
        
        x_main = x_main.contiguous()
        x_main = x_main.view(batch_size, seq_len // self.k, dim * self.k)
        x_main = self.linear1(x_main)
        x_main = self.relu(x_main)
        x_main = self.linear2(x_main)
        return x_main



class DualMLPSpeechProjector_type2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.speech_encoder_ds_rate
        self.encoder_dim = config.speech_encoder_hidden_size
        self.linear1 = nn.Linear(self.encoder_dim * self.k * 3, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, config.hidden_size)

    def forward(self, x):
        x_main, x_aux = x

        bsz, seq_len_main, dim = x_main.size()
        bsz, seq_len_aux,   _  = x_aux.size()
        x_aux = x_aux.contiguous()
        x_aux = x_aux.view(bsz, seq_len_main, seq_len_aux // seq_len_main * dim)
        x_concat = torch.cat([x_main, x_aux], dim=-1)

        batch_size, seq_len, dim = x_concat.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x_concat = x_concat[:, :-num_frames_to_discard, :]
        seq_len = x_concat.size(1)
        
        x_concat = x_concat.contiguous()
        x_concat = x_concat.view(batch_size, seq_len // self.k, dim * self.k)
        x_concat = self.linear1(x_concat)
        x_concat = self.relu(x_concat)
        x_concat = self.linear2(x_concat)

        return x_concat
    
class DualMLPSpeechProjector_type2_norm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.speech_encoder_ds_rate
        self.encoder_dim = config.speech_encoder_hidden_size
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 4096)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4096, config.hidden_size)
        # whisper and qwen2-audio
        dim1 = 1280
        dim2 = config.speech_encoder_hidden_size - dim1
        self.norm1 = nn.LayerNorm(dim1)
        self.norm2 = nn.LayerNorm(dim2)

    def forward(self, x):
        x_main, x_aux = x
        bsz, seq_len_main, _ = x_main.size()
        bsz, seq_len_aux,  dim_aux  = x_aux.size()
        x_aux = x_aux.contiguous()
        x_aux = x_aux.view(bsz, seq_len_main, seq_len_aux // seq_len_main * dim_aux)
        x_main = self.norm1(x_main)
        x_aux = self.norm2(x_aux)
        x_concat = torch.cat([x_main, x_aux], dim=-1)

        batch_size, seq_len, dim = x_concat.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x_concat = x_concat[:, :-num_frames_to_discard, :]
        seq_len = x_concat.size(1)
        
        x_concat = x_concat.contiguous()
        x_concat = x_concat.view(batch_size, seq_len // self.k, dim * self.k)
        x_concat = self.linear1(x_concat)
        x_concat = self.relu(x_concat)
        x_concat = self.linear2(x_concat)

        return x_concat
    

def build_speech_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_speech_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_speech_hidden_size, config.hidden_size)
    
    if projector_type == 'simple_mlp':
        return MLPSpeechProjector(config=config)
    
    if projector_type == 'dual_mlp' or projector_type == 'dual_mlp_attn':
        return DualMLPSpeechProjector(config=config)
    
    if projector_type == 'dual_mlp_type2':
        return DualMLPSpeechProjector_type2(config=config)
    
    if projector_type == 'dual_mlp_type2_norm':
        return DualMLPSpeechProjector_type2_norm(config=config)
        
    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')