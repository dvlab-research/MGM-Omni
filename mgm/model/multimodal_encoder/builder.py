import os
from .siglip_encoder import SiglipVisionTower
from .qwen2_5_vl_encoder import Qwen2_5_VLVisionTower
from .whisper_encoder import WhisperTower
from .qwen2_audio_encoder import Qwen2AudioTower
from .dual_audio_encoder import DualAudioTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    image_processor = getattr(vision_tower_cfg, 'image_processor', getattr(vision_tower_cfg, 'image_processor', "../processor/clip-patch14-224"))

    if "siglip" in vision_tower.lower():
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "qwen2.5-vl" in vision_tower.lower() or "qwen2_5_vl" in vision_tower.lower():
        return Qwen2_5_VLVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')

def build_speech_tower(speech_tower_cfg, **kwargs):
    speech_tower = getattr(speech_tower_cfg, 'mm_speech_tower', getattr(speech_tower_cfg, 'speech_tower', None))
    
    if "dual" in speech_tower.lower():
        speech_tower = speech_tower.replace('Dual-', '')
        return DualAudioTower(speech_tower, args=speech_tower_cfg, **kwargs)
    elif "whisper" in speech_tower.lower():
        return WhisperTower(speech_tower, args=speech_tower_cfg, **kwargs)
    elif "qwen2a" in speech_tower.lower():
        return Qwen2AudioTower(speech_tower, args=speech_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown speech tower: {speech_tower}')
