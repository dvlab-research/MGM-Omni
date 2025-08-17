CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
PREDICT_TOKEN_INDEX = -300
SPEECH_TOKEN_INDEX = -500
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_SPEECH_TOKEN = "<speech>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"
DEFAULT_PREDICT_TOKEN = "<predict>"
AUDIO_START = '<|audio_start|>'
AUDIO_END = '<|audio_end|>'
AUDIO_SEP = '<|audio_sep|>'

DESCRIPT_PROMPT = [
    "Describe this image thoroughly.",
    "Provide a detailed description in this picture.",
    "Detail every aspect of what's in this picture.",
    "Explain this image with precision and detail.",
    "Give a comprehensive description of this visual.",
    "Elaborate on the specifics within this image.",
    "Offer a detailed account of this picture's contents.",
    "Describe in detail what this image portrays.",
    "Break down this image into detailed descriptions.",
    "Provide a thorough description of the elements in this image."]

BLANK_SPEECH_TOKENS = [
    1707, 1788, 1950, 1951, 1959, 2031, 2040, 2112, 3894, 3903,
    3975, 4056, 4137, 4138, 4143, 4146, 4164, 4173, 4218, 4219,
    4227, 4299, 4300, 5520, 5523, 5547, 5760, 5763, 5766, 5790, 
    6009, 6081, 6082, 6087, 6090, 6091, 6092, 6117, 6162, 6163, 
    6165, 6168, 6171, 6172, 6198, 6243, 6244, 6249, 6252, 6253, 
    6261, 6276, 6279, 6296, 6299, 6321, 6324, 6325, 6330, 6331, 
    6333, 6334, 6335, 6339, 6342, 6351, 6357, 6360, 6361, 6378, 
    6387, 6399, 6402, 6405, 6406, 6408, 6411, 6412, 6413, 6414, 
    6415, 6416, 6432, 6433, 6435, 6436, 6438, 6439, 6441, 6459, 
    6460, 6461, 6466, 6468, 6469, 6480, 6483, 6486, 6487, 6489, 
    6492, 6493, 6495, 6496, 6511, 6513, 6514, 6519, 6522, 6523, 
    6540, 6541, 6549
]