from transformers import TextIteratorStreamer 
import uuid
import torch
import numpy as np
import soundfile as sf
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from transformers.generation.streamers import TextIteratorStreamer


class MGMOmniStreamer(TextIteratorStreamer):
    def __init__(self, tokenizer, cosyvoice, max_audio_token, skip_prompt=False, **decode_kwargs):
        self.out_q = queue.Queue()
        self._STOP = object()

        self.uuid = str(uuid.uuid1())
        self.cosyvoice = cosyvoice
        self.cosyvoice.hift_cache_dict[self.uuid] = None
        self.max_audio_token = max_audio_token - 2
        self.hop_len = 100
        self.this_hop_len = 100
        self.lookahead_len = self.cosyvoice.flow.pre_lookahead_len
        self.token_offset = 0
        self.speech_tokens = None
        self.audio = None

        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="audio_processor")
        self.audio_queue = queue.Queue(maxsize=8)
        self.is_processing = False
        self._shutdown = False
        self.audio_thread = threading.Thread(target=self._audio_processor, daemon=True)
        self.audio_thread.start()

        super().__init__(tokenizer, skip_prompt=skip_prompt, **decode_kwargs)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.out_q.get() 
        if item is self._STOP:
            raise StopIteration
        return item

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if text:
            self.out_q.put(("text", text))
        if stream_end:
            self.out_q.put(self._STOP)

    def set_refer_speech(self, refer_speech):
        self.ref_tokens, self.ref_feat, self.ref_embedding = refer_speech
        prompt_token_pad = int(np.ceil(self.ref_tokens.shape[1] / self.hop_len) * self.hop_len - self.ref_tokens.shape[1])
        self.this_hop_len += prompt_token_pad

    def _audio_processor(self):
        while not self._shutdown:
            try:
                task = self.audio_queue.get(timeout=1.0)
                if task is None:
                    break
                task_type, data = task
                if task_type == "dump_audio":
                    self._dump_audio_sync(data["finalize"])
                elif task_type == "finalize":
                    self._dump_audio_sync(True)
                self.audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio processing error: {e}")
            finally:
                self.is_processing = False

    def _dump_audio_sync(self, finalize=False):
        try:
            token_offset = self.token_offset
            self.token_offset += self.this_hop_len
            self.this_hop_len = self.hop_len * 2
            cur_audio = self.cosyvoice.token2wav(
                token=self.speech_tokens[:, :self.token_offset + self.lookahead_len],
                prompt_token=self.ref_tokens,
                prompt_feat=self.ref_feat,
                embedding=self.ref_embedding,
                uuid=self.uuid,
                token_offset=token_offset,
                stream=(not finalize),
                finalize=finalize
            ).detach().squeeze().cpu()
            self.audio = cur_audio if self.audio is None else torch.cat((self.audio, cur_audio), dim=0)
            self.out_q.put(("audio", cur_audio.numpy()))
        except Exception as e:
            print(f"Error in audio processing: {e}")

    def dump_audio_async(self, finalize=False):
        if not self._shutdown:
            self.audio_queue.put(("dump_audio", {"finalize": finalize}))

    def put(self, value):
        if hasattr(value, "shape") and len(value.shape) > 1 and value.shape[1] >= 4:
            if value.shape[1] == 5:
                speech_tokens = value[:, 1:]
            else:
                speech_tokens = value
            if speech_tokens.max() < self.max_audio_token:
                if self.speech_tokens is not None:
                    self.speech_tokens = torch.cat((self.speech_tokens, speech_tokens), dim=1)
                else:
                    self.speech_tokens = speech_tokens
                if self.speech_tokens.numel() - self.token_offset >= self.this_hop_len + self.lookahead_len and not self.is_processing:
                    self.is_processing = True
                    self.dump_audio_async()
        else:
            super().put(value)

    def end(self):
        if not self._shutdown:
            self.audio_queue.put(("finalize", {}))
        self.audio_queue.join()

        self._shutdown = True
        self.audio_queue.put(None)
        if self.audio_thread.is_alive():
            self.audio_thread.join(timeout=5.0)
        self.executor.shutdown(wait=True)

        super().end()
        self.out_q.put(self._STOP)

    def __del__(self):
        if hasattr(self, '_shutdown') and not self._shutdown:
            self._shutdown = True
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)