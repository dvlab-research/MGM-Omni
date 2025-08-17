import copy
import re
import torch
import inspect
import warnings
from dataclasses import dataclass
import numpy as np
import torch.nn as nn
from typing import Optional, Union, List, Callable
import torch.distributed as dist

from mgm.constants import BLANK_SPEECH_TOKENS
from mgm.model.multimodal_generator.mgm_omni_streamer import MGMOmniStreamer

import transformers
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerationMixin,
    GenerateDecoderOnlyOutput,
    GenerateNonBeamOutput,
    logging
)

logger = logging.get_logger(__name__)


@dataclass
class MGMGenDecoderOnlyOutput(GenerateDecoderOnlyOutput):
    tts_error: Optional[bool] = None


class MGMTTSGeneration(GenerationMixin):

    @torch.no_grad()
    def generate(
        self,
        **kwargs,
    ):
        self.tokenizer = kwargs.pop("tokenizer", None)
        self.sentence_end_list = ['.', '?', '!', ';', '...', '。', '？', '！', '；', '\n']
        return super().generate(**kwargs)
    
    def is_sentence_end(self, token):
        text = self.tokenizer.decode(token)
        return (text[0] in self.sentence_end_list or text[-1] == '\n')
    
    def process_first_token(self):
        while len(self.text_tokens_buffer) > 0:
            token = self.text_tokens_buffer[0]
            text = self.tokenizer.decode(token)
            if not re.fullmatch(r'[ \n*#]+', text): break
            self.text_tokens_buffer = self.text_tokens_buffer[1:]
        # remove blank token

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        input_ids = input_ids.reshape((1, 0, self.speech_token_count+1))

        is_first_text_token = True
        next_text_padding = False
        next_audio_sep_token = False
        is_audio_eos_token = False
        text_token_count = 0
        speech_token_count = 0
        speech_blank_count = 0
        tts_error = False

        speech_pad = self.config.tokenizer_speech_size - 1
        speech_eos = self.config.tokenizer_speech_size - 2
        text_audio_start = self.tokenizer_text_size - 3
        text_audio_end = self.tokenizer_text_size - 2
        text_audio_sep = self.tokenizer_text_size - 1

        # if streamer is not None:
        #     streamer.put(torch.tensor([text_audio_start]))

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need
            
            # speech tokens pre-process
            next_speech_logits = outputs.speech_logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_speech_scores = logits_processor(input_ids[0, ..., 1:].T, next_speech_logits[0])

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_speech_scores[0],)
                if output_logits:
                    raw_logits += (next_speech_logits[:, 0],)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )
            
            # speech token selection
            if do_sample:
                probs = nn.functional.softmax(next_speech_scores, dim=-1)
                next_speech_tokens = torch.multinomial(probs, num_samples=1).squeeze(1).unsqueeze(0)
            else:
                next_speech_tokens = torch.argmax(next_speech_scores, dim=-1).unsqueeze(0)
            
            blank_count = sum(token in BLANK_SPEECH_TOKENS for token in next_speech_tokens.flatten().tolist())
            if blank_count >= 3:
                speech_blank_count += 1
            else:
                speech_blank_count = 0
            
            # get next text token
            next_tokens = next_speech_tokens[..., 0].clone()
            if next_text_padding:
                # add padding
                next_tokens[:] = self.tokenizer.pad_token_id
                if (
                    (next_speech_tokens == speech_eos).all() or
                    speech_blank_count > 8 or
                    speech_token_count > text_token_count * 8
                ):
                    next_speech_tokens[:] = speech_eos
                    next_text_padding = False
                    next_audio_sep_token = (
                        text_audio_end if len(self.text_tokens_buffer) == 0 else text_audio_sep
                    )
                    if (
                        speech_blank_count > 8 or
                        speech_token_count > text_token_count * 8
                    ):
                        tts_error = True
            elif next_audio_sep_token:
                # add audio_sep or audio_end
                next_tokens[:] = next_audio_sep_token
                next_speech_tokens[:] = speech_pad
                is_audio_eos_token = (next_tokens == text_audio_end)
                is_first_text_token = True
                next_audio_sep_token = False
                text_token_count = 0
                speech_token_count = 0
            elif is_audio_eos_token:
                # add eos
                next_tokens[:] = self.tokenizer.eos_token_id
            else:
                # add text
                if is_first_text_token:
                    self.process_first_token()
                    is_first_text_token = False
                next_tokens[:] = self.text_tokens_buffer[0]
                self.text_tokens_buffer = self.text_tokens_buffer[1:]
                text_token_count += 1
                if (
                    (self.is_sentence_end(next_tokens) and text_token_count > 50) or
                    text_token_count > 100 or
                    len(self.text_tokens_buffer) == 0
                ):
                    next_text_padding = True
            speech_token_count += 1

            if streamer is not None and isinstance(streamer, MGMOmniStreamer):
                streamer.put(next_speech_tokens.cpu())

            is_eos_token = ((next_tokens == self.tokenizer.eos_token_id) & is_audio_eos_token)
            next_tokens = torch.cat((next_tokens[:, None], next_speech_tokens), dim=1)
            
            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=1)
            if streamer is not None:
                if next_tokens[0, 0] < text_audio_start:
                    streamer.put(next_tokens[:, 0].cpu())

            unfinished_sequences = unfinished_sequences & ~is_eos_token
            this_peer_finished = unfinished_sequences.max() == 0 | (input_ids.shape[1] > generation_config.max_new_tokens)

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return MGMGenDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
                tts_error=tts_error,
            )
        else:
            return input_ids
