import copy
import re
import torch
import inspect
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Callable
import torch.distributed as dist

from mgm.model.multimodal_generator.mgm_omni_streamer import MGMOmniStreamer
from mgm.constants import BLANK_SPEECH_TOKENS

import transformers
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    GenerationConfig,
    GenerationMode,
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerateOutput, 
    GenerationMixin,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    GenerateNonBeamOutput,
    is_deepspeed_zero3_enabled,
    is_fsdp_managed_module,
    logging
)

logger = logging.get_logger(__name__)


class MGMOmniGeneration(GenerationMixin):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        inputs_speech: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        custom_generate: Optional[str] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)  # only used for assisted generation
        self.tokenizer = tokenizer
        self.assistant_tokenizer = assistant_tokenizer

        # get speech generator params
        inputs_embeds_speech = kwargs.pop("inputs_embeds_speech", None)
        attention_mask_speech = kwargs.pop("attention_mask_speech", None)
        position_ids_speech = kwargs.pop("position_ids_speech", None)
        kwargs_speech = copy.deepcopy(kwargs)
        kwargs_speech['inputs_embeds'] = inputs_embeds_speech
        kwargs_speech['attention_mask'] = attention_mask_speech
        kwargs_speech['position_ids'] = position_ids_speech
        del kwargs_speech['rope_deltas']

        self.sentence_end_list = ['.', '?', '!', ';', '...', '。', '？', '！', '；', '\n']

        # get model_kwargs
        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

        _, model_kwargs_speech = self._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs_speech
        )
        self._validate_model_kwargs(model_kwargs_speech.copy())
        self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        inputs_tensor_speech, model_input_name, model_kwargs_speech = self._prepare_model_inputs(
            inputs_speech, generation_config.bos_token_id, model_kwargs_speech
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )
        elif kwargs_has_attention_mask:
            # TODO (joao): generalize this check with other types of inputs
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")
            input_ids_speech = inputs_tensor_speech if model_input_name == "input_ids" else model_kwargs_speech.pop("input_ids")
            input_ids_speech = input_ids_speech.reshape((1, 0, self.speechlm.speech_token_count + 1)).clone()

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
        # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
        # dynamically overrides this value as it can need more than the last token logits
        if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
        # - different models have a different cache name expected by the model (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache length
        max_cache_length = generation_config.max_length - 1
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(
            generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
        )
        # prepare kvcache for speech generator
        self._prepare_cache_for_generation(
            generation_config, model_kwargs_speech, assistant_model, batch_size, max_cache_length, device
        )

        # 8. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 9. prepare logits processors and stopping criteria
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )

        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache
        model_kwargs_speech["use_cache"] = generation_config.use_cache

        # 10. go into different generation modes
        if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # 11. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            result = self._sample(
                input_ids,
                input_ids_speech,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                model_kwargs=model_kwargs,
                model_kwargs_speech=model_kwargs_speech,
            )
        else:
            raise NotImplementedError

        # Convert to legacy cache format if requested
        if (
            generation_config.return_legacy_cache is True
            and hasattr(result, "past_key_values")
            and getattr(result.past_key_values, "to_legacy_cache") is not None
        ):
            result.past_key_values = result.past_key_values.to_legacy_cache()
        return result
    

    def is_sentence_end(self, token):
        text = self.assistant_tokenizer.decode(token)
        return (text[0] in self.sentence_end_list or text[-1] == '\n')


    def process_first_token(self):
        while len(self.text_tokens_buffer) > 0:
            token = self.text_tokens_buffer[0]
            text = self.assistant_tokenizer.decode(token)
            if not re.fullmatch(r'[ \n*#]+', text): break
            self.text_tokens_buffer = self.text_tokens_buffer[1:]
        # remove blank token


    def _sample(
        self,
        input_ids: torch.LongTensor,
        input_ids_speech: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        model_kwargs: dict,
        model_kwargs_speech: dict,
        **kwargs
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
        cur_len_speech = input_ids_speech.shape[1]
        this_peer_finished = False
        this_peer_finished_speech = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        unfinished_sequences_speech = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
        model_kwargs_speech = self._get_initial_cache_position(cur_len_speech, input_ids_speech.device, model_kwargs_speech)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            # If we use FA2 and a static cache, we cannot compile with fullgraph
            if self.config._attn_implementation == "flash_attention_2" and getattr(
                model_kwargs.get("past_key_values"), "is_compileable", False
            ):
                if generation_config.compile_config is None:
                    generation_config.compile_config = CompileConfig(fullgraph=False)
                # only raise warning if the user passed an explicit compile-config (otherwise, simply change the default without confusing the user)
                elif generation_config.compile_config.fullgraph:
                    logger.warning_once(
                        "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                        "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                    )
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
            is_prefill_speech = False
        else:
            is_prefill = True
            is_prefill_speech = True
        
        # prapare speech generation
        is_first_text_token = True
        next_text_padding = False
        next_audio_sep_token = None
        is_audio_eos_token = False
        text_token_count = 0
        speech_pad = self.speechlm.tokenizer_speech_size - 1
        speech_eos = self.speechlm.tokenizer_speech_size - 2
        text_audio_start = self.speechlm.tokenizer_text_size - 3
        text_audio_end = self.speechlm.tokenizer_text_size - 2
        text_audio_sep = self.speechlm.tokenizer_text_size - 1
        self.text_tokens_buffer = []
        speech_blank_count = 0
        speech_token_count = 0
        tts_error = False

        while self._has_unfinished_sequences(this_peer_finished_speech, synced_gpus, device=input_ids.device):
            # if text generation if not finished
            if not this_peer_finished:
                # prepare model inputs
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                # prepare variable output controls (note: some models won't accept all output controls)
                model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

                if is_prefill:
                    outputs = self(**model_inputs, return_dict=True)
                    is_prefill = False
                else:
                    outputs = model_forward(**model_inputs, return_dict=True)

                # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                )
                if synced_gpus and this_peer_finished:
                    continue

                # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                # (the clone itself is always small)
                next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

                # pre-process distribution
                next_token_scores = logits_processor(input_ids, next_token_logits)

                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_logits:
                        raw_logits += (next_token_logits,)
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

                # token selection
                if do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)
                
                # add text tokens to buffer
                if next_tokens != self.tokenizer.eos_token_id:
                    self.text_tokens_buffer.append(next_tokens)
                
                # finished sentences should have their next token be a padding token
                if has_eos_stopping_criteria:
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                
                # update generated ids, model inputs, and length for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                if streamer is not None:
                    streamer.put(next_tokens.cpu())
                
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
                this_peer_finished = unfinished_sequences.max() == 0
                cur_len += 1
                
                del outputs

            # forward speech tokens
            model_inputs_speech = self.speechlm.prepare_inputs_for_generation(input_ids_speech, **model_kwargs_speech)
            outputs_speech = self.speechlm(**model_inputs_speech, return_dict=True)
            model_kwargs_speech = self.speechlm._update_model_kwargs_for_generation(
                outputs_speech,
                model_kwargs_speech,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            # predict speech tokens
            next_speech_logits = outputs_speech.speech_logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_speech_scores = logits_processor(input_ids_speech[0, :, 1:].T, next_speech_logits[0])
            if do_sample:
                probs = nn.functional.softmax(next_speech_scores, dim=-1)
                next_speech_tokens = torch.multinomial(probs, num_samples=1).squeeze(1).unsqueeze(0)
            else:
                next_speech_tokens = torch.argmax(next_speech_scores, dim=-1).unsqueeze(0)
            
            # check if generation fail
            blank_count = sum(token in BLANK_SPEECH_TOKENS for token in next_speech_tokens.flatten().tolist())
            if blank_count >= 3:
                speech_blank_count += 1
            else:
                speech_blank_count = 0

            # get next text token for tts
            next_text_tokens = next_tokens.clone()
            if next_text_padding:
                # add padding
                next_text_tokens[:] = pad_token_id
                if (
                    (next_speech_tokens == speech_eos).all() or
                    speech_blank_count > 10 or
                    speech_token_count > text_token_count * 8
                ):
                    next_speech_tokens[:] = speech_eos
                    next_text_padding = False
                    next_audio_sep_token = (
                        text_audio_end if this_peer_finished and len(self.text_tokens_buffer) == 0 else text_audio_sep
                    )
                    if (
                        speech_blank_count > 10 or
                        speech_token_count > text_token_count * 8
                    ):
                        tts_error = True
            elif next_audio_sep_token:
                # add audio_sep or audio_end
                next_text_tokens[:] = next_audio_sep_token
                next_speech_tokens[:] = speech_pad
                is_audio_eos_token = (next_text_tokens == text_audio_end)
                is_first_text_token = True
                next_audio_sep_token = False
                text_token_count = 0
                speech_token_count = 0
            elif is_audio_eos_token:
                next_text_tokens[:] = self.assistant_tokenizer.eos_token_id
            else:
                # assign text token from buffer
                if is_first_text_token:
                    self.process_first_token()
                    is_first_text_token = False
                if this_peer_finished and len(self.text_tokens_buffer) == 0:
                    next_text_padding = True
                    next_text_tokens[:] = pad_token_id
                else:
                    next_text_tokens[:] = self.text_tokens_buffer[0]
                    self.text_tokens_buffer = self.text_tokens_buffer[1:]
                    text_token_count += 1
                    if (
                        (self.is_sentence_end(next_text_tokens) and text_token_count > 50) or
                        text_token_count > 100
                    ):
                        next_text_padding = True
            speech_token_count += 1

            # concat text and speech tokens
            next_speech_tokens = torch.cat([next_text_tokens[:, None], next_speech_tokens], dim=1)
            if streamer is not None and isinstance(streamer, MGMOmniStreamer):
                streamer.put(next_speech_tokens.cpu())

            # update generated ids, model inputs, and length for next step
            input_ids_speech = torch.cat([input_ids_speech, next_speech_tokens[:, None]], dim=1)

            unfinished_sequences_speech = unfinished_sequences_speech & ~stopping_criteria(input_ids_speech[..., 0], scores)
            this_peer_finished_speech = unfinished_sequences_speech.max() == 0
            cur_len_speech += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs_speech

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids_speech,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids_speech,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids, input_ids_speech

