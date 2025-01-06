from transformers import GPT2Model, GPT2Config, GPT2PreTrainedModel
from transformers.utils import ModelOutput
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss


class YALMGPT2Config(GPT2Config):
    model_type = "YALM"

    def __init__(
        self,
        dni_loss_threshold=0.01,
        **kwargs,
    ):
        self.dni_loss_threshold = dni_loss_threshold
        super().__init__(**kwargs)


@dataclass
class YALMOutputWithCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    dni_points: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class YALMGPT2Model(GPT2PreTrainedModel):

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.dni_hidden_size = int(config.n_embd / 4)
        self.lm_n_embd = config.n_embd - self.dni_hidden_size
        self.lm_head = nn.Linear(self.lm_n_embd, config.vocab_size, bias=False)
        self.dni_head = nn.Linear(self.dni_hidden_size, 2, bias=False)
        self.dni_loss_threshold = torch.tensor(config.dni_loss_threshold)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        dni_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, YALMOutputWithCrossAttentions]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=None,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states[..., : self.lm_n_embd])
        dni_points = self.dni_head(hidden_states[..., self.lm_n_embd :])

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)

            # print("lm_logits",lm_logits.size())
            # print("labels",labels.size())

            # Shift so that tokens < n predict n
            shift_logits = lm_logits.contiguous()
            shift_labels = labels.contiguous()

            # print("shift_logits",shift_logits.size())
            # print("shift_labels",shift_labels.size())

            dni_labels = dni_labels.to(dni_points.device)
            shift_points = dni_points.contiguous()
            shift_dni = dni_points.contiguous()

            # print("shift_points",shift_points.size())
            # print("shift_dni",shift_dni.size())

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            dni_loss_fct = MSELoss()
            dni_loss = dni_loss_fct(shift_points, shift_dni)

            # if dni_loss >= self.dni_loss_threshold or dni_loss >= loss:
            #     loss = loss + dni_loss
            loss = torch.where(
                dni_loss > torch.min(loss, self.dni_loss_threshold),
                loss + dni_loss,
                loss,
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return YALMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            dni_points=dni_points,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            # attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
