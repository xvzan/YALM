from datasets import load_dataset
from transformers import GPT2TokenizerFast, TrainingArguments, Trainer

from model import YALMGPT2Model, YALMGPT2Config
from coach_collator_wt import CoachCollator

import torch
torch.set_default_dtype(torch.bfloat16)

new_tokenizer = GPT2TokenizerFast.from_pretrained(
    "openai-community/gpt2", cache_dir="hfcache"
)

config = YALMGPT2Config(n_positions=768, n_layer=8, n_embd=512, n_head=8)
model = YALMGPT2Model(config)


ds = load_dataset("facebook/belebele", "zho_Hans", cache_dir="hfcache", split="test")
ds2 = ds.rename_column("flores_passage", "text")

c2r = [i for i in ds2.column_names if i != "text"]
ds2 = ds2.remove_columns(c2r)

new_tokenizer.model_max_length = config.n_positions
new_tokenizer.pad_token = new_tokenizer.eos_token


coach = CoachCollator(ds2, new_tokenizer)

training_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=2,
    output_dir="./t2",
    label_names=["output_ids", "dni_labels", "text", "dni_masks", "loss_mask"],
    bf16=True,
    # deepspeed=dscfg,
)
trainer = Trainer(
    model=model,
    train_dataset=ds2,
    processing_class=new_tokenizer,
    data_collator=coach,
    args=training_args,
)

trainer.train()
