from datasets import load_dataset
from transformers import GPT2TokenizerFast, TrainingArguments, Trainer

from model import YALMGPT2Model, YALMGPT2Config
from coach_collator import CoachCollator

new_tokenizer = GPT2TokenizerFast.from_pretrained(
    "openai-community/gpt2", cache_dir="hfcache"
)

config = YALMGPT2Config(n_positions=1024, n_layer=12, n_embd=768, n_head=8)
model = YALMGPT2Model(config)

ds = load_dataset("facebook/belebele", "zho_Hans", cache_dir="hfcache", split="test")
ds2 = ds.rename_column("flores_passage", "text")

new_tokenizer.model_max_length = config.n_positions
new_tokenizer.pad_token = new_tokenizer.eos_token


def tokenize_function(example):
    return new_tokenizer(example["text"], truncation=False)


tokenized_datasets = ds2.map(tokenize_function, batched=True)

coach = CoachCollator(tokenized_datasets, new_tokenizer)

training_args = TrainingArguments(
    output_dir="./t2", label_names=["output_ids", "dni_labels"]
)
trainer = Trainer(
    model=model,
    train_dataset=tokenized_datasets,
    tokenizer=new_tokenizer,
    data_collator=coach,
    args=training_args,
)

trainer.train()
