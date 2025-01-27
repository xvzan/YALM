import torch

from datasets import load_dataset  # , load_from_disk, concatenate_datasets

from transformers import GPT2TokenizerFast  # , TrainingArguments, Trainer

from model import YALMGPT2Model, YALMGPT2Config
from coach_collator_wt import CoachCollator

new_tokenizer = GPT2TokenizerFast.from_pretrained(
    "openai-community/gpt2", cache_dir="../hfcache"
)


ds = load_dataset("facebook/belebele", "eng_Latn", cache_dir="../hfcache", split="test")
ds2 = ds.rename_column("flores_passage", "text")

config = YALMGPT2Config(n_positions=2048, n_layer=96, n_embd=1024, n_head=8)

new_tokenizer.model_max_length = config.n_positions
new_tokenizer.pad_token = new_tokenizer.eos_token

coach = CoachCollator(ds2, new_tokenizer)

c01 = coach([ds2[0]])
position = (c01["labels"][0] == 50256).nonzero(as_tuple=True)[0][0]
print("labels_decoded: ", new_tokenizer.decode(c01["labels"][0][: position + 2]))
print(c01["input_ids"])
position = (c01["input_ids"][0] == 50256).nonzero(as_tuple=True)[0][0]
print("inputs_decoded: ", new_tokenizer.decode(c01["input_ids"][0][: position + 2]))

model = (
    YALMGPT2Model.from_pretrained("../checkpoint-18500", torch_dtype=torch.bfloat16)
    .to("cuda")
    .requires_grad_(False)
    .eval()
)

res = model(c01["input_ids"].to("cuda"))
probs = torch.softmax(res["logits"], dim=-1)
max_probs, max_indices = torch.max(probs, dim=-1)
print("output_decoded: ", new_tokenizer.decode(max_indices[0][: position + 3]))

sss = torch.cat((c01["dni_labels"].cuda(), res["dni_points"]), -1)
print(sss[0, : position + 3, :])
