import random
from typing import Any, Dict, List, Mapping

from transformers.data.data_collator import DataCollatorMixin, InputDataClass
import torch
import numpy as np


class CoachCollator(DataCollatorMixin):
    def __init__(self, err_source, tokenizer):
        self.err_source = err_source
        self.tokenizer = tokenizer
        self.pad_base = self.tokenizer(self.tokenizer.pad_token)["input_ids"][-1:]

    def delete_tokens(self, ds, dc, t, im, dm):
        de = ds + dc
        del t[ds:de]
        del im[ds:de]
        del dm[ds:de]
        dm[ds - 1] = min(dc, 3)

    def insert_tokens(self, istart, ti, t, im, dm):
        t = t[:istart] + ti + t[istart:]
        mi = [1] * len(ti)
        im = im[:istart] + mi + im[istart:]
        md = [0] * len(ti)
        dm = dm[:istart] + md + dm[istart:]
        return t, im, dm

    def err_tokens_at_len(self, length: int):
        if random.random() < 0.03:
            pads = self.pad_base * length
            return pads
        random_index = random.randint(0, len(self.err_source) - 1)
        erids = self.tokenizer(self.err_source[random_index]["text"], truncation=False)[
            "input_ids"
        ]
        token_start = random.randint(0, len(erids))
        tks = erids[token_start : token_start + length]
        lackcount = length - len(tks)
        if lackcount > 0:
            tks = tks + self.err_tokens_at_len(lackcount)
        return tks

    def generate_data(self, tokens):
        original_tokens = tokens.copy()

        insert_marks = [0] * len(tokens)
        delete_marks = [0] * len(tokens)

        num_tokens = len(original_tokens)
        modified_tokens = tokens.copy()
        is_delete = random.random() < 0.8
        delete_count = 0
        insert_count = 0

        ts = 0

        if is_delete:
            # 删除词元
            delete_count = random.randint(1, num_tokens - 2)
            delete_start = random.randint(1, num_tokens - delete_count)
            self.delete_tokens(
                delete_start, delete_count, modified_tokens, insert_marks, delete_marks
            )
            ts = delete_start
            # print(delete_marks[:he])
            if random.random() < 0.9 and delete_count <= num_tokens / 2:
                insert_count = random.randint(
                    1, min(int(delete_count * 1.5), int(num_tokens * 0.5))
                )
                modified_tokens, insert_marks, delete_marks = self.insert_tokens(
                    delete_start,
                    self.err_tokens_at_len(insert_count),
                    modified_tokens,
                    insert_marks,
                    delete_marks,
                )
                ts += insert_count
        else:
            if random.random() < 0.9:
                insert_count = random.randint(1, int(num_tokens * 0.5))
                insert_start = random.randint(1, num_tokens)
                modified_tokens, insert_marks, delete_marks = self.insert_tokens(
                    insert_start,
                    self.err_tokens_at_len(insert_count),
                    modified_tokens,
                    insert_marks,
                    delete_marks,
                )
                ts = insert_start + insert_count
            else:
                insert_count = num_tokens
        ec = delete_count + insert_count

        return original_tokens, modified_tokens, insert_marks, delete_marks, ts, ec

    def recursion_modify(self, original, modified, inserts, deletes, ts, ec):
        if (
            ts < len(modified) - 2
            and ec / len(original) < 0.5
            and random.random() < 0.95
        ):
            _, mm, ii, dd, ts2, ec2 = self.generate_data(modified[ts:])
            ec = ec + ec2
            _, mm2, ii2, dd2, ec = self.recursion_modify(original, mm, ii, dd, ts2, ec)
            modified = modified[:ts] + mm2
            inserts = inserts[:ts] + ii2
            deletes = deletes[:ts] + dd2
        return original, modified, inserts, deletes, ec

    def __call__(self, features: List[InputDataClass]) -> Dict[str, Any]:
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        edited_features = []
        for example in features:
            edited = {}
            example = self.tokenizer(example["text"], truncation=False)
            original, modified, inserts, deletes, _ = self.recursion_modify(
                example["input_ids"], example["input_ids"], [], [], 0, 0
            )
            mask_length = len(original) + 2
            real_mask_len = len(modified) + 2
            if mask_length >= self.tokenizer.model_max_length:
                loss_mask = [1] * self.tokenizer.model_max_length
            else:
                loss_mask = [1] * mask_length + [0] * (
                    self.tokenizer.model_max_length - mask_length
                )
            if real_mask_len >= self.tokenizer.model_max_length:
                attention_mask = [1] * self.tokenizer.model_max_length
            else:
                attention_mask = [1] * real_mask_len + [0] * (
                    self.tokenizer.model_max_length - real_mask_len
                )
            if len(original) > self.tokenizer.model_max_length:
                original = original[: self.tokenizer.model_max_length]
            if len(modified) > self.tokenizer.model_max_length:
                modified = modified[: self.tokenizer.model_max_length]
            if len(inserts) > self.tokenizer.model_max_length:
                inserts = inserts[: self.tokenizer.model_max_length]
            if len(deletes) > self.tokenizer.model_max_length:
                deletes = deletes[: self.tokenizer.model_max_length]

            if len(original) < self.tokenizer.model_max_length:
                original = original + self.pad_base * (
                    self.tokenizer.model_max_length - len(original)
                )
            if len(modified) < self.tokenizer.model_max_length:
                modified = modified + self.pad_base * (
                    self.tokenizer.model_max_length - len(modified)
                )
            if len(inserts) < self.tokenizer.model_max_length:
                inserts = inserts + [0] * (
                    self.tokenizer.model_max_length - len(inserts)
                )
            if len(deletes) < self.tokenizer.model_max_length:
                deletes = deletes + [0] * (
                    self.tokenizer.model_max_length - len(deletes)
                )
            # result = {}
            original = torch.tensor(original)
            modified = torch.tensor(modified)
            loss_mask = torch.tensor(loss_mask)
            diff = torch.where(original == modified, 0, 1)
            rand = torch.rand(diff.size())
            threshold = (torch.sum(diff) + 1) / (torch.sum(loss_mask) * 5)
            diff_and_rand = torch.where(rand < threshold, 1, diff)
            loss_mask = loss_mask * diff_and_rand
            edited["labels"] = original
            edited["loss_mask"] = loss_mask
            edited["attention_mask"] = attention_mask
            edited["input_ids"] = modified
            t_del = torch.tensor(deletes)
            real_mask_len = min(real_mask_len, self.tokenizer.model_max_length)
            dmn = min(sum(deletes), self.tokenizer.model_max_length)
            m_del = torch.clamp(t_del, 0, 1)
            m_del[torch.randperm(real_mask_len)[:dmn]] = 1
            t_ins = torch.tensor(inserts)
            imn = min(sum(inserts), self.tokenizer.model_max_length)
            m_ins = torch.clamp(t_ins, 0, 1)
            m_ins[torch.randperm(real_mask_len)[:imn]] = 1
            edited["dni_labels"] = torch.stack([t_del, t_ins]).transpose(-1, -2)
            edited["dni_masks"] = torch.stack([m_del, m_ins]).transpose(-1, -2)
            edited_features.append(edited)
        first = edited_features[0]
        batch = {}
        for k, v in first.items():
            if v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in edited_features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.from_numpy(
                        np.stack([f[k] for f in edited_features])
                    )
                else:
                    batch[k] = torch.tensor([f[k] for f in edited_features])
        return batch
