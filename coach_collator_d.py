import random
from typing import Any, Dict, List, Mapping

from transformers.data.data_collator import DataCollatorMixin, InputDataClass
import torch
import numpy as np


class CoachCollator(DataCollatorMixin):
    def __init__(self, err_source, tokenizer):
        self.err_source = err_source
        self.tokenizer = tokenizer
        self.pad_base = self.tokenizer(self.tokenizer.pad_token)["input_ids"]
        self.bos_base = self.tokenizer(self.tokenizer.bos_token)["input_ids"]

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
        random_index = random.randint(0, len(self.err_source) - 1)
        token_start = random.randint(0, len(self.err_source[random_index]["input_ids"]))
        tks = self.err_source[random_index]["input_ids"][
            token_start : token_start + length
        ]
        lackcount = length - len(tks)
        if lackcount > 0:
            tks = tks + self.err_tokens_at_len(lackcount)
        return tks

    def generate_data(self, tokens):
        # 保留全部词元
        original_tokens = tokens.copy()

        # 初始化插入标记和删除标记
        insert_marks = [0] * len(tokens)
        delete_marks = [0] * len(tokens)

        # def modify_tokens(tokens):
        num_tokens = len(tokens)
        modified_tokens = tokens.copy()
        is_delete = random.random() < 0.8
        delete_count = 0
        insert_count = 0

        he = num_tokens
        ts = 0

        if is_delete:
            # 删除词元
            delete_count = random.randint(1, num_tokens - 2)
            delete_start = random.randint(1, num_tokens - delete_count)
            self.delete_tokens(
                delete_start, delete_count, modified_tokens, insert_marks, delete_marks
            )
            he = delete_start - 1
            ts = delete_start
            # print(delete_marks[:he])
            if random.random() < 0.8 and delete_count <= num_tokens / 2:
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
                he = insert_start - 1
                ts = insert_start + insert_count
            else:
                insert_count = num_tokens
        # print(delete_count+insert_count)
        ec = delete_count + insert_count

        # print("Original:", original_tokens)
        # print("Modified:", modified_tokens)
        # print("Insert:", insert_marks)
        # print("Delete:", delete_marks)
        return original_tokens, modified_tokens, insert_marks, delete_marks, he, ts, ec

    def recursion_modify(self, original, modified, inserts, deletes, he, ts, ec):
        he = he - 1
        if he >= len(modified) - ts:
            # print("head")
            if ts2 < len(mm) - 2 and ec / len(original) < 0.5 and random.random() < 0.8:
                _, mm, ii, dd, he2, ts2, ec2 = self.generate_data(modified[: he + 1])
                ec = ec + ec2
                _, mm, ii, dd, _, _, ec = self.recursion_modify(
                    original, mm, ii, dd, he2, ts2, ec
                    )
                modified = mm + modified[he + 1 :]
                inserts = ii + inserts[he + 1 :]
                deletes = dd + deletes[he + 1 :]
        else:
            # print("tail")
            if ts2 < len(mm) - 2 and ec / len(original) < 0.5 and random.random() < 0.8:
                _, mm, ii, dd, he2, ts2, ec2 = self.generate_data(modified[ts:])
                ec = ec + ec2
                _, mm, ii, dd, _, _, ec = self.recursion_modify(
                    original, mm, ii, dd, he2, ts2, ec
                )
                modified = modified[:ts] + mm
                inserts = inserts[:ts] + ii
                deletes = deletes[:ts] + dd
        return original, modified, inserts, deletes, he, ts, ec

    def __call__(self, features: List[InputDataClass]) -> Dict[str, Any]:
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        edited_features = []
        for example in features:
            edited = {}
            tokens = self.bos_base + example["input_ids"]
            original, modified, inserts, deletes, _, _ = self.recursion_modify(
                tokens, tokens, [], [], 0, 0
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
            edited["labels"] = original
            # del example["attention_mask"]
            edited["input_ids"] = modified
            edited["dni_labels"] = [deletes, inserts]
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
