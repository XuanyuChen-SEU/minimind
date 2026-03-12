from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length - 2,# 留出位置给BOS和EOS
            truncation=True,# 如果长度超过max，自动剪切
        ).input_ids

        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        input_ids = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens))# 右侧用PAD补齐到max_length，保证batch内等长
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        