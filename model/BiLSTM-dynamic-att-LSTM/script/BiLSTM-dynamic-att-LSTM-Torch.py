#!/usr/bin/env python
# coding: utf-8

import torch
import time
import os
from datetime import datetime
from torch.cuda import is_available
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import wandb  # For logging

# Import custom utilities
from utils import *
from ChunkEvaluatorTorch import *

# Generate dictionary from training and dev files
gernate_dic("data1/dev.txt", "data1/train.txt", "data1/tag.dic")
id2label, label2id, label_list = load_dicts("data1/tag.dic")

print(id2label)
print(label2id)
print(label_list)

# Extract entity and relation types
entity_types = set([i.split("_")[1] for i in label_list if i != "O"])
relation_types = set([i.split("_")[2] for i in label_list if i != "O" and len(i.split("_")) > 2])

print(entity_types)
print(relation_types)

# Label mapping
label_map = label2id


# Dataset Preparation
def read_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            words, labels = line.strip().split("\t")
            words = words.split()
            labels = [label_map[label] for label in labels.split()]
            yield {"tokens": words, "labels": labels}


# Load dataset
train_data = list(read_data("data1/train.txt"))
dev_data = list(read_data("data1/dev.txt"))

# Vocabulary Creation
label_vocab = {label: idx for idx, label in enumerate(label_list)}

word_vocab = {word: idx + 2 for idx, word in enumerate(set([word for item in train_data for word in item["tokens"]]))}
word_vocab["PAD"] = 0
word_vocab["OOV"] = 1

char_vocab = {
    char: idx + 2
    for idx, char in enumerate(sorted(set("".join(word for item in train_data for word in item["tokens"]))))
}
char_vocab["PAD"] = 0
char_vocab["OOV"] = 1

# Token and Character Conversion
max_char_len = 15  # Set based on data


def convert_tokens_to_ids(tokens, vocab):
    return [vocab.get(token, vocab["OOV"]) for token in tokens]


def convert_word_to_char_ids(word, char_vocab):
    char_ids = [char_vocab.get(char, char_vocab["OOV"]) for char in word[:max_char_len]]
    char_ids += [char_vocab["PAD"]] * (max_char_len - len(char_ids))
    return char_ids


# Dataset Class
class NERDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        labels = item["labels"]

        token_ids = convert_tokens_to_ids(tokens, word_vocab)
        char_ids = [convert_word_to_char_ids(word, char_vocab) for word in tokens]

        return torch.tensor(token_ids), torch.tensor(char_ids), torch.tensor(labels)


# Collate Function
def collate_fn(batch):
    token_ids, char_ids, labels = zip(*batch)
    lengths = torch.tensor([len(tokens) for tokens in token_ids])

    token_ids = nn.utils.rnn.pad_sequence(token_ids, batch_first=True, padding_value=word_vocab["PAD"])
    char_ids = nn.utils.rnn.pad_sequence(
        [torch.tensor(c) for c in char_ids], batch_first=True, padding_value=char_vocab["PAD"]
    )
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=label_vocab["O"])

    return token_ids, char_ids, labels, lengths


# Dataloaders
train_loader = DataLoader(NERDataset(train_data), batch_size=16, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(NERDataset(dev_data), batch_size=16, shuffle=False, collate_fn=collate_fn)


# Self-Attention Layer
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size**0.5

    def forward(self, inputs, mask):
        queries = self.query(inputs)
        keys = self.key(inputs)
        values = self.value(inputs)

        attn_weights = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)

        return torch.matmul(attn_weights, values)


# BiLSTM-CRF Model
class BiLSTMCRF(nn.Module):
    def __init__(self, emb_size, hidden_size, word_num, label_num, char_vocab_size, char_emb_dim, alpha=10):
        super(BiLSTMCRF, self).__init__()
        self.word_emb = nn.Embedding(word_num, emb_size)
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim)

        self.cnn = nn.Conv2d(in_channels=1, out_channels=char_emb_dim, kernel_size=(5, char_emb_dim), padding=(2, 0))

        self.bilstm = nn.LSTM(emb_size + char_emb_dim, hidden_size, bidirectional=True, batch_first=True)
        self.att = SelfAttention(hidden_size * 2)
        self.bigru = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        self.lstm = nn.LSTM(hidden_size, hidden_size * 2, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, label_num)
        self.o_label_index = 195
        self.alpha = alpha
        self.loss_fn = self.custom_loss_fn  # nn.CrossEntropyLoss(reduction='none')

    def custom_loss_fn(self, logits, labels):
        standard_loss = F.cross_entropy(logits, labels, reduction="none")  # Compute standard CrossEntropyLoss
        is_o_label = (labels == self.o_label_index).float()  # Identify 'O' labels
        adjusted_loss = (1 - is_o_label) * self.alpha * standard_loss + is_o_label * standard_loss  # Adjust loss
        return adjusted_loss.sum()

    def forward(self, input_ids, char_input_ids, labels=None):
        word_embs = self.word_emb(input_ids)

        char_embs = self.char_emb(char_input_ids)  # Shape [B, Seq_Len, Max_Char_Len, Char_Emb_Dim]
        char_embs = char_embs.view(
            -1, 1, max_char_len, char_embs.shape[-1]
        )  # Reshape to [B * Seq_Len, 1, Max_Char_Len, Char_Emb_Dim]
        char_embs = self.cnn(char_embs).squeeze(2).max(dim=2)[0]  # CNN expects [B * Seq_Len, C, H, W]
        char_embs = char_embs.view(
            input_ids.shape[0], input_ids.shape[1], -1
        )  # Reshape back to [B, Seq_Len, Char_Features]

        concat_output = torch.cat([word_embs, char_embs], dim=-1)
        output, _ = self.bilstm(concat_output)
        output = self.att(output, input_ids != 0)
        output, _ = self.bigru(output)
        output, _ = self.lstm(output)
        output = self.fc(output)

        if labels is not None:
            loss = self.loss_fn(output.view(-1, output.shape[-1]), labels.view(-1))
            return loss
        else:
            return torch.argmax(output, dim=-1)


# Model and Optimizer
model = BiLSTMCRF(300, 300, len(word_vocab), len(label_vocab), len(char_vocab), 15)
if torch.cuda.is_available():
    model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

metric = ChunkEvaluator(label_list=label_vocab.keys(), suffix=True)


@torch.no_grad()
def predict(text):
    tokens = text.split()
    lens = len(tokens)
    token_ids = convert_tokens_to_ids(tokens, word_vocab)
    char_ids = [convert_word_to_char_ids(word, char_vocab) for word in tokens]
    token_ids, char_ids = torch.tensor(token_ids).unsqueeze(0), torch.tensor(char_ids).unsqueeze(0)
    if torch.cuda.is_available():
        token_ids, char_ids = token_ids.cuda(), char_ids.cuda()

    model.eval()
    output = model(token_ids, char_ids)

    preds = output.cpu().numpy().tolist()[0]

    reverse_label_map = id2label
    preds = [reverse_label_map[pred] for pred in preds]
    return tokens, preds


entity_counts = {}
for entity_type in entity_types:
    entity_counts[entity_type] = {"TP": 0, "FP": 0, "FN": 0}

relation_counts = {}
for relation_type in relation_types:
    relation_counts[relation_type] = {"TP": 0, "FP": 0, "FN": 0}


@torch.no_grad()
def evaluate_classification(model, file_path):
    model.eval()
    lines = [i.strip("\n") for i in open(file_path, "r", encoding="utf-8").readlines()]
    for line in tqdm(lines):
        tmp_list = line.split("\t")
        sentence = tmp_list[0]
        tag = tmp_list[1]
        sentence_list = sentence.split()
        tag_list = tag.split()

        sent, tags = predict(" ".join(sentence_list))
        tag_list = tag_list[0 : len(tags)]
        for index, tag in enumerate(tag_list):
            if tag == "O" and tags[index] == "O":
                continue
            else:
                if tag == "O":
                    entity_type1 = tags[index].split("_")[1]
                    entity_counts[entity_type1]["FP"] += 1
                elif tags[index] == "O":
                    entity_type2 = tag.split("_")[1]
                    entity_counts[entity_type2]["FN"] += 1
                elif tag.split("_")[1] == tags[index].split("_")[1]:
                    entity_type = tag.split("_")[1]
                    entity_counts[entity_type]["TP"] += 1
                else:
                    entity_type3 = tags[index].split("_")[1]
                    entity_counts[entity_type3]["FN"] += 1

                    entity_type4 = tag.split("_")[1]
                    entity_counts[entity_type4]["FP"] += 1

                if len(tag.split("_")) >= 3 and len(tags[index].split("_")) >= 3:
                    if tag.split("_")[2] == tags[index].split("_")[2]:
                        relation_type = tag.split("_")[2]
                        relation_counts[relation_type]["TP"] += 1
                    else:
                        if tag == "O":
                            relation_type1 = tags[index].split("_")[2]
                            relation_counts[relation_type1]["FP"] += 1
                        elif tags[index] == "O":
                            relation_type2 = tag.split("_")[2]
                            relation_counts[relation_type2]["FN"] += 1
                        else:
                            relation_type3 = tags[index].split("_")[2]
                            relation_counts[relation_type3]["FN"] += 1

                            relation_type4 = tag.split("_")[2]
                            relation_counts[relation_type4]["FP"] += 1
                elif len(tag.split("_")) == 2 and len(tags[index].split("_")) >= 3:
                    relation_type1 = tags[index].split("_")[2]
                    relation_counts[relation_type1]["FP"] += 1
                elif len(tag.split("_")) >= 3 and len(tags[index].split("_")) == 2:
                    relation_type2 = tag.split("_")[2]
                    relation_counts[relation_type2]["FN"] += 1
    evaluation_parameters = {}
    print("-----------------Evaluate entities---------------------")
    for entity_type in entity_counts:
        TP = entity_counts[entity_type]["TP"]
        FP = entity_counts[entity_type]["FP"]
        FN = entity_counts[entity_type]["FN"]
        if TP + FP == 0:
            precision = 0.0
        else:
            precision = TP / (TP + FP)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        if precision + recall == 0:
            F1 = 0.0
        else:
            F1 = 2 * precision * recall / (precision + recall)
        evaluation_parameters[entity_type] = []
        evaluation_parameters[entity_type].extend([precision * 100, recall * 100, F1])
        wandb.log(
            {
                f"entities/{entity_type}_precision": precision * 100,
                f"entities/{entity_type}_recall": recall * 100,
                f"entities/{entity_type}_F1": F1 * 100,
            }
        )
        print(f"{entity_type}: precision={precision * 100:.2f}%, recall={recall * 100:.2f}%, F1={F1 * 100:.2f}%")

    print("-----------------Evaluate relation type---------------------")
    for relation_type in relation_counts:
        TP = relation_counts[relation_type]["TP"]
        FP = relation_counts[relation_type]["FP"]
        FN = relation_counts[relation_type]["FN"]
        if TP + FP == 0:
            precision = 0.0
        else:
            precision = TP / (TP + FP)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        if precision + recall == 0:
            F1 = 0.0
        else:
            F1 = 2 * precision * recall / (precision + recall)
        evaluation_parameters[relation_type] = []
        evaluation_parameters[relation_type].extend([precision * 100, recall * 100, F1])
        wandb.log(
            {
                f"relation/{relation_type}_precision": precision * 100,
                f"relation/{relation_type}_recall": recall * 100,
                f"relation/{relation_type}_F1": F1 * 100,
            }
        )
        print(f"{relation_type}: precision={precision * 100:.2f}%, recall={recall * 100:.2f}%, F1={F1 * 100:.2f}%")
    model.train()


@torch.no_grad()
def evaluate_v1(model, metric, data_loader):
    model.eval()
    metric.reset()
    losses = []
    for input_ids, char_ids, labels, lengths in data_loader:
        if torch.cuda.is_available():
            input_ids, char_ids, labels = input_ids.cuda(), char_ids.cuda(), labels.cuda()
        loss = model(input_ids, char_ids, labels)
        losses.append(loss.cpu().numpy())
        preds = model(input_ids, char_ids, None)
        n_infer, n_label, n_correct = metric.compute(lengths, preds, labels)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
    precision, recall, f1_score = metric.accumulate()
    wandb.log({"val/precision": precision, "val/recall": recall, "val/F1": f1_score})
    print(f"[EVAL] Precision: {precision:.6f} - Recall: {recall:.6f} - F1: {f1_score:.6f}")
    avg_loss = np.mean(losses)
    return avg_loss, precision, recall, f1_score


date = datetime.today().strftime("%Y-%m-%d")
nowname = f"BiLSTM-dynamic-att-LSTM_{date}"
wandb.init(
    name=nowname,
    project="SPZC",
)

# Training Loop
best_f1 = 0
for epoch in range(150):
    for step, (input_ids, char_ids, labels, lengths) in enumerate(train_loader):
        if torch.cuda.is_available():
            input_ids, char_ids, labels = input_ids.cuda(), char_ids.cuda(), labels.cuda()

        model.train()
        optimizer.zero_grad()
        loss = model(input_ids, char_ids, labels)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
            wandb.log({"loss": loss.item()})

    avg_loss, precision, recall, f1 = evaluate_v1(model, metric, dev_loader)
    if f1 > best_f1 and f1 >= 0.4:
        evaluate_classification(model, "data1/dev.txt")
        torch.save(model.state_dict(), f"checkpoints/{nowname}_best.pth")
        best_f1 = f1

# Save Model
torch.save(model.state_dict(), f"checkpoints/{nowname}.pth")
