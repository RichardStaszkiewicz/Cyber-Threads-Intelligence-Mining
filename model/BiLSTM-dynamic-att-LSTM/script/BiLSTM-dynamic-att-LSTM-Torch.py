#!/usr/bin/env python
# coding: utf-8

import torch
import time
import os
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
gernate_dic('data1/dev.txt', 'data1/train.txt', 'data1/tag.dic')
id2label, label2id, label_list = load_dicts('data1/tag.dic')

print(id2label)
print(label2id)
print(label_list)

# Extract entity and relation types
entity_types = set([i.split('_')[1] for i in label_list if i != 'O'])
relation_types = set([i.split('_')[2] for i in label_list if i != 'O' and len(i.split('_')) > 2])

print(entity_types)
print(relation_types)

# Label mapping
label_map = label2id

# Dataset Preparation
def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            words, labels = line.strip().split('\t')
            words = words.split()
            labels = [label_map[label] for label in labels.split()]
            yield {'tokens': words, 'labels': labels}

# Load dataset
train_data = list(read_data('data1/train.txt'))
dev_data = list(read_data('data1/dev.txt'))

# Vocabulary Creation
label_vocab = {label: idx for idx, label in enumerate(label_list)}

word_vocab = {word: idx + 2 for idx, word in enumerate(set([word for item in train_data for word in item['tokens']]))}
word_vocab['PAD'] = 0
word_vocab['OOV'] = 1

char_vocab = {char: idx + 2 for idx, char in enumerate(sorted(set(''.join(word for item in train_data for word in item['tokens']))))}
char_vocab['PAD'] = 0
char_vocab['OOV'] = 1

# Token and Character Conversion
max_char_len = 15  # Set based on data

def convert_tokens_to_ids(tokens, vocab):
    return [vocab.get(token, vocab['OOV']) for token in tokens]

def convert_word_to_char_ids(word, char_vocab):
    char_ids = [char_vocab.get(char, char_vocab['OOV']) for char in word[:max_char_len]]
    char_ids += [char_vocab['PAD']] * (max_char_len - len(char_ids))
    return char_ids

# Dataset Class
class NERDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        labels = item['labels']

        token_ids = convert_tokens_to_ids(tokens, word_vocab)
        char_ids = [convert_word_to_char_ids(word, char_vocab) for word in tokens]

        return torch.tensor(token_ids), torch.tensor(char_ids), torch.tensor(labels)

# Collate Function
def collate_fn(batch):
    token_ids, char_ids, labels = zip(*batch)
    
    token_ids = nn.utils.rnn.pad_sequence(token_ids, batch_first=True, padding_value=word_vocab['PAD'])
    char_ids = nn.utils.rnn.pad_sequence([torch.tensor(c) for c in char_ids], batch_first=True, padding_value=char_vocab['PAD'])
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=label_vocab['O'])
    
    return token_ids, char_ids, labels

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
        self.scale = hidden_size ** 0.5

    def forward(self, inputs, mask):
        queries = self.query(inputs)
        keys = self.key(inputs)
        values = self.value(inputs)

        attn_weights = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
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
        self.loss_fn = self.custom_loss_fn #nn.CrossEntropyLoss(reduction='none')
    
    def custom_loss_fn(self, logits, labels):
      standard_loss = F.cross_entropy(logits, labels, reduction='none')  # Compute standard CrossEntropyLoss
      is_o_label = (labels == self.o_label_index).float()  # Identify 'O' labels
      adjusted_loss = (1 - is_o_label) * self.alpha * standard_loss + is_o_label * standard_loss  # Adjust loss
      return adjusted_loss.sum()

    def forward(self, input_ids, char_input_ids, labels=None):
        word_embs = self.word_emb(input_ids)
        
        char_embs = self.char_emb(char_input_ids)  # Shape [B, Seq_Len, Max_Char_Len, Char_Emb_Dim]
        char_embs = char_embs.view(-1, 1, max_char_len, char_embs.shape[-1])  # Reshape to [B * Seq_Len, 1, Max_Char_Len, Char_Emb_Dim]
        char_embs = self.cnn(char_embs).squeeze(2).max(dim=2)[0]  # CNN expects [B * Seq_Len, C, H, W]
        char_embs = char_embs.view(input_ids.shape[0], input_ids.shape[1], -1)  # Reshape back to [B, Seq_Len, Char_Features]

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

# Training Loop
for epoch in range(150):
    for step, (input_ids, char_ids, labels) in enumerate(train_loader):
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

# Save Model
torch.save(model.state_dict(), 'model/bilstm_crf.pth')
