import os
import time
import logging
from tqdm import tqdm
from functools import partial
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from TorchCRF import CRF
import numpy as np

from utils import *
from ChunkEvaluatorTorch import ChunkEvaluator

gernate_dic('data1/dev.txt', 'data1/train.txt', 'data1/tag.dic')
id2label, label2id, label_list = load_dicts('data1/tag.dic')
id2label[183] = 'no_entity'
label2id['no_entity'] = 183
label_list = [id2label[i] for i in sorted(id2label.keys())]

entity_types = set(i.split('_')[1] for i in label_list if i != 'O')
relation_types = set(i.split('_')[2] for i in label_list if i != 'O' and len(i.split('_')) > 2)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
max_seq_len = 128
batch_size = 16
no_entity_id = 183
ignore_label = no_entity_id
relation_counts = {}
for relation_type in relation_types:
    relation_counts[relation_type] = {"TP": 0, "FP": 0, "FN": 0}
entity_counts = {}
for entity_type in entity_types:
    entity_counts[entity_type] = {"TP": 0, "FP": 0, "FN": 0}

def evaluate_classification(model, file_path):
    model.eval()
    lines = [i.strip('\n') for i in open(file_path, 'r', encoding='utf-8').readlines()]
    for line in tqdm(lines):
        tmp_list = line.split('\t')
        sentence = tmp_list[0]
        tag = tmp_list[1]
        sentence_list = sentence.split()
        tag_list = tag.split()

        # 第二步，取出模型预测的y
        sent, tags = predict(' '.join(sentence_list))
        tag_list = tag_list[0:len(tags)]

        # 遍历，二者进行比较
        for index, tag in enumerate(tag_list):
            #print(f"Original Tag: {tag}")  # 打印原始标签

            true_entity, true_relation = get_entity_and_relation(tag)
            #print(f"True Entity: {true_entity}, True Relation: {true_relation}")  # 打印解析后的实体和关系

            predicted_entity, predicted_relation = get_entity_and_relation(tags[index])
            #print(f"Predicted Entity: {predicted_entity}, Predicted Relation: {predicted_relation}")  # 打印预测的实体和关系

            if true_entity is None or predicted_entity is None:
                continue
            true_entity_type = true_entity.split('_')[-1] if true_entity != 'O' else ''
            predicted_entity_type = predicted_entity.split('_')[-1] if predicted_entity != 'O' else ''


            # 确保键存在
            if true_entity_type not in entity_counts and true_entity != 'O':
                entity_counts[true_entity_type] = {"TP": 0, "FP": 0, "FN": 0}
            if predicted_entity_type not in entity_counts and predicted_entity != 'O':
                entity_counts[predicted_entity_type] = {"TP": 0, "FP": 0, "FN": 0}

            if true_entity != 'O':  # 如果有真实实体
                if true_entity == predicted_entity:
                    entity_counts[true_entity_type]['TP'] += 1  # 真正例
                else:
                    entity_counts[true_entity_type]['FN'] += 1  # 假负例：未能正确识别真实实体
            else:
                if predicted_entity != 'O':
                    entity_counts[predicted_entity_type]['FP'] += 1  # 假正例：错误识别或预测了不存在的实体

            # 确保关系类型的键也存在
            if true_relation not in relation_counts and true_relation != '':
                relation_counts[true_relation] = {"TP": 0, "FP": 0, "FN": 0}
            if predicted_relation not in relation_counts and predicted_relation != '':
                relation_counts[predicted_relation] = {"TP": 0, "FP": 0, "FN": 0}

            # 关系评估
            if true_relation != '':
                if true_relation == predicted_relation:
                    relation_counts[true_relation]['TP'] += 1  # 真正例：正确预测关系类型
                else:
                    relation_counts[true_relation]['FN'] += 1
            else:
                if predicted_relation != '':
                    relation_counts[predicted_relation]['FP'] += 1
                    
def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            words, labels = line.strip().split('\t')
            yield {'tokens': words.split(), 'labels': [label2id[l] for l in labels.split()]}

class NERDataset(Dataset):
    def __init__(self, examples):
        self.examples = [self.convert(e) for e in examples]

    def convert(self, ex):
        encoding = tokenizer(
            ex['tokens'], 
            is_split_into_words=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        token_type_ids = encoding['token_type_ids'].squeeze(0)
        labels = [no_entity_id] + ex['labels'][:max_seq_len-2] + [no_entity_id]
        labels += [no_entity_id] * (max_seq_len - len(labels))
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'labels': torch.tensor(labels),
            'seq_len': len(labels)
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    return (
        pad_sequence([x['input_ids'] for x in batch], batch_first=True),
        pad_sequence([x['token_type_ids'] for x in batch], batch_first=True),
        torch.tensor([x['seq_len'] for x in batch]),
        pad_sequence([x['labels'] for x in batch], batch_first=True, padding_value=ignore_label)
    )

train_data = list(read_data('data1/train.txt'))
dev_data = list(read_data('data1/dev.txt'))
train_loader = DataLoader(NERDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(NERDataset(dev_data), batch_size=batch_size, collate_fn=collate_fn)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.wq = nn.Linear(hidden_size, hidden_size)
        self.wk = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** 0.5

    def forward(self, x):
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)
        attn_weights = torch.softmax(Q @ K.transpose(-2, -1) / self.scale, dim=-1)
        return attn_weights @ V

class BertBiLstmAttCRF(nn.Module):
    def __init__(self, label_num, lstm_hidden=800, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.lstm = nn.LSTM(self.bert.config.hidden_size, lstm_hidden, num_layers=1, 
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.att = AttentionLayer(lstm_hidden * 2)
        self.fc = nn.Linear(lstm_hidden * 2, label_num)
        self.crf = CRF(num_labels=label_num)#, batch_first=True)

    def forward(self, input_ids, token_type_ids, lengths=None, labels=None):
        mask = input_ids != tokenizer.pad_token_id
        bert_out = self.bert(input_ids, token_type_ids=token_type_ids).last_hidden_state
        lstm_out, _ = self.lstm(bert_out)
        attn_out = self.att(lstm_out)
        emissions = self.fc(attn_out)

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask)
            return loss
        else:
            preds = self.crf.viterbi_decode(emissions, mask=mask)
            return preds

# ========== Optimizer and Scheduler ==========
#BertModel.from_pretrained('bert-base-cased'),
model = BertBiLstmAttCRF(len(label_list)).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8, weight_decay=1e-4)

metric = ChunkEvaluator(label_list=label_list, suffix=False)

# ========== Evaluation Logic ==========

def predict(text):
    model.eval()
    tokens = text.split()
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = enc['input_ids'].cuda()
    token_type_ids = enc['token_type_ids'].cuda()
    preds = model(input_ids, token_type_ids)
    pred_labels = preds[0]
    tags = [id2label[x] for x in pred_labels[1:-1]]  # remove [CLS] and [SEP]
    return tokens, tags

@torch.no_grad()
def evaluate(model, metric, dataloader):
    model.eval()
    metric.reset()
    for batch in dataloader:
        input_ids, token_type_ids, lengths, labels = [x.cuda() for x in batch]
        preds = model(input_ids, token_type_ids)
        preds_padded = torch.full_like(labels, fill_value=no_entity_id)
        for i, p in enumerate(preds):
            preds_padded[i, :len(p)] = torch.tensor(p, device=labels.device)
        num_infer, num_label, num_correct = metric.compute(lengths, preds_padded, labels)
        metric.update(num_infer.numpy(), num_label.numpy(), num_correct.numpy())
    p, r, f1 = metric.accumulate()
    print(f"[EVAL] Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
    return f1

# ========== Training Loop ==========
num_train_epochs = 85
logging_steps = 50
best_f1 = 0
global_step = 0
tic_train = time.time()
output_dir = 'model'
os.makedirs(output_dir, exist_ok=True)

for epoch in range(num_train_epochs):
    model.train()
    for step, batch in enumerate(train_loader):
        global_step += 1
        input_ids, token_type_ids, lengths, labels = [x.cuda() for x in batch]
        loss = model(input_ids, token_type_ids, lengths=lengths, labels=labels).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if global_step % logging_steps == 0:
            print(f"global step {global_step}, epoch: {epoch}, batch: {step}, loss: {loss.item():.4f}, speed: {logging_steps / (time.time() - tic_train):.2f} step/s")
            tic_train = time.time()

    print("---- Evaluating on dev set ----")
    f1 = evaluate(model, metric, dev_loader)

    if f1 > best_f1:
        print("Saving best model...")
        torch.save(model.state_dict(), os.path.join(output_dir, "bert_bilstm_att_crf.pt"))
        best_f1 = f1

        if f1 >= 0.5:
            evaluate_classification(model, 'data1/dev.txt')

tokenizer.save_pretrained(output_dir)