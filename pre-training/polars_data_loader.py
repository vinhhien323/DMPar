import torch, polars
import numpy as np
from torch.utils.data import Dataset, DataLoader
import unicodedata
import csv
import copy
from functools import partial
import time

class MyDataset(Dataset):

    def ispunct(self,token):
      return all(unicodedata.category(char).startswith('P') for char in token)

    def custom_convert_examples_to_features(self, text_list, label_list, head_list):
        tokens = []
        head_idx = []
        labels = []
        valid = []
        label_mask = []

        punctuation_idx = []

        if len(text_list) > self.max_seq_length - 2:
            text_list = text_list[:self.max_seq_length - 2]
            label_list = label_list[:self.max_seq_length - 2]
            head_list = head_list[:self.max_seq_length - 2]

        for i, word in enumerate(text_list):
            if self.ispunct(token = word):
                punctuation_idx.append(i + 1)
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for m in range(len(token)):
                if m == 0:
                    valid.append(1)
                    head_idx.append(head_list[i])
                    labels.append(label_list[i])
                    label_mask.append(1)
                else:
                    valid.append(0)


        ntokens = []
        segment_ids = []
        label_ids = []
        head_idx = []

        ntokens.append(self.tokenizer.cls_token)
        segment_ids.append(0)

        valid.insert(0, 1)
        label_mask.insert(0, 1)
        head_idx.append(-1)
        label_ids.append(self.label_map[self.tokenizer.cls_token])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
        for i in range(len(labels)):
            if labels[i] in self.label_map:
                label_ids.append(self.label_map[labels[i]])
            else:
                label_ids.append(self.label_map[self.tokenizer.unk_token])
            head_idx.append(head_idx[i])
        ntokens.append(self.tokenizer.sep_token)

        segment_ids.append(0)
        valid.append(1)

        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        head_idx = torch.tensor(head_idx, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        valid_ids = torch.tensor(valid, dtype=torch.long)
        lmask_ids = torch.tensor(label_mask, dtype=torch.bool)

        return [input_ids, input_mask, segment_ids, head_idx, label_ids, valid_ids, lmask_ids]

    def get_label_list(self, tokenizer, label_path):
      #label_list = [tokenizer.unk_token]
      label_list = [tokenizer.cls_token, tokenizer.pad_token, tokenizer.sep_token, tokenizer.unk_token]
      with open(label_path, 'r', encoding='utf8') as f:
          lines = f.readlines()
          for line in lines:
              line = line.strip()
              if line == '':
                  continue
              label_list.append(line)

      assert 'amod' in label_list

      #label_list.extend([tokenizer.cls_token, tokenizer.sep_token])
      return label_list

    def __init__(self, file_path, tokenizer, label_path, max_seq_length = 256):
        self.file_path = file_path
        self.data = polars.read_csv(source=file_path, has_header = False, separator = '\t', quote_char = None)
        self.tokenizer = tokenizer
        self.label_list = self.get_label_list(tokenizer,label_path)
        self.label_map = {label: i for i, label in enumerate(self.label_list, 1)}
        self.max_seq_length = max_seq_length

    def __len__(self):
        row, column = self.data.shape
        return row

    def __getitem__(self, idx):
        sentences = self.data[idx,0].split()
        heads = [int(i) for i in self.data[idx,1].split()]
        labels = self.data[idx,2].split()
        return self.custom_convert_examples_to_features(text_list= sentences, label_list=labels, head_list= heads)


from torch.nn.functional import pad


def custom_collate(data, tokenizer):
    np.random.shuffle(data)

    # 0:  input_ids
    # 1:  input_mask
    # 2:  segment_ids
    # 3:  head_idx
    # 4:  labels_ids
    # 5:  valid_ids
    # 6:  lmask_ids
    # 7:  eval_mask_ids

    seq_pad_idx = [0, 1, 2, 5]
    label_pad_idx = [3, 4, 6]
    add_token = [tokenizer.pad_token_id, 0, 0, -1, 0, 1, 0, 0]
    seq_pad_len = 0
    label_pad_len = 0

    for col in seq_pad_idx:
        seq_pad_len = max(seq_pad_len, max([len(item[col]) for item in data]))
    for col in label_pad_idx:
        label_pad_len = max(label_pad_len, max([len(item[col]) for item in data]))

    # time_x = time.time()
    # for item in data:
    #     for col in range(8):
    #       if col in seq_pad_idx:
    #         need = seq_pad_len - len(item[col])
    #       else:
    #         need = label_pad_len - len(item[col])
    #       if need == 0:
    #         continue
    #       item[col] = pad(item[col],(0,need),value = add_token[col])
    #       #add = torch.full([need],add_token[col])
    #       #item[col] = torch.cat((item[col],add))
    # time_y = time.time()
    # print('Padding costs:',time_y-time_x)
    # np.random.shuffle(data)

    all_input_ids = torch.stack([pad(item[0], (0, seq_pad_len - len(item[0])), value=add_token[0]) for item in data])
    all_input_mask = torch.stack([pad(item[1], (0, seq_pad_len - len(item[1])), value=add_token[1]) for item in data])
    all_segment_ids = torch.stack([pad(item[2], (0, seq_pad_len - len(item[2])), value=add_token[2]) for item in data])
    all_head_idx = torch.stack([pad(item[3], (0, label_pad_len - len(item[3])), value=add_token[3]) for item in data])
    all_label_ids = torch.stack([pad(item[4], (0, label_pad_len - len(item[4])), value=add_token[4]) for item in data])
    all_valid_ids = torch.stack([pad(item[5], (0, seq_pad_len - len(item[5])), value=add_token[5]) for item in data])
    all_lmask_ids = torch.stack([pad(item[6], (0, label_pad_len - len(item[6])), value=add_token[6]) for item in data])

    ngram_ids = None
    ngram_positions = None

    return all_input_ids, all_input_mask, all_lmask_ids, all_head_idx, all_label_ids, ngram_ids, ngram_positions, all_segment_ids, all_valid_ids

