import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import pandas as pd
import numpy as np
import copy
import unicodedata
import csv
from functools import partial


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, head=None, label=None):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.head = head
        self.label = label



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, head_idx, label_id, valid_ids=None,
                 label_mask=None, eval_mask=None,
                 ngram_ids=None, ngram_positions=None, ngram_lengths=None,
                 ngram_tuples=None, ngram_seg_ids=None, ngram_masks=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.head_idx = head_idx
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.eval_mask = eval_mask

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks


def ispunct(token):
  return all(unicodedata.category(char).startswith('P') for char in token)


class MyDataset(IterableDataset):


    @staticmethod
    def process_data(lines):

        examples = []
        for i, (sentence, head, label) in enumerate(lines):
            guid = "%s" % str(i)
            examples.append(InputExample(guid=guid, text_a=sentence, text_b=None, head=head,
                                         label=label))
        return examples

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
            if ispunct(word):
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

        eval_mask = copy.deepcopy(label_mask)
        eval_mask[0] = 0
        # ignore all punctuation if not specified
        for idx in punctuation_idx:
            if idx < len(eval_mask):
                eval_mask[idx] = 0


        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        head_idx = torch.tensor(head_idx, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        valid_ids = torch.tensor(valid, dtype=torch.long)
        lmask_ids = torch.tensor(label_mask, dtype=torch.bool)
        eval_mask_ids = torch.tensor(eval_mask, dtype=torch.bool)


        return [input_ids, input_mask, segment_ids, head_idx, label_ids, valid_ids, lmask_ids, eval_mask_ids]

    def convert_examples_to_features(self, examples):

          max_seq_length = self.max_seq_length
          labelmap = self.label_map
          tokenizer = self.tokenizer

          features = []

          length_list = []
          tokens_list = []
          head_idx_list = []
          labels_list = []
          valid_list = []
          label_mask_list = []
          punctuation_idx_list = []

          for (ex_index, example) in enumerate(examples):
              textlist = example.text_a
              labellist = example.label
              head_list = example.head
              tokens = []
              head_idx = []
              labels = []
              valid = []
              label_mask = []

              punctuation_idx = []

              if len(textlist) > max_seq_length - 2:
                  textlist = textlist[:max_seq_length - 2]
                  labellist = labellist[:max_seq_length - 2]
                  head_list = head_list[:max_seq_length - 2]

              for i, word in enumerate(textlist):
                  if ispunct(word):
                      punctuation_idx.append(i+1)
                  token = tokenizer.tokenize(word)
                  tokens.extend(token)
                  label_1 = labellist[i]
                  for m in range(len(token)):
                      if m == 0:
                          valid.append(1)
                          head_idx.append(head_list[i])
                          labels.append(label_1)
                          label_mask.append(1)
                      else:
                          valid.append(0)
              if len(tokens) > 300:
                  continue
              length_list.append(len(tokens))
              tokens_list.append(tokens)
              head_idx_list.append(head_idx)
              labels_list.append(labels)
              valid_list.append(valid)
              label_mask_list.append(label_mask)
              punctuation_idx_list.append(punctuation_idx)

          label_len_list = [len(label) for label in labels_list]
          if len(length_list) == 0:
            print(examples)

          seq_pad_length = max(length_list) + 2
          label_pad_length = max(label_len_list) + 1


          for example, tokens, head_idxs, labels, valid, label_mask, punctuation_idx in \
                  zip(examples,
                      tokens_list, head_idx_list, labels_list, valid_list, label_mask_list, punctuation_idx_list):

              ntokens = []
              segment_ids = []
              label_ids = []
              head_idx = []

              ntokens.append(tokenizer.cls_token)
              segment_ids.append(0)

              valid.insert(0, 1)
              label_mask.insert(0, 1)
              head_idx.append(-1)
              label_ids.append(labelmap[tokenizer.cls_token])
              for i, token in enumerate(tokens):
                  ntokens.append(token)
                  segment_ids.append(0)
              for i in range(len(labels)):
                  if labels[i] in labelmap:
                      label_ids.append(labelmap[labels[i]])
                  else:
                      label_ids.append(labelmap[tokenizer.unk_token])
                  head_idx.append(head_idxs[i])
              ntokens.append(tokenizer.sep_token)

              segment_ids.append(0)
              valid.append(1)

              input_ids = tokenizer.convert_tokens_to_ids(ntokens)
              input_mask = [1] * len(input_ids)
              while len(input_ids) < seq_pad_length:
                  input_ids.append(0)
                  input_mask.append(0)
                  segment_ids.append(0)
                  valid.append(1)
              while len(label_ids) < label_pad_length:
                  head_idx.append(-1)
                  label_ids.append(0)
                  label_mask.append(0)

              eval_mask = copy.deepcopy(label_mask)
              eval_mask[0] = 0
              # ignore all punctuation if not specified
              for idx in punctuation_idx:
                  if idx < label_pad_length:
                      eval_mask[idx] = 0

              input_ids = torch.tensor(input_ids, dtype=torch.long)
              input_mask = torch.tensor(input_mask, dtype=torch.long)
              segment_ids = torch.tensor(segment_ids, dtype=torch.long)
              head_idx = torch.tensor(head_idx, dtype=torch.long)
              label_ids = torch.tensor(label_ids, dtype=torch.long)
              valid_ids = torch.tensor(valid, dtype=torch.long)
              lmask_ids = torch.tensor(label_mask, dtype=torch.bool)
              eval_mask_ids = torch.tensor(eval_mask, dtype=torch.bool)
              features.append([input_ids,input_mask,segment_ids,head_idx,label_ids,valid_ids,lmask_ids,eval_mask_ids])

              # assert len(input_ids) == seq_pad_length
              # assert len(input_mask) == seq_pad_length
              # assert len(segment_ids) == seq_pad_length
              # assert len(valid) == seq_pad_length

              # assert len(label_ids) == label_pad_length
              # assert len(head_idx) == label_pad_length
              # assert len(label_mask) == label_pad_length
              # assert len(eval_mask) == label_pad_length

              ngram_ids = None
              ngram_positions_matrix = None
              ngram_lengths = None
              ngram_tuples = None
              ngram_seg_ids = None
              ngram_mask_array = None

              # features.append(
              #     InputFeatures(input_ids=input_ids,
              #                   input_mask=input_mask,
              #                   segment_ids=segment_ids,
              #                   head_idx=head_idx,
              #                   label_id=label_ids,
              #                   valid_ids=valid,
              #                   label_mask=label_mask,
              #                   eval_mask=eval_mask,
              #                   ngram_ids=ngram_ids,
              #                   ngram_positions=ngram_positions_matrix,
              #                   ngram_lengths=ngram_lengths,
              #                   ngram_tuples=ngram_tuples,
              #                   ngram_seg_ids=ngram_seg_ids,
              #                   ngram_masks=ngram_mask_array,
              #                   ))


              input_ids = torch.tensor(input_ids, dtype=torch.long)
              input_mask = torch.tensor(input_mask, dtype=torch.long)
              segment_ids = torch.tensor(segment_ids, dtype=torch.long)
              head_idx = torch.tensor(head_idx, dtype=torch.long)
              label_ids = torch.tensor(label_ids, dtype=torch.long)
              valid_ids = torch.tensor(valid, dtype=torch.long)
              lmask_ids = torch.tensor(label_mask, dtype=torch.bool)
              eval_mask_ids = torch.tensor(eval_mask, dtype=torch.bool)
              features.append([input_ids,input_mask,segment_ids,head_idx,label_ids,valid_ids,lmask_ids,eval_mask_ids])
          return features


    def get_label_list(self, tokenizer, label_path):
      label_list = [tokenizer.unk_token]

      with open(label_path, 'r', encoding='utf8') as f:
          lines = f.readlines()
          for line in lines:
              line = line.strip()
              if line == '':
                  continue
              label_list.append(line)

      assert 'amod' in label_list

      label_list.extend([tokenizer.cls_token, tokenizer.sep_token])
      return label_list




    def Get_data(self, df):
        sentence = []
        head = []
        label = []
        for i in range(len(df)):
          sentence.append(str(df.values[i,0]))
          head.append(int(df.values[i,1]))
          label.append(df.values[i,2])
        return (sentence,head,label)



    def iter_chunk(self,file_path):
      # create iterator by sentences
      chunk_data = pd.read_csv(file_path,
                              skip_blank_lines=False,
                              iterator=True,
                              chunksize=1,
                              delimiter='\t',
                              header=None,
                              encoding='utf-8',
                              quoting=csv.QUOTE_NONE)
      sentences = []
      heads = []
      labels = []
      first_chunk = chunk_data.get_chunk()
      chunk = pd.DataFrame(first_chunk)
      sentences.append(str(chunk.values[0,0]))
      heads.append(int(chunk.values[0,1]))
      labels.append(chunk.values[0,2])
      for l in chunk_data:
          #if not np.isnan(l.iloc[0,0]):
          if not pd.isnull(l.iloc[0,0]):
              sentences.append(str(l.values[0, 0]))
              heads.append(int(l.values[0, 1]))
              labels.append(l.values[0, 2])
              continue
          #splitted_data = (sentences,heads,labels)
          #processed_data = self.process_data(lines = [splitted_data])
          example = self.custom_convert_examples_to_features(text_list= sentences, label_list=labels, head_list= heads)
          yield example
          first_chunk = next(chunk_data)
          chunk = pd.DataFrame(first_chunk)
          sentences = []
          heads = []
          labels = []
          sentences.append(str(chunk.values[0, 0]))
          heads.append(int(chunk.values[0, 1]))
          labels.append(chunk.values[0, 2])
          continue
      #splitted_data = (sentences,heads,labels)
      #processed_data = self.process_data(lines = [splitted_data])
      #example = self.convert_examples_to_features(examples = processed_data)
      #yield example[0]
      example = self.custom_convert_examples_to_features(text_list=sentences, label_list=labels, head_list=heads)
      yield example

    def __init__(self, file_path, tokenizer, label_path, max_seq_length = 256):
        super(IterableDataset).__init__()
        self.file_path = file_path
        self.iterable = self.iter_chunk(self.file_path)
        self.tokenizer = tokenizer
        self.label_list = self.get_label_list(tokenizer,label_path)
        self.label_map = {label: i for i, label in enumerate(self.label_list, 1)}
        self.max_seq_length = max_seq_length

    def __iter__(self):
        return iter(self.iterable)

from torch.nn.functional import pad
def custom_collate(data, tokenizer):
    # 0:  input_ids
    # 1:  input_mask
    # 2:  segment_ids
    # 3:  head_idx
    # 4:  labels_ids
    # 5:  valid_ids
    # 6:  lmask_ids
    # 7:  eval_mask_ids
    seq_pad_idx = [0,1,2,5]
    label_pad_idx = [3,4,6,7]
    add_token = [tokenizer.pad_token_id,0,0,-1,0,1,0,0]
    seq_pad_len = 0
    label_pad_len = 0

    for item in data:
        for col in range(8):
            if col in seq_pad_idx:
                seq_pad_len = max(seq_pad_len, len(item[col]))
            else:
                label_pad_len = max(label_pad_len, len(item[col]))

    # for col in seq_pad_idx:
    #     seq_pad_len = max(seq_pad_len,max([len(item[col]) for item in data]))
    # for col in label_pad_idx:
    #     label_pad_len = max(label_pad_len,max([len(item[col]) for item in data]))

    for item in data:
        for col in range(8):
          if col in seq_pad_idx:
            need = seq_pad_len - len(item[col])
          else:
            need = label_pad_len - len(item[col])
          if need == 0:
            continue
          item[col] = pad(item[col],(0,need),value = add_token[col])
          #add = torch.full([need],add_token[col])
          #item[col] = torch.cat((item[col],add))

    return data

