import numpy as np
import scipy.stats as st
import random
import pandas as pd
from tqdm.notebook import tqdm
import xlrd
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import torch
from torch import nn
from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModel
from transformers import PreTrainedTokenizerFast, RobertaModel
import io
import base64
import os


class BertRegression(torch.nn.Module):
    def __init__(self, model_name, model_initalization, metric_names, dropout=0.2, hidden_dim=768, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        Initialize the model.
        :param model_name: the name of the model
        :param model_initalization: the initialization of the model (e.g. AutoModel.from_pretrained('bert-base-uncased'))
        :param metric_names: the names of the metrics to use
        :param dropout: the dropout rate
        :param hidden_dim: the hidden dimension of the model
        """
        super(BertRegression, self).__init__()

        self.metric_names = metric_names
        self.model_name = model_name
        self.model_initalization = model_initalization

        for name, initalization in zip(self.model_name, self.model_initalization):
            setattr(self, name, initalization)

        for name in self.metric_names:
            setattr(self, name, nn.Linear(hidden_dim, 1))
            setattr(self, 'l_1_' + name, nn.Linear(hidden_dim, hidden_dim))

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.device = device


    def tokenize(self, tokenizer, words, silent = True):
        if isinstance(words,list):
          class Tokens(torch.utils.data.Dataset):
              def __init__(self, words):
                  if type(tokenizer) == list:
                      self.texts1 = [tokenizer[0](str(text),
                                                padding='max_length', max_length = 10, truncation=True,
                                                  return_tensors="pt") for text in words]
                      self.texts2 = [tokenizer[1](str(text),
                                                padding='max_length', max_length = 10, truncation=True,
                                                  return_tensors="pt") for text in words]
                  else:
                      self.texts = [tokenizer(str(text),
                                            padding='max_length', max_length = 10, truncation=True,
                                            return_tensors="pt") for text in words]
                  self.words = words

              def __len__(self):
                if type(tokenizer) == list:
                  return len(self.texts2)
                else:
                  return len(self.texts)

              def get_batch_texts(self, idx):
                  # Fetch a batch of inputs
                  if type(tokenizer) == list:
                      return [self.texts1[idx], self.texts2[idx]], self.words[idx]
                  else:
                      return self.texts[idx], self.words[idx]

              def __getitem__(self, idx):
                  return self.get_batch_texts(idx)


          outputs = []
          loop_wrap = lambda x:x if silent else tqdm
          for word_tok, words in loop_wrap(torch.utils.data.DataLoader(Tokens(words), batch_size=256)):
              if type(word_tok)!=list: word_tok = [word_tok]
              output_list = []
              for idx, name in enumerate(self.model_name):
                  mask = word_tok[idx]['attention_mask'].to(self.device)
                  input_id = word_tok[idx]['input_ids'].squeeze(1).to(self.device)
                  _,x = getattr(self, self.model_name[idx])(input_id, mask, return_dict=False)
                  output_list.append(x.detach().cpu().numpy())

              x = [np.hstack(emb) for emb in zip(*output_list)]

              for word, embed in zip(words, x):
                  outputs.append([word, embed])
              del mask, input_id,  x, _
          o = np.array(outputs, dtype=object)
          df = pd.DataFrame({"word": o[:, 0], "embed": o[:, 1]})
        else:
          word_tok = tokenizer(words,
                              padding='max_length', max_length=10, truncation=True,
                              return_tensors="pt")
          mask = word_tok['attention_mask'].to(self.device)
          input_id = word_tok['input_ids'].squeeze(1).to(self.device)
          _, x = getattr(self, self.model_name[0])(input_id, mask, return_dict=False)

          df = pd.DataFrame({"word":[words],"embed":[x.detach().cpu().numpy()[0,:]]})

        return df


    def forward(self, *args):
        """
        Forward pass of the model.
        :param args: the inputs
        :return: the outputs
        """
        output_list = []
        for idx, name in enumerate(self.model_name):
            _, x = getattr(self, name)(args[idx*2], args[idx*2 + 1], return_dict=False)
            output_list.append(x)

        x = torch.cat(output_list, dim=1)

        output = self.rate_embedding(x)
        return output

    def rate_embedding(self, x):
        output_ratings = []
        for name in self.metric_names:
            first_layer =  self.relu(self.dropout(self.layer_norm(getattr(self, 'l_1_' + name)(x) + x)))
            second_layer = self.sigmoid(getattr(self, name)(first_layer))
            output_ratings.append(second_layer)

        return output_ratings




class Tokens2(torch.utils.data.Dataset):
    def __init__(self,df):
        self.embeds = df.embed
        self.words = df.word

    def __len__(self): return len(self.words)

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.embeds[idx], self.words[idx]

    def __getitem__(self, idx):
        batch_texts, words = self.get_batch_texts(idx)
        return batch_texts, words


def get_valence_arousal_dominance(words):
    tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-en")
    metric_names = ['valence', 'arousal', 'dominance', 'aoa', 'concreteness']
    model = BertRegression(["bert"], [AutoModel.from_pretrained("nghuyong/ernie-2.0-en")],
                             metric_names)
    model.load_state_dict(torch.load(r'D:\GitHub\bias_free_modeling\models/english_one_nghuyong'), strict=False)

    model.cuda()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # to device
    model.to(device)
    df = model.tokenize(tokenizer, words)
    new_columns = dict(zip(model.metric_names, [[] for m in model.metric_names]))
    with torch.no_grad():
        for i, (embed, word) in (enumerate(torch.utils.data.DataLoader(Tokens2(df), batch_size=256))):
            predicts = (model.rate_embedding(embed.to(device)))
            for m, predict in enumerate(predicts):
                for p in (predict):
                    new_columns[model.metric_names[m]].append(p.detach().cpu().item())

    del model
    torch.cuda.empty_cache()

    return new_columns['valence'], new_columns['arousal'], new_columns['dominance']
