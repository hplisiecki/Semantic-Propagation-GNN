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

    def __init__(self, model_name, model_initialization, emotion, dropout=0.2, hidden_dim=768):
        """
        Initialize the model.
        :param model_name: the name of the model
        :param model_initalization: the initialization of the model (e.g. AutoModel.from_pretrained('bert-base-uncased'))
        :param metric_names: the names of the metrics to use
        :param dropout: the dropout rate
        :param hidden_dim: the hidden dimension of the model
        """
        super(BertRegression, self).__init__()

        self.emotion = emotion
        self.model_name = model_name
        self.model_initialization = model_initialization

        for name, initialization in zip(self.model_name, self.model_initialization):
            setattr(self, name, initialization)

        setattr(self, self.emotion, nn.Linear(hidden_dim, 1))
        setattr(self, 'l_1_' + self.emotion, nn.Linear(hidden_dim, hidden_dim))


        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, *args):
        """
        Forward pass of the model.
        :param args: the inputs
        :return: the outputs
        """
        output_list = []
        for idx, name in enumerate(self.model_name):
            _, x = getattr(self, name)(args[idx * 2], args[idx * 2 + 1], return_dict=False)
            output_list.append(x)

        x = torch.cat(output_list, dim=1)

        output = self.rate_embedding(x)
        return output

    def rate_embedding(self, x):

        first_layer = self.relu(self.dropout(self.layer_norm(getattr(self, 'l_1_' + self.emotion)(x) + x)))
        outputs = self.sigmoid(getattr(self, self.emotion)(first_layer))

        return outputs

class Dataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, words, max_len):
        """
        Initialize the dataset.
        :param tokenizer:  the tokenizer to use
        :param df: the dataframe to use
        :param max_len:  the maximum length of the sequences
        :param metric_names: the names of the metrics to use
        """
        self.tokenizer = tokenizer


        # check if tokenizer is a list
        if type(self.tokenizer) == list:
            self.texts1 = [self.tokenizer[0](str(text),
                                             padding='max_length', max_length=max_len[0], truncation=True,
                                             return_tensors="pt") for text in df['word']]
            self.texts2 = [self.tokenizer[1](str(text),
                                             padding='max_length', max_length=max_len[1], truncation=True,
                                             return_tensors="pt") for text in df['word']]
        else:
            self.texts = [self.tokenizer(str(text),
                                         padding='max_length', max_length=max_len, truncation=True,
                                         return_tensors="pt") for text in words]

    def get_batch_texts(self, idx):
        """
        Return the texts of the batch.
        :param idx: the index of the batch
        :return: the texts
        """
        if type(self.tokenizer) == list:
            return self.texts1[idx], self.texts2[idx]
        else:
            return self.texts[idx]

    def __len__(self):
        """
        Return the length of the dataset.
        :return: the length
        """
        setattr(self, 'length', len(self.texts))
        return self.length

    def __getitem__(self, idx):
        """
        Return the item at the given index.
        :param idx: the index
        :return: the item
        """
        batch_texts = self.get_batch_texts(idx)
        return batch_texts

def get_discrete_emotions(words):
    """
    Get the discrete emotions.
    :return: the discrete emotions
    """
    emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    max_lens = [8, 7, 8, 8, 8, 8, 7, 7]
    hidden_dim = 768
    dropout = 0.1

    model_dir = "nghuyong/ernie-2.0-en"

    model_name = ['bert']

    model_initialization = [AutoModel.from_pretrained(model_dir)]

    batch_size = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    word_emo_dict = {emo: {} for emo in emotions}
    for emotion, max_len in zip(emotions, max_lens):
        save_dir = fr'D:\GitHub\bias_free_modeling\models/{emotion}'
        model = BertRegression(model_name, model_initialization, emotion, dropout, hidden_dim)
        model.cuda()

        # load model
        model.load_state_dict(torch.load(save_dir))
        model.eval()
        dataset = Dataset(tokenizer, words, max_len)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        predictions = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].squeeze(1).to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask)
                predictions.append(outputs.cpu().detach().view(-1).numpy())
        word_emo_dict[emotion] = {word: prediction for word, prediction in zip(words, np.concatenate(predictions))}

    return word_emo_dict

# from training_word_emotion.utils import load_data, check_max_token_length
#
# emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
# max_lens = []
# for emotion in emotions:
#     df_train, df_test, df_val = load_data(emotion)
#     model_dir = "nghuyong/ernie-2.0-en"
#     tokenizer = AutoTokenizer.from_pretrained(model_dir)
#     max_len = check_max_token_length(tokenizer, emotion)
#     max_lens.append(max_len)
#
