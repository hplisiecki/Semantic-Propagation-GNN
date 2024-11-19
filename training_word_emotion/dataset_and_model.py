import torch
from torch import nn
import numpy as np

###############################################################################
"""
Dataset and model classes.
"""
###############################################################################

class Dataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, df, max_len):
        """
        Initialize the dataset.
        :param tokenizer:  the tokenizer to use
        :param df: the dataframe to use
        :param max_len:  the maximum length of the sequences
        :param metric_names: the names of the metrics to use
        """
        self.tokenizer = tokenizer
        self.labels = df['score'].values.astype(float)


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
                                         return_tensors="pt") for text in df['word']]

    def classes(self):
        """
        Return the classes of the dataset.
        :return: the classes
        """
        return self.labels

    def __len__(self):
        """
        Return the length of the dataset.
        :return: the length
        """
        setattr(self, 'length', len(self.labels))
        return self.length

    def get_batch_labels(self, idx):
        """
        Return the labels of the batch.
        :param idx: the index of the batch
        :return: the labels
        """

        return np.array(self.labels[idx])

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

    def __getitem__(self, idx):
        """
        Return the item at the given index.
        :param idx: the index
        :return: the item
        """
        if type(self.tokenizer) == list:
            batch_texts1, batch_texts2 = self.get_batch_texts(idx)
            batch_y = self.get_batch_labels(idx)
            return batch_texts1, batch_texts2, batch_y
        else:
            batch_texts = self.get_batch_texts(idx)
            batch_y = self.get_batch_labels(idx)
            return batch_texts, batch_y


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
