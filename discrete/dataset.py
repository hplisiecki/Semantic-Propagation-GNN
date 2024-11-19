from torch_geometric.data import Data, Dataset
import spacy
import torch
from discrete.word_emo_prediction_discrete import get_discrete_emotions
import pickle
import requests
import numpy as np

# Load the small English language model in spaCy
nlp = spacy.load('en_core_web_sm')

# load the english pos tags to index dictionary
with open(r'D:\GitHub\bias_free_modeling\data/english/pos_tags.pkl', 'rb') as f:
    pos_tags = pickle.load(f)

# load the english parser tags to index dictionary
with open(r'D:\GitHub\bias_free_modeling\data/english/parser_tags.pkl', 'rb') as f:
    parser_tags = pickle.load(f)

# load the label map for discrete emotions
with open(r'D:\GitHub\bias_free_modeling\data/discrete/label_map.pkl', 'rb') as f:
    label_map = pickle.load(f)

# load the lower level to custom higher level parser tag categories mapping
with open(r'D:\GitHub\bias_free_modeling\data/english/parser_tag_categories.pkl', 'rb') as f:
    parser_tag_categories = pickle.load(f)

# load the higher level parser tag categories to index dictionary
with open(r'D:\GitHub\bias_free_modeling\data/english/parser_tag_categories_dict.pkl', 'rb') as f:
    parser_tag_categories_dict = pickle.load(f)

# Define a set of punctuation marks to retain
retained_punctuation = {'!', '?'}
ellipsis = '…' # This is the actual ellipsis character

# Load the English stopwords
stopwords_url = "https://gist.githubusercontent.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords"
stop_words = requests.get(stopwords_url).text.splitlines()

class SPropDataset(Dataset):
    """
    This class prepares a dataset for the Discrete English SProp model. It tokenizes the text, calls the word_emo_prediction_discrete to
    get the emotion intensity for each word in the text, builds syntactic graphs, and prepares the data for the training.

    The dataset precomputes al texts so that they don't have to be computed during training.

    Parameters:
    df (DataFrame): A DataFrame containing the text and most_voted columns
    """
    def __init__(self, df):
        self.texts = df['text'] # Get the text column
        self.labels = df['most_voted'].map(label_map).values.astype(int) # Get the most_voted column and map the labels to integers
        self.pos_tags = pos_tags # Get the pos tags dictionary
        self.num_pos_tags = len(self.pos_tags) # Get the number of pos tags
        self.parser_tag_categories = parser_tag_categories # Get the parser tag categories dictionary
        self.emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust'] # Define relevant emotions
        self.word_emo_dict = self.build_discrete_emotions() # Get the emotion intensity for each word in the text

        # Precompute and store data for each text
        self.data_list = [self.build_data(idx) for idx in range(len(self.texts))]

    def get_parser_tag(self, tag):
        """
        Utility function to get the higher level parser tag category from the lower level parser tag
        """
        general_category = parser_tag_categories_dict[tag]
        return parser_tag_categories[general_category]

    def build_discrete_emotions(self):
        """
        This function tokenizes all texts, gets the emotion intensity for each word in the text, and returns a dictionary
        """
        words = set()
        for text in self.texts:
            doc = nlp(text)
            for token in doc:
                if ((token.text in retained_punctuation or token.text == ellipsis) or # Doesn't predict for punctuation
                        (not token.is_punct and token.text.lower() not in stop_words) or # or stopwords
                        token.dep_ == 'neg'):
                    words.add(token.text)

        word_emo_dict = get_discrete_emotions(list(words))

        return word_emo_dict

    def build_data(self, idx):
        """
        This function builds the data for a given text.
        """

        text = self.texts[idx] # Get the text
        label = int(self.labels[idx])  # Convert label to integer

        doc = nlp(text) # Tokenize the text
        sentences = list(doc.sents) # Get the sentences
        num_sentences = len(sentences) # Get the number of sentences

        tokens = [] # Collect features for word nodes

        # Because we are removing some symbols, we have to keep track of the moving index:
        index_map = {}
        new_index = 0
        # Loop over sentences
        for sent_id, sent in enumerate(sentences):
            sent_id += 1 # Assign sentence ID
            for token in sent:
                if ((token.text in retained_punctuation or token.text == ellipsis) or # Filter out punctuation
                        (not token.is_punct) or
                        token.dep_ == 'neg'):
                    pos_tag = token.text if token.text in retained_punctuation or token.text == ellipsis else token.pos_ # Get the pos tag
                    # Collect token features
                    tokens.append({
                        'pos': pos_tag, # Note down the pos tag
                        'parser': token.dep_, # Note down the parser tag
                        'head': token.head.i, # Note down the head
                        'index': token.i, # Note down the index
                        'sentence_id': sent_id,  # Assign sentence ID
                        'word': token.text # Note down the word
                    })
                    emotions = {emotion: self.word_emo_dict[emotion].get(token.text, 0.0) for emotion in self.emotions} # Get the emotion intensity for the word
                    tokens[-1].update(emotions) # And note them down as well
                    index_map[token.i] = new_index # Update the index map
                    new_index += 1

        # Create node features
        node_features = []
        for token in tokens: # these include
            feature = [token[emotion] for emotion in self.emotions]  # emotion intensities
            feature.append(token['sentence_id'] / num_sentences) # and sentence location
            node_features.append(feature)

        # Create sentence nodes
        sentence_nodes = []
        for sent_id in range(num_sentences): # these include
            sent_id += 1
            # Ensure that some tokens exist  (needed if you want to assign nonempty emotion intensities)
            # for example by averaging the words in the sentence
            if len([token for token in tokens if token['sentence_id'] == sent_id]) > 0:
                sentence_feature = [
                    0.0                     # empty emotion intensities
                    for _ in self.emotions
                ]
                sentence_feature.append(sent_id / num_sentences) # and sentence location

            else: # else create an empty sentence node (in current implementation, this node is the same as the "nonempty" node
                sentence_feature = [0.0 for _ in self.emotions]
                sentence_feature.append(sent_id / num_sentences)
            sentence_nodes.append(sentence_feature)

        node_features.extend(sentence_nodes)  # Append sentence nodes to node features


        # Create node types
        node_types = []
        for token in tokens:
            node_types.append(self.pos_tags[token['pos']]) # We are just using the pos tags as node types

        """
        The types of the sentence nodes are not explicitly defined in the node type dictionaries.
        I initialize them instead as the index of the last pos tag + 1 for simplicity
        """
        node_types.extend([len(pos_tags)] * num_sentences)  # Append sentence node types



        # Reformat the node features and node types to tensors, or initialize them as empty tensors if no nodes are present
        if node_features:
            node_features = torch.tensor(node_features, dtype=torch.float)
            node_types = torch.tensor(node_types, dtype=torch.long)  # Shape: [num_nodes]
        else:
            node_features = torch.empty((0, 3), dtype=torch.float)
            node_types = torch.empty((0,), dtype=torch.long)

        # Create edge indices and edge types
        edge_index = []
        edge_type = []

        """
        The code below iterates through tokens, gets their updated head and child indexes and creates edge tuples
        for words that are connected according to the spacy syntactic parser.
        """
        for token in tokens:
            original_head_idx = token['head']
            if original_head_idx in index_map and token['index'] in index_map:
                new_head_idx = index_map[original_head_idx] # Get the new head index
                new_child_idx = index_map[token['index']] # Get the new child index
                if new_head_idx != new_child_idx:
                    # Add forward edge
                    edge_index.append([new_head_idx, new_child_idx]) # Add the edge tuple
                    edge_type.append(self.get_parser_tag(token['parser'])) # And we add the parser index as the edge type

        # Create edges from words to sentence nodes and assign the edge type as the number of parser tag categories
        for token in tokens:
            edge_index.append([index_map[token['index']], len(tokens) + token['sentence_id'] - 1])  # word to sentence node
            edge_type.append(len(self.parser_tag_categories))  # As in the case of pos sentence tags, we assign the edge type as the number of parser tag categories

        # Create edges between sentence nodes sequentially
        for sent_id in range(num_sentences):
            if sent_id < num_sentences - 1:
                edge_index.append([len(tokens), len(tokens) + 1]) # Add the edge tuple
                edge_type.append(len(self.parser_tag_categories) + 1) # Special edge type for sentence to sentence edges

        # Reformat the edge indices and edge types to tensors, or initialize them as empty tensors if no edges are present
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(edge_type, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)

        # get the words
        words = [token['word'] for token in tokens]

        # Create Data object
        data = Data(
            x=node_features,                         # Node features
            node_type=node_types,                    # Node types (integer indices)
            edge_index=edge_index,                   # Edge indices
            edge_type=edge_type,                     # Edge types (scalar indices)
            y=torch.tensor(label, dtype=torch.long),  # Labels
            word = words # The words in the text
        )

        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# DEBUGGING CODE / EXAMPLE USAGE:

# import pandas as pd
# df = pd.DataFrame({'text': ['This is a test sentence. This is a test sentence.', 'This is another test sentence. This is another test sentence.'], \
#                           'most_voted' : ['anger','joy']})
#
# data = SPropDataset(df)

# from polish_hierarchical.utils import load_data
# train, test, val = load_data()
#
# val_dataset = SentimentGraphDataset(val)
#
# for idx, i in enumerate(val_dataset):
#     x = i.x
#     if torch.isnan(x).any():
#         print('found nan')
#         break
#
# for i in val['text']:
#     data =
#
# for row in data.x:
#     if torch.isnan(row).any():
#         break
#
#
# text = 'Równowaga między pracą a snem wg. Konfederacji. #Sejm #worklifesleepbalance odnośnik '

