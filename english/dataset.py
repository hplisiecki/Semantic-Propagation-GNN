from torch_geometric.data import Data, Dataset
import spacy
import torch
from english.word_emo_prediction import get_valence_arousal_dominance
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

# load the lower level to custom higher level parser tag categories mapping
with open(r'D:\GitHub\bias_free_modeling\data/english/parser_tag_categories.pkl', 'rb') as f:
    parser_tag_categories = pickle.load(f)

# load the higher level parser tag categories to index dictionary
with open(r'D:\GitHub\bias_free_modeling\data/english/parser_tag_categories_dict.pkl', 'rb') as f:
    parser_tag_categories_dict = pickle.load(f)

# Define a set of punctuation marks to retain
retained_punctuation = {'!', '?'}
ellipsis = '…'  # This is the actual ellipsis character

# Load the English stopwords
stopwords_url = "https://gist.githubusercontent.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords"
stop_words = requests.get(stopwords_url).text.splitlines()

class SPropDataset(Dataset):
    """
    This class prepares a dataset for the Continuous English SProp model. It tokenizes the text, calls the build_valence_arousal_dominance to
    get the affective metrics for each word in the text, builds syntactic graphs, and prepares the data for the training.

    The dataset precomputes al texts so that they don't have to be computed during training.

    Parameters:
    df (DataFrame): A DataFrame containing the text and ground truth for valence, arousal, and dominance

    Comment:
    This dataset is set up so that using it with minor tweaking you can also try to predict Dominance of texts as well. I was thinking about including
    it in the paper, but decided not to, as it would be yet another dimension I had to report, and I know that in some
    of my earlier affective experiments it can have a detrimental result on predicting valence. Hence, I decided to leave it be
    and focus on Valence and Arousal. The performance for Dominance is not that great, but still better than normal lexicon averaging.
    """
    def __init__(self, df):
        self.texts = df['text'] # Get the text from the DataFrame
        self.valence = df['norm_Valence_M'].values.astype(float) # Get the normalized valence from the DataFrame
        self.arousal = df['norm_Arousal_M'].values.astype(float) # Get the normalized arousal from the DataFrame
        self.dominance = df['norm_Dominance_M'].values.astype(float) # Get the normalized dominance from the DataFrame
        self.pos_tags = pos_tags # Get the POS tags
        self.parser_tag_categories = parser_tag_categories # Get the parser tag categories
        self.valence_words, self.arousal_words, self.dominance_words = self.build_valence_arousal_dominance() # get the affective metrics for all words


        # Precompute and store data for each text
        self.data_list = [self.build_data(idx) for idx in range(len(self.texts))]

    def get_parser_tag(self, tag):
        """
        Utility function to get the higher level parser tag category from the lower level parser tag
        """
        general_category = parser_tag_categories_dict[tag]
        return parser_tag_categories[general_category]

    def build_valence_arousal_dominance(self):
        """
        This function tokenizes all texts, gets the affective metrics for each word,
        and returns the valence, arousal, and dominance dictionaries
        """
        words = set()
        for text in self.texts:
            doc = nlp(text)
            for token in doc:
                if ((token.text in retained_punctuation or token.text == ellipsis) or # Doesn't predict for punctuation
                        (not token.is_punct and token.text.lower() not in stop_words)): # or stopwords
                    words.add(token.text)
        valence, arousal, dominance = get_valence_arousal_dominance(list(words))
        valence_words = {word: v for word, v in zip(words, valence)}
        arousal_words = {word: a for word, a in zip(words, arousal)}
        dominance_words = {word: d for word, d in zip(words, dominance)}
        return valence_words, arousal_words, dominance_words

    def build_data(self, idx):
        """
        This function builds the data for a given text.
        """
        text = self.texts[idx] # Get the text
        valence = self.valence[idx] # Get the valence
        arousal = self.arousal[idx] # Get the arousal

        doc = nlp(text) # Tokenize the text
        sentences = list(doc.sents) # Get the sentences
        num_sentences = len(sentences) # Get the number of sentences

        tokens = [] # Collect features for word nodes

        # Because we are removing some symbols, we have to keep track of the moving index:
        index_map = {}
        new_index = 0
        # Loop over sentences
        for sent_id, sent in enumerate(sentences):
            sent_id += 1
            for token in sent:
                if ((token.text in retained_punctuation or token.text == ellipsis) or # Filter out punctuation
                        (not token.is_punct) or
                        token.dep_ == 'neg'):
                    pos_tag = token.text if token.text in retained_punctuation or token.text == ellipsis else token.pos_ # Get the pos tag
                    # Collect token features
                    tokens.append({
                        'pos': pos_tag, # The POS tag
                        'parser': token.dep_, # The parser tag
                        'head': token.head.i, # The head of the token
                        'valence': self.valence_words.get(token.text, 0.0), # The valence of the word
                        'arousal': self.arousal_words.get(token.text, 0.0), # The arousal of the word
                        'index': token.i, # The index of the token
                        'sentence_id': sent_id,  # Assign sentence ID
                        'word': token.text # The word
                    })

                    index_map[token.i] = new_index # Keep track of the moving index
                    new_index += 1

        # Create node features
        node_features = []
        for token in tokens: # these include
            feature = [
                token['valence'], # The valence of the word
                token['arousal'], # The arousal of the word
                # token['dominance'] # as I said you can use dominance if you want, but there was no improvement for other metrics
                token['sentence_id'] / num_sentences  # Normalized sentence position
            ]

            node_features.append(feature)

        # Create sentence nodes
        sentence_nodes = []  # these include
        for sent_id in range(num_sentences):
            sent_id += 1
            # Ensure that some tokens exist  (needed if you want to assign nonempty emotion intensities)
            # for example by averaging the words in the sentence
            if len([token for token in tokens if token['sentence_id'] == sent_id]) > 0:
                sentence_feature = [
                    0,                      # empty emotion intensities
                    0,
                    sent_id / num_sentences # and sentence location
                ]
            else: # else create an empty sentence node (in current implementation, this node is the same as the "nonempty" node
                sentence_feature = [0.0, 0.0, sent_id / num_sentences]
            sentence_nodes.append(sentence_feature)  # Define as needed

        node_features.extend(sentence_nodes)  # Append sentence nodes to node features

        # Create node types
        node_types = []
        for token in tokens:
            node_types.append(self.pos_tags[token['pos']])  # We are just using the pos tags as node types

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
                    edge_type.append(self.get_parser_tag(token['parser']))  # And we add the parser index as the edge type

        # Create edges from words to sentence nodes and assign the edge type as the number of parser tag categories
        for token in tokens:
            edge_index.append([index_map[token['index']], len(tokens) + token['sentence_id'] - 1])  # word to sentence node
            edge_type.append(len(self.parser_tag_categories))  # As in the case of pos sentence tags, we assign the edge type as the number of parser tag categories

        # Create edges between sentence nodes sequentially
        for sent_id in range(num_sentences):
            if sent_id < num_sentences - 1:
                edge_index.append([len(tokens), len(tokens) + 1]) # sentence to sentence node
                edge_type.append(len(self.parser_tag_categories) + 1)  # Special edge type for sentence to sentence edges

        # Reformat the edge indices and edge types to tensors, or initialize them as empty tensors if no edges are present
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Shape: [2, num_edges]
            edge_type = torch.tensor(edge_type, dtype=torch.long)  # Shape: [num_edges]
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)

        # Get the words
        words = [token['word'] for token in tokens]

        # Create Data object
        data = Data(
            x=node_features,                         # Node features
            node_type=node_types,                    # Node types (integer indices)
            edge_index=edge_index,                   # Edge indices
            edge_type=edge_type,                     # Edge types (scalar indices)
            v=torch.tensor(valence, dtype=torch.float), # Valence
            a=torch.tensor(arousal, dtype=torch.float), # Arousal
            word = words # The words
        )

        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# DEBUGGING CODE / EXAMPLE USAGE:

# import pandas as pd
# df = pd.DataFrame({'text': ['This is a test sentence. This is a test sentence?!', 'This is another test sentence. This is another test sentence.'], \
#                           'norm_Valence_M' : [0.5, 0.6], 'norm_Arousal_M' : [0.7, 0.8], 'norm_Dominance_M' : [0.9, 0.1]})
#
# data = SPropDataset(df)

