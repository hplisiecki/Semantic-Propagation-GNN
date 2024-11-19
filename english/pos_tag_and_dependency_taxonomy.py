"""
This file creates the taxonomies for pos tags (later node types), and parser tags (later edge types) for the English dataset.
"""

import pickle

# pos tags categories
pos_tags   = ['ADJ',
                 'ADP',
                 'ADV',
                 'AUX',
                 'CCONJ',
                 'DET',
                 'INTJ',
                 'NOUN',
                 'NUM',
                 'PART',
                 'PRON',
                 'PROPN',
                 'PUNCT',
                 'SCONJ',
                 'SPACE',
                 'SYM',
                 'VERB',
                 'X',
                 '!',
                 '?',
                 '…']

# parser tags categories
parser_tags = ['ROOT',
             'acl',
             'acomp',
             'advcl',
             'advmod',
             'agent',
             'amod',
             'appos',
             'attr',
             'aux',
             'auxpass',
             'case',
             'cc',
             'ccomp',
             'compound',
             'conj',
             'csubj',
             'csubjpass',
             'dative',
             'dep',
             'det',
             'dobj',
             'expl',
             'intj',
             'mark',
             'meta',
             'neg',
             'nmod',
             'npadvmod',
             'nsubj',
             'nsubjpass',
             'nummod',
             'oprd',
             'parataxis',
             'pcomp',
             'pobj',
             'poss',
             'preconj',
             'predet',
             'prep',
             'prt',
             'punct',
             'quantmod',
             'relcl',
             'xcomp']


# higher level custom parser tag categories
parser_tag_categories = {
    'ROOT': 'Root of the Sentence',

    # Descriptive Modifiers of Nouns
    'amod': 'Descriptive Modifiers of Nouns',
    'compound': 'Descriptive Modifiers of Nouns',
    'nummod': 'Descriptive Modifiers of Nouns',
    'quantmod': 'Descriptive Modifiers of Nouns',
    'det': 'Descriptive Modifiers of Nouns',
    'predet': 'Descriptive Modifiers of Nouns',
    'poss': 'Descriptive Modifiers of Nouns',
    'appos': 'Descriptive Modifiers of Nouns',
    'nmod': 'Descriptive Modifiers of Nouns',
    'acl': 'Descriptive Modifiers of Nouns',
    'relcl': 'Descriptive Modifiers of Nouns',

    # Descriptive Modifiers of Verbs
    'advmod': 'Descriptive Modifiers of Verbs',
    'npadvmod': 'Descriptive Modifiers of Verbs',
    'advcl': 'Descriptive Modifiers of Verbs',
    'prt': 'Descriptive Modifiers of Verbs',
    'mark': 'Descriptive Modifiers of Verbs',
    'acomp': 'Descriptive Modifiers of Verbs',  # Added acomp here

    # Negations
    'neg': 'Negations',

    # Auxiliary and Modal Verbs
    'aux': 'Auxiliary and Modal Verbs',
    'auxpass': 'Auxiliary and Modal Verbs',

    # Conjunctions and Coordination
    'cc': 'Conjunctions and Coordination',
    'conj': 'Conjunctions and Coordination',
    'preconj': 'Conjunctions and Coordination',

    # Prepositional Modifiers
    'prep': 'Prepositional Modifiers',
    'pobj': 'Prepositional Modifiers',
    'pcomp': 'Prepositional Modifiers',
    'case': 'Prepositional Modifiers',

    # Subjects and Objects
    'nsubj': 'Subjects and Objects',
    'nsubjpass': 'Subjects and Objects',
    'csubj': 'Subjects and Objects',
    'csubjpass': 'Subjects and Objects',
    'dobj': 'Subjects and Objects',
    'dative': 'Subjects and Objects',
    'attr': 'Subjects and Objects',
    'agent': 'Subjects and Objects',
    'expl': 'Subjects and Objects',

    # Clausal Complements
    'ccomp': 'Clausal Complements',
    'xcomp': 'Clausal Complements',
    'oprd': 'Clausal Complements',

    # Punctuation and Symbols
    'punct': 'Punctuation and Symbols',

    # Interjections and Discourse Elements
    'intj': 'Interjections and Discourse Elements',
    'meta': 'Interjections and Discourse Elements',

    # Unspecified or Miscellaneous Dependencies
    'dep': 'Unspecified or Miscellaneous Dependencies',

    # Parataxis
    'parataxis': 'Parataxis'
}

# create a name to index mapping for the pos tags and parser tags
pos_tags_dict = {pos: idx for idx, pos in enumerate(pos_tags)}
parser_tags_dict = {pos: idx for idx, pos in enumerate(parser_tags)}

# create a name to index mapping for the higher level custom parser tag categories
parser_tag_categories_dict = {tag: idx for idx, tag in enumerate(set(parser_tag_categories.values()))}

# save pos tag to index mapping
with open('data/english/pos_tags.pkl', 'wb') as f:
    pickle.dump(pos_tags_dict, f)

# save parser tag to index mapping
with open('data/english/parser_tags.pkl', 'wb') as f:
    pickle.dump(parser_tags_dict, f)

# save parser tag categories to index mapping (higher level)
with open('data/english/parser_tag_categories.pkl', 'wb') as f:
    pickle.dump(parser_tag_categories_dict, f)

# save the lower level parser tags to higher level parser tag categories mapping
with open('data/english/parser_tag_categories_dict.pkl', 'wb') as f:
    pickle.dump(parser_tag_categories, f)





