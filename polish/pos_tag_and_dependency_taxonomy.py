"""
This file creates the taxonomies for pos tags (later node types), and parser tags (later edge types) for the Polish dataset.
"""

import pickle

# pos tags categories
pos_tags = ['ADJ',
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
parser_tags =['ROOT',
 'acl',
 'acl:relcl',
 'advcl',
 'advcl:cmpr',
 'advcl:relcl',
 'advmod',
 'advmod:arg',
 'advmod:emph',
 'advmod:neg',
 'amod',
 'amod:flat',
 'appos',
 'aux',
 'aux:cnd',
 'aux:imp',
 'aux:pass',
 'case',
 'cc',
 'cc:preconj',
 'ccomp',
 'ccomp:cleft',
 'ccomp:obj',
 'conj',
 'cop',
 'csubj',
 'dep',
 'det',
 'det:numgov',
 'det:nummod',
 'det:poss',
 'discourse:intj',
 'expl:pv',
 'fixed',
 'flat',
 'flat:foreign',
 'iobj',
 'list',
 'mark',
 'nmod',
 'nmod:arg',
 'nmod:flat',
 'nmod:poss',
 'nsubj',
 'nsubj:pass',
 'nummod',
 'nummod:gov',
 'obj',
 'obl',
 'obl:agent',
 'obl:arg',
 'obl:cmpr',
 'orphan',
 'parataxis:insert',
 'parataxis:obj',
 'punct',
 'vocative',
 'xcomp',
 'xcomp:cleft',
 'xcomp:pred']

# higher level custom parser tag categories
parser_tag_categories = {
    # 1. Descriptive Modifiers of Nouns
    "amod": "Descriptive Modifiers of Nouns",
    "amod:flat": "Descriptive Modifiers of Nouns",
    "det": "Descriptive Modifiers of Nouns",
    "det:numgov": "Descriptive Modifiers of Nouns",
    "det:nummod": "Descriptive Modifiers of Nouns",
    "det:poss": "Descriptive Modifiers of Nouns",
    "nmod": "Descriptive Modifiers of Nouns",
    "nmod:arg": "Descriptive Modifiers of Nouns",
    "nmod:flat": "Descriptive Modifiers of Nouns",
    "nmod:poss": "Descriptive Modifiers of Nouns",
    "nummod": "Descriptive Modifiers of Nouns",
    "nummod:gov": "Descriptive Modifiers of Nouns",
    "appos": "Descriptive Modifiers of Nouns",
    "acl": "Descriptive Modifiers of Nouns",
    "acl:relcl": "Descriptive Modifiers of Nouns",
    "flat": "Descriptive Modifiers of Nouns",
    "flat:foreign": "Descriptive Modifiers of Nouns",

    # 2. Descriptive Modifiers of Verbs
    "advmod": "Descriptive Modifiers of Verbs",
    "advmod:arg": "Descriptive Modifiers of Verbs",
    "advmod:emph": "Descriptive Modifiers of Verbs",
    "advmod:neg": "Descriptive Modifiers of Verbs",
    "advcl": "Descriptive Modifiers of Verbs",
    "advcl:cmpr": "Descriptive Modifiers of Verbs",
    "advcl:relcl": "Descriptive Modifiers of Verbs",
    "xcomp": "Descriptive Modifiers of Verbs",
    "xcomp:cleft": "Descriptive Modifiers of Verbs",
    "xcomp:pred": "Descriptive Modifiers of Verbs",

    # 3. Negations
    "advmod:neg": "Negations",

    # 4. Auxiliary and Modal Verbs
    "aux": "Auxiliary and Modal Verbs",
    "aux:cnd": "Auxiliary and Modal Verbs",
    "aux:imp": "Auxiliary and Modal Verbs",
    "aux:pass": "Auxiliary and Modal Verbs",
    "cop": "Auxiliary and Modal Verbs",

    # 5. Conjunctions and Coordination
    "cc": "Conjunctions and Coordination",
    "cc:preconj": "Conjunctions and Coordination",
    "conj": "Conjunctions and Coordination",
    "fixed": "Conjunctions and Coordination",
    "list": "Conjunctions and Coordination",

    # 6. Prepositions and Case Markers
    "case": "Prepositions and Case Markers",
    "mark": "Prepositions and Case Markers",
    "obl": "Prepositions and Case Markers",
    "obl:agent": "Prepositions and Case Markers",
    "obl:arg": "Prepositions and Case Markers",
    "obl:cmpr": "Prepositions and Case Markers",

    # 7. Subjects and Objects
    "nsubj": "Subjects and Objects",
    "nsubj:pass": "Subjects and Objects",
    "csubj": "Subjects and Objects",
    "obj": "Subjects and Objects",
    "iobj": "Subjects and Objects",
    "expl:pv": "Subjects and Objects",

    # 8. Clausal Complements and Subordinate Clauses
    "ccomp": "Clausal Complements and Subordinate Clauses",
    "ccomp:cleft": "Clausal Complements and Subordinate Clauses",
    "ccomp:obj": "Clausal Complements and Subordinate Clauses",
    "xcomp": "Clausal Complements and Subordinate Clauses",
    "xcomp:cleft": "Clausal Complements and Subordinate Clauses",
    "xcomp:pred": "Clausal Complements and Subordinate Clauses",
    "advcl": "Clausal Complements and Subordinate Clauses",
    "advcl:cmpr": "Clausal Complements and Subordinate Clauses",
    "advcl:relcl": "Clausal Complements and Subordinate Clauses",
    "acl": "Clausal Complements and Subordinate Clauses",
    "acl:relcl": "Clausal Complements and Subordinate Clauses",

    # 9. Punctuation and Symbols
    "punct": "Punctuation and Symbols",

    # 10. Interjections and Discourse Elements
    "discourse:intj": "Interjections and Discourse Elements",
    "vocative": "Interjections and Discourse Elements",

    # 11. Unspecified or Miscellaneous Dependencies
    "dep": "Unspecified or Miscellaneous Dependencies",
    "orphan": "Unspecified or Miscellaneous Dependencies",

    # 12. Parataxis
    "parataxis:insert": "Parataxis",
    "parataxis:obj": "Parataxis",

    # 13. Root of the Sentence
    "ROOT": "Root of the Sentence",

    # 14. Possessive Modifiers
    "det:poss": "Possessive Modifiers",
    "nmod:poss": "Possessive Modifiers",

    # 15. Reflexive and Pronominal Verbs
    "expl:pv": "Reflexive and Pronominal Verbs"
}

# create a name to index mapping for the pos tags and parser tags
pos_tags_dict = {pos: idx for idx, pos in enumerate(pos_tags)}
parser_tags = {tag: i for i, tag in enumerate(parser_tags)}

# create a name to index mapping for the higher level custom parser tag categories
parser_categories = {tag: i for i, tag in enumerate(set(parser_tag_categories.values()))}

# save pos tag to index mapping
with open('data/polish/pos_tags.pkl', 'wb') as f:
    pickle.dump(pos_tags_dict, f)

# save parser tag to index mapping
with open('data/polish/parser_tags.pkl', 'wb') as f:
    pickle.dump(parser_tags, f)

# save parser tag categories to index mapping (higher level)
with open('data/polish/parser_categories.pkl', 'wb') as f:
 pickle.dump(parser_categories, f)

# save the lower level parser tags to higher level parser tag categories mapping
with open('data/polish/parser_categories_dict.pkl', 'wb') as f:
 pickle.dump(parser_tag_categories, f)


# for i in parser_tags:
#  print(parser_tag_categories[i])
#
