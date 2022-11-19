import re
import nltk
from nltk.metrics import jaccard_distance
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet, wordnet_ic
from nltk.stem import WordNetLemmatizer
import numpy as np 
import pandas as pd
import itertools


nltk.download('words')
nltk.download('omw-1.4')
nltk.download('wordnet_ic')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')
regex = re.compile('[^a-z0-9]')

stopw = set(nltk.corpus.stopwords.words('english')) # english stopwords

tags = {'NN': wordnet.NOUN,
        'VB': wordnet.VERB,
        'JJ': wordnet.ADJ, 
        'RB': wordnet.ADV}

wnl = WordNetLemmatizer()

def tokenize(sentences):
    sentences_tokens = []
    sentences_pairs = []
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence) # tokenize
        pairs = nltk.pos_tag(tokens) # get the pos of the tokens
        tokens = [regex.sub('', w.lower()) for w in tokens if not w.lower() in stopw] # remove stopwords and symbols from each word
        tokens = [w for w in tokens if w] # remove empty elements
        sentences_tokens.append(set(tokens))
        sentences_pairs.append(pairs)
    return sentences_tokens, sentences_pairs

def pos_wn(pair):
    word = pair[0].lower()
    tag = tags.get(pair[1][:2].upper())
    if tag:
        word = wnl.lemmatize(word, pos = tag)
    return word

def lemmatize(sentences_POS):
    lemmas = []
    for sentence in sentences_POS:
        lemmatized = [pos_wn(pair) for pair in sentence]
        lemmatized = [regex.sub('', w.lower()) for w in lemmatized if not w.lower() in stopw] # remove stopwords and symbols from each word
        lemmatized = [w for w in lemmatized if w] # remove empty elements
        lemmas.append(set(lemmatized))
    return lemmas

def lesk(sentences, POS):
    syns = []
    for sentence, pairs in zip(sentences, POS) :
        synsets = [nltk.wsd.lesk(sentence, pair[0], pos = tags.get(pair[1][:2].upper())) for pair in pairs if pair[0].lower() not in stopw] # skip if the word is stopword else use lesk 
        synsets = [syn.name() for syn in synsets if syn] # we ignore the words without meaning
        syns.append(set(synsets))
    return syns

def syntactic_role_sim(POS1, POS2):
    similarities = []
    for pairs1, pairs2 in zip(POS1, POS2):
        syn1 = []
        syn2 = []
        for pair in pairs1:
            tag = tags.get(pair[1])
            if tag and pair[0].lower() not in stopw and pair[0].lower() in nltk.corpus.words.words():
                syn1.append((pair, (wordnet.synsets(pair[0], tag))))
        for pair in pairs2:
            tag = tags.get(pair[1])
            if tag and pair[0].lower() not in stopw and pair[0].lower() in nltk.corpus.words.words():
                syn2.append((pair, (wordnet.synsets(pair[0], tag))))
        lin_sum = 0
        for s1, s2 in itertools.product(syn1, syn2):
            print(s1, s2)
            if s1[0][1] == s2[0][1]:
                lin_sum += s1[1][0].lin_similarity(s2[1][0], semcor_ic)
        print(lin_sum)
            
        similarities.append((len(syn1) + len(syn2))/lin_sum)
    return similarities


def jaccard_empty(set1, set2):
    if len(set1) != 0 and len(set2) != 0:
        return jaccard_distance(set1, set2)
    else:
        return 0

class Features:
    def __init__(self, data):      
        self.pair1 = data['Sentence 1']
        self.pair2 = data['Sentence 2']

        self.tokens1, self.pos1 = tokenize(self.pair1)
        self.tokens2, self.pos2 = tokenize(self.pair2)

        self.lemmas1 = lemmatize(self.pos1)
        self.lemmas2 = lemmatize(self.pos2)

    def simple_tokens(self):
        self.jac_simp = [5*(1 - jaccard_distance(t[0], t[1])) for t in zip(self.tokens1, self.tokens2)]
        
    def lemmatizer(self): 
        self.jac_lemmas = [5*(1 - jaccard_distance(t[0], t[1])) for t in zip(self.lemmas1, self.lemmas2)]

    def lesk(self):
        l1 = lesk(self.pair1, self.pos1)
        l2 = lesk(self.pair2, self.pos2)
        self.jac_lesk = [5*(1 - jaccard_empty(t[0], t[1])) for t in zip(l1, l2)]

    def syntatic_role(self):
        self.synt = syntactic_role_sim(self.pos1, self.pos2)

    def extract_all(self):
        self.syntatic_role()
        self.simple_tokens()
        self.lemmatizer()
        self.lesk()
        return pd.DataFrame({'Simple': self.jac_simp, 'Lemmas': self.jac_lemmas, 'LESK': self.jac_lesk, 
                            'Syntatic': self.synt})




    