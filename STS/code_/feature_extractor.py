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
brown_ic = wordnet_ic.ic('ic-brown.dat')
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

def longest_common_subsequence(t1, t2):

    tokens1 = list(t1)
    tokens2 = list(t2)

    dp = [[None] * (len(tokens1) + 1) for i in range(len(tokens2) + 1)]
    for i in range(len(tokens2) + 1):
        for j in range(len(tokens1) + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif tokens2[i - 1] == tokens1[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[len(tokens2)][len(tokens1)]

def longest_common_substring(t1, t2):

    s = ''.join(t1)
    t = ''.join(t2)
    n = len(s)
    m = len(t)

    dp = [[0 for i in range(m + 1)] for j in range(2)]
    res = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if (s[i - 1] == t[j - 1]):
                dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
                if (dp[i % 2][j] > res):
                    res = dp[i % 2][j]
            else:
                dp[i % 2][j] = 0
    return res

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

def normalize(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

def get_trigrams(tokens):
    tkn_lst = list(tokens)
    trigrams = []
    for i in range(len(tkn_lst) - 2):
        trigram = (tkn_lst[i], tkn_lst[i + 1], tkn_lst[i + 2])
        trigrams.append(trigram)
    return trigrams

def get_fourgrams(tokens):
    tkn_lst = list(tokens)
    fourgrams = []
    for i in range(len(tkn_lst) - 3):
        fourgram = (tkn_lst[i], tkn_lst[i + 1], tkn_lst[i + 2], tkn_lst[i + 3])
        fourgrams.append(fourgram)
    return fourgrams

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

    def n_grams(self):
        self.jac_trigrams = [5*(1 - jaccard_empty(set(get_trigrams(t[0])), set(get_trigrams(t[1])))) for t in zip(self.tokens1, self.tokens2)]
        self.jac_fourgrams = [5 * (1 - jaccard_empty(set(get_fourgrams(t[0])), set(get_fourgrams(t[1])))) for t in
                             zip(self.tokens1, self.tokens2)]

    def syntatic_role(self):
        self.synt = syntactic_role_sim(self.pos1, self.pos2)

    def lcs_sequence(self):
        self.lcs_sequence_score = 5 * normalize([longest_common_subsequence(t[0], t[1])/len(t[0]) for t in
                             zip(self.tokens1, self.tokens2)])

    def lcs_string(self):
        self.lcs_string_score = 5 * normalize([longest_common_substring(t[0], t[1])/len(t[0]) for t in
                             zip(self.tokens1, self.tokens2)])

    def extract_all(self):
        #self.syntatic_role()
        self.simple_tokens()
        self.lemmatizer()
        self.lesk()
        self.n_grams()
        self.lcs_sequence()
        self.lcs_string()

        return pd.DataFrame({'Simple': self.jac_simp, 'Lemmas': self.jac_lemmas, 'LESK': self.jac_lesk, '3-grams': self.jac_trigrams, '4-grams': self.jac_fourgrams, 'Longest Common Subsequence': self.lcs_sequence_score, 'Longest Common Substring': self.lcs_string_score})




    