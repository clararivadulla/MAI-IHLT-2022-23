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

stopw = set(nltk.corpus.stopwords.words('english'))  # english stopwords

tags = {'NN': wordnet.NOUN,
        'VB': wordnet.VERB,
        'JJ': wordnet.ADJ,
        'RB': wordnet.ADV}

wnl = WordNetLemmatizer()

def tokenize(sentences):
    sentences_tokens = []
    sentences_pairs = []
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)  # tokenize
        pairs = nltk.pos_tag(tokens)  # get the pos of the tokens
        tokens = [regex.sub('', w.lower()) for w in tokens if
                  not w.lower() in stopw]  # remove stopwords and symbols from each word
        tokens = [w for w in tokens if w]  # remove empty elements
        sentences_tokens.append(set(tokens))
        sentences_pairs.append(pairs)
    return sentences_tokens, sentences_pairs


def pos_wn(pair):
    word = pair[0].lower()
    tag = tags.get(pair[1][:2].upper())
    if tag:
        word = wnl.lemmatize(word, pos=tag)
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

    return dp[len(tokens2)][len(tokens1)] / min(len(tokens1), len(tokens2))

def num_verbs_diff(pos1, pos2):
    v1 = len([v for v in pos1 if v[1].startswith('V')])
    v2 = len([v for v in pos2 if v[1].startswith('V')])
    if v1 == 0 and v2 == 0:
        return 0
    return abs(v1 - v2) / (v1 + v2)

def num_nouns_diff(pos1, pos2):
    n1 = len([n for n in pos1 if n[1].startswith('N')])
    n2 = len([n for n in pos2 if n[1].startswith('N')])
    if n1 == 0 and n2 == 0:
        return 0
    return abs(n1 - n2) / (n1 + n2)

def num_adjs_diff(pos1, pos2):
    a1 = len([a for a in pos1 if a[1].startswith('J')])
    a2 = len([a for a in pos2 if a[1].startswith('J')])
    if a1 == 0 and a2 == 0:
        return 0
    return abs(a1 - a2) / (a1 + a2)

def num_advs_diff(pos1, pos2):
    adv1 = len([a for a in pos1 if a[1].startswith('R')])
    adv2 = len([a for a in pos2 if a[1].startswith('R')])
    if adv1 == 0 and adv2 == 0:
        return 0
    return abs(adv1 - adv2) / (adv1 + adv2)

def longest_common_substring(t1, t2):

    s = ''.join(t1)
    t = ''.join(t2)
    n = len(s)
    m = len(t)

    dp = [[0 for i in range(m + 1)] for j in range(2)]
    res = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
                if dp[i % 2][j] > res:
                    res = dp[i % 2][j]
            else:
                dp[i % 2][j] = 0
    return res / min(m, n)


def lemmatize(sentences_POS):
    lemmas = []
    for sentence in sentences_POS:
        lemmatized = [pos_wn(pair) for pair in sentence]
        lemmatized = [regex.sub('', w.lower()) for w in lemmatized if
                      not w.lower() in stopw]  # remove stopwords and symbols from each word
        lemmatized = [w for w in lemmatized if w]  # remove empty elements
        lemmas.append(set(lemmatized))
    return lemmas


def lesk(sentences, POS):
    syns = []
    for sentence, pairs in zip(sentences, POS):
        synsets = [nltk.wsd.lesk(sentence, pair[0], pos=tags.get(pair[1][:2].upper())) for pair in pairs if
                   pair[0].lower() not in stopw]  # skip if the word is stopword else use lesk
        synsets = [syn.name() for syn in synsets if syn]  # we ignore the words without meaning
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

        similarities.append((len(syn1) + len(syn2)) / lin_sum)
    return similarities

def jaccard_empty(set1, set2):
    if len(set1) != 0 and len(set2) != 0:
        return jaccard_distance(set1, set2)
    else:
        return 0

def levenshtein_distance(s1, s2):
    return nltk.edit_distance(s1, s2) / max(len(s1), len(s2))

class Features:

    def __init__(self, data):

        self.pair1 = data['Sentence 1']
        self.pair2 = data['Sentence 2']

        self.tokens1, self.pos1 = tokenize(self.pair1)
        self.tokens2, self.pos2 = tokenize(self.pair2)

        self.lemmas1 = lemmatize(self.pos1)
        self.lemmas2 = lemmatize(self.pos2)

    def simple_tokens(self):
        self.jac_simp = [(1 - jaccard_distance(t[0], t[1])) for t in zip(self.tokens1, self.tokens2)]

    def lemmatizer(self):
        self.jac_lemmas = [(1 - jaccard_distance(t[0], t[1])) for t in zip(self.lemmas1, self.lemmas2)]

    def lesk(self):
        l1 = lesk(self.pair1, self.pos1)
        l2 = lesk(self.pair2, self.pos2)
        self.jac_lesk = [5 * (1 - jaccard_empty(t[0], t[1])) for t in zip(l1, l2)]

    def n_grams(self):
        self.jac_2grams = [(1 - jaccard_empty(set([g for g in nltk.ngrams(t[0], 2)]), set([g for g in nltk.ngrams(t[1], 2)]))) for
                           t in zip(self.tokens1, self.tokens2)]
        self.jac_3grams = [(1 - jaccard_empty(set([g for g in nltk.ngrams(t[0], 3)]), set([g for g in nltk.ngrams(t[1], 3)]))) for
                           t in zip(self.tokens1, self.tokens2)]
        self.jac_4grams = [(1 - jaccard_empty(set([g for g in nltk.ngrams(t[0], 4)]), set([g for g in nltk.ngrams(t[1], 4)]))) for
                           t in zip(self.tokens1, self.tokens2)]
        self.jac_2grams_l = [
            (1 - jaccard_empty(set([g for g in nltk.ngrams(l[0], 2)]), set([g for g in nltk.ngrams(l[1], 2)]))) for
            l in zip(self.lemmas1, self.lemmas2)]
        self.jac_3grams_l = [
            (1 - jaccard_empty(set([g for g in nltk.ngrams(l[0], 3)]), set([g for g in nltk.ngrams(l[1], 3)]))) for
            l in zip(self.lemmas1, self.lemmas2)]
        self.jac_4grams_l = [
            (1 - jaccard_empty(set([g for g in nltk.ngrams(l[0], 4)]), set([g for g in nltk.ngrams(l[1], 4)]))) for
            l in zip(self.lemmas1, self.lemmas2)]

    def syntatic_role(self):
        self.synt = syntactic_role_sim(self.pos1, self.pos2)

    def lcs_sequence(self):
        self.lcs_sequence_score = [longest_common_subsequence(t[0], t[1]) for t in
                                             zip(self.tokens1, self.tokens2)]

    def lcs_string(self):
        self.lcs_string_score = [longest_common_substring(t[0], t[1]) for t in
                                           zip(self.tokens1, self.tokens2)]
    def levenshtein_dist(self):
        self.leven_dist = [levenshtein_distance(p[0], p[1]) for p in
                                             zip(self.pair1, self.pair2)]
    def num_tags(self):
        self.verbs_diff = [num_verbs_diff(p[0], p[1]) for p in
                                             zip(self.pos1, self.pos2)]
        self.nouns_diff = [num_nouns_diff(p[0], p[1]) for p in
                           zip(self.pos1, self.pos2)]
        self.adjs_diff = [num_adjs_diff(p[0], p[1]) for p in
                           zip(self.pos1, self.pos2)]
        self.advs_diff = [num_advs_diff(p[0], p[1]) for p in
                          zip(self.pos1, self.pos2)]

    def extract_all(self):
        # self.syntatic_role()
        self.simple_tokens()
        self.lemmatizer()
        self.lesk()
        self.n_grams()
        self.lcs_sequence()
        self.lcs_string()
        self.levenshtein_dist()
        self.num_tags()

        return pd.DataFrame(
            {'Jaccard Tokens': self.jac_simp, 'Jaccard Lemmas': self.jac_lemmas, 'Lesk Similarity': self.jac_lesk, '2-grams Tokens': self.jac_2grams,
             '3-grams Tokens': self.jac_3grams, '4-grams Tokens': self.jac_4grams, '2-grams Lemmas': self.jac_2grams_l,
             '3-grams Lemmas': self.jac_3grams_l, '4-grams Lemmas': self.jac_4grams_l,
             'Longest Common Subsequence': self.lcs_sequence_score, 'Longest Common Substring': self.lcs_string_score,
             'Levenshtein Distance': self.leven_dist, 'Verb Tags': self.verbs_diff, 'Noun Tags': self.nouns_diff,
             'Adjective Tags': self.adjs_diff, 'Adverb Tags': self.advs_diff})
