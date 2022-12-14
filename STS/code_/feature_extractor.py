import re
import nltk
from nltk.metrics import jaccard_distance
from nltk.stem import PorterStemmer
from nltk.corpus.reader.wordnet import information_content
from nltk.corpus import wordnet, wordnet_ic
from locale import atof, setlocale, LC_NUMERIC
from nltk import ne_chunk
from numpy import dot
from numpy.linalg import norm
from nltk.stem import WordNetLemmatizer
from scipy.spatial.distance import hamming
from unidecode import unidecode
import math
import spacy
import numpy as np
import pandas as pd
import itertools
from scipy.stats import pearsonr


setlocale(LC_NUMERIC, 'en_US.UTF-8')

nltk.download('words')
nltk.download('omw-1.4')
nltk.download('wordnet_ic')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('conll2000')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')
brown_ic = wordnet_ic.ic('ic-brown.dat')

nlp = spacy.load('en_core_web_sm', disable = ['parser','ner'])

stopw = set(nltk.corpus.stopwords.words('english'))  # english stopwords

tags = {'NN': wordnet.NOUN,
        'VB': wordnet.VERB,
        'JJ': wordnet.ADJ,
        'RB': wordnet.ADV}

wnl = WordNetLemmatizer()


def preprocess(sentence):
    sentence = unidecode(sentence)                    # converts to ascii everything
    sentence = re.sub(r"(-\b|\b-|/)", "", sentence)   # remove hyphens and forward slashes
    sentence = re.sub(r"$US", "$", sentence)          # normalize dollar values, no other money symbol appears in the train dataset
    sentence = re.sub(r"<\.?(.*?)>", r"\1", sentence) # matches the interior string of the format <XYZ> and returns XYZ
    
    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'s", " is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)
    sentence = re.sub(r"  ", " ", sentence)
    return sentence


def preprocess_tokenized(tokens):
    clean_tokens = []
    for token in tokens:
        if token.isalnum():
            clean_tokens.append(token.lower())
        else:
            try:
                numeric_str = str(round(atof(token), 1))
                if numeric_str[-2:] == '.0':
                    clean_tokens.append(numeric_str[:-2])
                else:
                    clean_tokens.append(numeric_str)
            except:
                pass
    return clean_tokens


def tokenize(sentence, sw=False):
    tokens = nltk.word_tokenize(sentence)                       # tokenize the sentence
    pairs = nltk.pos_tag(tokens)                                # get the POS-tag of the tokens
    if sw:                                                      
        tokens = [w for w in tokens if not w.lower() in stopw]  # remove stopwords

    tokens = preprocess_tokenized(tokens)
    return set(tokens), pairs, tokens

def lemmatize(sentences_POS, sw=False):
    lemmas = [pos_wn(pair) for pair in sentences_POS]           # obtain the lemmas
    if sw:
        lemmas = [w for w in lemmas if not w.lower() in stopw]  # remove stopwords
    lemmas = preprocess_tokenized(lemmas)                       # removes non alphanumeric from each word
    return set(lemmas)

def spacy_lemmatize(sentence):
    doc = nlp(sentence)
    return set([token.lemma_ for token in doc if token.lemma_ not in stopw])

def pos_wn(pair):
    word = pair[0].lower()
    tag = tags.get(pair[1][:2].upper())
    if tag:
        word = wnl.lemmatize(word, pos=tag)
    return word

def synset(POS, synsets = {}):
    keys = []
    for pair in POS:
        if pair[0] in stopw:
            pass
        tag = tags.get(pair[1])
        if tag:
            synset = wordnet.synsets(pair[0], tag)
            if synset:
                synsets[pair[0]] = (synset[0], synset[0].pos())
                keys.append(pair[0])
    
    return synsets, keys

def word_ngrams(tokens_list, n):
    #tokens_list = set([w for w in tokens_list if not w.replace('.', '', 1).isdigit()]) # remove digits from ngrams
    return set(nltk.ngrams(tokens_list, n))

def char_ngrams(tokens_list, n):
    #tokens_list = set([w for w in tokens_list if not w.replace('.', '', 1).isdigit()]) # remove digits from ngrams
    return set(re.findall(fr"(?=([^\W_]{{{n}}}))", ' '.join(tokens_list)))


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


def lesk(sentence, pairs):
    synsets = [nltk.wsd.lesk(sentence, pair[0], pos=tags.get(pair[1][:2].upper())) for pair in pairs if
                   pair[0].lower() not in stopw]  # skip if the word is stopword else use lesk
    synsets = [syn.name() for syn in synsets if syn]  # we ignore the words without meaning
    return set(synsets)


def NE_nltk(pair):
    chunks = nltk.ne_chunk(pair, binary=False)  # chunk it 
    triads = nltk.tree2conlltags(chunks)        # get the triads with the token, pos tag, and if its named entity
    text = []
    for triad in triads:
        token = re.sub(r'[^\w\s]', '', triad[0]).lower()  # remove punctuaction inside each token
        if token in stopw or re.match(r'^[_\W]+$', token) or token == '':  # if its stopword, non-alphanumeric token, or empty we skip
            continue
        if triad[2][0] == 'O': # if not named enitity we append to text
            text.append(token)
        elif triad[2][0] == 'B': # if named entity we append to text as well
            text.append(token)
        else: # if the named entity continues, we concatanate it with the last string
            text[-1] += ' ' + token
    return set(text)


def syntactic_role_sim(synsets, keys1, keys2, method='lch'):
    sim = []
    for w1, w2 in list(itertools.product(keys1, keys2)):
        if w1 == w2:
            sim.append(1)
            continue

        syn1, tag1 = synsets[w1]
        syn2, tag2 = synsets[w1]

        if tag1 != tag2:
            continue

        try:
            if method == 'lch':
                if tag1 == tag2:
                    s = syn1.lch_similarity(syn2)
                    if s:
                        sim.append(s/syn1.lch_similarity(syn1))
                    else:
                        sim.append(0)
                else:
                    sim.append(0)
            elif method == 'wup':
                s = syn1.wup_similarity(syn2)
                sim.append(s) if s else sim.append(0)
            elif method == 'path':
                s = syn1.path_similarity(syn2)
                sim.append(s) if s else sim.append(0)
            elif method == 'lin':
                if tag1 == tag2 and tag1 in ['n', 'v']:
                    s = syn1.lin_similarity(syn2, brown_ic)
                    sim.append(s) if s else sim.append(0)
                else:
                    sim.append(0)
        except:
            sim.append(0)

    if len(sim) > 0:
        return sum(sim)/len(sim)
    else:
        return 0

# Features from the paper "TakeLab: Systems for Measuring Semantic Text Similarity"
def wawo_score(synset, keys1, keys2):
    pwn = 0
    for w1 in keys1:
        if w1 in keys2:
            pwn += 1
        else:
            syn1 = synset[w1][0]
            pwn += max([syn1.path_similarity(synset[w2][0], brown_ic) for w2 in keys2])
    return pwn

def wordnet_augmented_word_overlap(synset, keys1, keys2):
    if len(keys2) == 0 or len(keys1) == 0:
        return 0
    pwn1 = wawo_score(synset, keys1, keys2)/len(keys2)
    pwn2 = wawo_score(synset, keys2, keys1)/len(keys1)
    return 2*pwn1*pwn2/(pwn1 + pwn2) # harmonic mean


def weighted_word_overlap(synset, keys1, keys2):
    s1_s2 = set(keys1).intersection(set(keys2))
    numerator = 0
    if len(s1_s2) == 0:
        return 0
    
    numerator = sum([information_content(synset[w][0], brown_ic) for w in s1_s2 if synset[w][1] in ['v', 'n'] and information_content(synset[w][0], brown_ic) < 1e100])

    wwc1 = sum([information_content(synset[w][0], brown_ic) for w in keys1 if synset[w][1] in ['v', 'n'] and information_content(synset[w][0], brown_ic) < 1e100])
    wwc2 = sum([information_content(synset[w][0], brown_ic) for w in keys2 if synset[w][1] in ['v', 'n'] and information_content(synset[w][0], brown_ic) < 1e100])

    if wwc1 == 0 or wwc2 == 0 or numerator == 0:
        return 0
    
    wwc1 = numerator / wwc1
    wwc2 = numerator / wwc2
    return 2*wwc1*wwc2/(wwc1 + wwc2) # harmonic mean


def cosine(v1, v2):
        """ cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
        
        denom = (norm(v1) * norm(v2))
        if denom == 0:
            return 0
        return float(dot(v1, v2) / denom)

def vector_space_sentence(synset, keys1, keys2):
    s1_s2 = set(keys1).union(set(keys2))

    u1 = [0] * len(s1_s2)
    u2 = [0] * len(s1_s2)

    u1_ic = [0] * len(s1_s2)
    u2_ic = [0] * len(s1_s2)

    for i, w in enumerate(s1_s2):
        if w in keys1:
            u1[i] += 1

            if synset[w][1] in ['v', 'n']:
                ic = information_content(synset[w][0], brown_ic)
                if ic > 1e100:
                    u1_ic[i] = 0
                else:
                    u1_ic[i] += ic


        if w in keys2:
            u2[i] += 1

            if synset[w][1] in ['v', 'n']:
                ic = information_content(synset[w][0], brown_ic)
                if ic > 1e100:
                    u2_ic[i] = 0
                else:
                    u2_ic[i] += ic
    
    return cosine(u1, u2), cosine(u1_ic, u2_ic)


def function_word_similarity(tokens1, tokens2):
    x1 = [0] * len(stopw)
    x2 = [0] * len(stopw)
    for i, w in enumerate(stopw):
        if w in tokens1:
            x1[i] += 1
        if w in tokens2:
            x2[i] += 1
        
    if sum(x1) == 0 or sum(x2) == 0:
        return 0
    return pearsonr(x1, x2)[0]


        
def number_features(tokens1, tokens2):
    n1 = set([w for w in tokens1 if w.replace('.', '', 1).isdigit()])
    n2 = set([w for w in tokens2 if w.replace('.', '', 1).isdigit()])

    number_log = math.log(1 + len(n1) + len(n2))

    if (len(n1) + len(n2) == 0):
        number_intersection = 0
    else:
        number_intersection = 2 * len(n1.intersection(n2))/(len(n1) + len(n2))
    
    number_bool = int(n1.issubset(n2) or n2.issubset(n1))

    return number_log, number_intersection, number_bool


def levenshtein_distance(l1, l2):
    sent1 = ' '.join(l1)
    sent2 = ' '.join(l2)
    return nltk.edit_distance(sent1, sent2) / max(len(sent1), len(sent2))

def hamming(l1, l2):
    sent1 = ' '.join(l1)
    sent2 = ' '.join(l2)
    hamming_distance = hamming(sent1, sent2)
    return hamming_distance / max(len(sent1), len(sent2))


def num_verbs(pos1, pos2):
    count_v1 = len([v for v in pos1 if v[1].startswith('V')])
    count_v2 = len([v for v in pos2 if v[1].startswith('V')])
    if count_v1 == 0 and count_v2 == 0 or count_v1 == count_v2:
        return 1
    return 1 - (abs(count_v1 - count_v2) / (count_v1 + count_v2))


def num_nouns(pos1, pos2):
    count_n1 = len([n for n in pos1 if n[1].startswith('N')])
    count_n2 = len([n for n in pos2 if n[1].startswith('N')])
    if count_n1 == 0 and count_n2 == 0 or count_n1 == count_n2:
        return 1
    return 1 - (abs(count_n1 - count_n2) / (count_n1 + count_n2))


def num_adjs(pos1, pos2):
    count_a1 = len([a for a in pos1 if a[1].startswith('J')])
    count_a2 = len([a for a in pos2 if a[1].startswith('J')])
    if count_a1 == 0 and count_a2 == 0 or count_a1 == count_a2:
        return 1
    return 1 - (abs(count_a1 - count_a2) / (count_a1 + count_a2))


def num_advs(pos1, pos2):
    count_adv1 = len([a for a in pos1 if a[1].startswith('R')])
    count_adv2 = len([a for a in pos2 if a[1].startswith('R')])
    if count_adv1 == 0 and count_adv2 == 0 or count_adv1 == count_adv2:
        return 1
    return 1 - (abs(count_adv1 - count_adv2) / (count_adv1 + count_adv2))

def jaccard_empty(set1, set2):
    if len(set1) == 0 or len(set2) == 0:
        return 0
    return jaccard_distance(set1, set2)

def overlap(set1, set2):
    denom = len(set1.intersection(set2))
    if denom == 0:
        return 0
    return 2*pow(len(set1)/denom + len(set2)/denom, -1)


def dice_empty(set1, set2):
    if len(set1) + len(set2) == 0:
        return 0
    return 2*len(set1.intersection(set2))/(len(set1) + len(set2))

def similarity(set1, set2, sim_func = 'jaccard_empty'):
    return [1 - globals()[sim_func](s[0], s[1]) for s in zip(set1, set2)]


class Features:

    def __init__(self, data):
        self.pair1 = data['Sentence 1']
        self.pair2 = data['Sentence 2']

        self.features = {
            'tokens1': [], 'tokens2': [],
            'lemmas1': [], 'lemmas2': [],
            'tokens1_sw': [], 'tokens2_sw': [],
            'lemmas1_sw': [], 'lemmas2_sw': [],

            'lemmas1_spacy': [], 'lemmas2_spacy': [],

            'NE1': [], 'NE2': [],
            'lesk1': [], 'lesk2': [],
            'syns1': [], 'syns2': [],

            'word_ngrams1_1': [], 'word_ngrams1_sw_1': [],
            'word_ngrams1_2': [], 'word_ngrams1_sw_2': [],
            'word_ngrams1_3': [], 'word_ngrams1_sw_3': [],
            'word_ngrams1_4': [], 'word_ngrams1_sw_4': [],
            'word_ngrams2_1': [], 'word_ngrams2_sw_1': [],
            'word_ngrams2_2': [], 'word_ngrams2_sw_2': [],
            'word_ngrams2_3': [], 'word_ngrams2_sw_3': [],
            'word_ngrams2_4': [], 'word_ngrams2_sw_4': [],

            'char_ngrams1_2': [], 'char_ngrams1_sw_2': [],
            'char_ngrams1_3': [], 'char_ngrams1_sw_3': [],
            'char_ngrams1_4': [], 'char_ngrams1_sw_4': [],
            'char_ngrams2_2': [], 'char_ngrams2_sw_2': [],
            'char_ngrams2_3': [], 'char_ngrams2_sw_3': [],
            'char_ngrams2_4': [], 'char_ngrams2_sw_4': [],

            'lcs_subsequence': [], 'lcs_subsequence_sw': [],
            'lcs_substring': [], 'lcs_substring_sw': [],

            'lch_sim': [],
            'wup_sim': [],
            'lin_sim': [],
            'path_sim': [],

            'verbs_diff': [],
            'nouns_diff': [],
            'adjs_diff': [],
            'advs_diff': [],

            'synets1_lemmas': [],
            'synets2_lemmas': [],

            'WAWO': [],
            'WWO': [],

            'sorensen_dice': [],
            'levenshtein': [],

            'vector_space_sentence': [],
            'vector_space_sentence_ic': [],

            'number_log': [],
            'number_intersection': [],
            'number_bool': [],

            'function_word_similarity': [],

        }
        self.extract()

    def extract(self):
        for s1, s2 in zip(self.pair1, self.pair2):
            # Preprocess the data
            s1 = preprocess(s1)
            s2 = preprocess(s2)

            # Tokenize and obtain the POS-tags
            tokens1, pos1, tokens_list1 = tokenize(s1)
            tokens2, pos2, tokens_list2 = tokenize(s2)

            # Tokenize removing stopwords
            tokens1_sw, _, tokens_list1_sw = tokenize(s1, sw = True) # removes stopwords
            tokens2_sw, _, tokens_list2_sw = tokenize(s2, sw = True) # removes stopwords

            # From the POS-tag lemmatize
            lemmas1 = lemmatize(pos1)
            lemmas2 = lemmatize(pos2)

            lemmas1_spacy = spacy_lemmatize(s1)
            lemmas2_spacy = spacy_lemmatize(s2)

            # Lemmatize removing stopwords
            lemmas1_sw = lemmatize(pos1, sw = True)
            lemmas2_sw = lemmatize(pos2, sw = True)

            # Named entities
            self.features['NE1'].append(NE_nltk(pos1))
            self.features['NE2'].append(NE_nltk(pos2))

            # LESK
            self.features['lesk1'].append(lesk(s1, pos1))
            self.features['lesk2'].append(lesk(s2, pos2))

            # Longest common subsequence
            self.features['lcs_subsequence'].append(longest_common_subsequence(tokens1, tokens2))
            self.features['lcs_subsequence_sw'].append(longest_common_subsequence(tokens1_sw, tokens2_sw))
            
            # Longest common substring
            self.features['lcs_substring'].append(longest_common_substring(tokens1, tokens2))
            self.features['lcs_substring_sw'].append(longest_common_substring(tokens1_sw, tokens2_sw))

            # Function word Similarity
            self.features['function_word_similarity'].append(function_word_similarity(tokens1, tokens2))

            # Extract Synsets
            syns, keys1 = synset(pos1)
            syns, keys2 = synset(pos2, syns)

            # Syntactic role similarity
            self.features['lch_sim'].append(syntactic_role_sim(syns, keys1, keys2, method='lch'))
            self.features['wup_sim'].append(syntactic_role_sim(syns, keys1, keys2, method='wup'))
            self.features['lin_sim'].append(syntactic_role_sim(syns, keys1, keys2, method='lin'))
            self.features['path_sim'].append(syntactic_role_sim(syns, keys1, keys2, method='path'))

            self.features['WAWO'].append(wordnet_augmented_word_overlap(syns, keys1, keys2))
            self.features['WWO'].append(weighted_word_overlap(syns, keys1, keys2))

            self.features['levenshtein'].append(levenshtein_distance(lemmas1, lemmas2))

            vss, vss_ic = vector_space_sentence(syns, keys1, keys2)
            self.features['vector_space_sentence'].append(vss)
            self.features['vector_space_sentence_ic'].append(vss_ic)

            #print(tokens1)
            number_log, number_intersection, number_bool = number_features(tokens1, tokens2)
            self.features['number_log'].append(number_log)
            self.features['number_intersection'].append(number_intersection)
            self.features['number_bool'].append(number_bool)

            # Word n-grams
            for n in range(1, 5):
                self.features['word_ngrams1_' + str(n)].append(word_ngrams(tokens_list1, n))
                self.features['word_ngrams2_' + str(n)].append(word_ngrams(tokens_list2, n))
                self.features['word_ngrams1_sw_' + str(n)].append(word_ngrams(tokens_list1_sw, n))
                self.features['word_ngrams2_sw_' + str(n)].append(word_ngrams(tokens_list2_sw, n))

            # Character n-grams
            for n in range(2, 5):
                self.features['char_ngrams1_' + str(n)].append(char_ngrams(tokens_list1, n))
                self.features['char_ngrams2_' + str(n)].append(char_ngrams(tokens_list2, n))
                self.features['char_ngrams1_sw_' + str(n)].append(char_ngrams(tokens_list1_sw, n))
                self.features['char_ngrams2_sw_' + str(n)].append(char_ngrams(tokens_list2_sw, n))

            self.features['verbs_diff'].append(num_verbs(pos1, pos2))
            self.features['nouns_diff'].append(num_nouns(pos1, pos2))
            self.features['adjs_diff'].append(num_adjs(pos1, pos2))
            self.features['advs_diff'].append(num_advs(pos1, pos2))

            # We store the information in vectors to compute the Pearson Correlation
            self.features['tokens1'].append(tokens1)
            self.features['tokens2'].append(tokens2)
            self.features['lemmas1'].append(lemmas1)
            self.features['lemmas2'].append(lemmas2)

            self.features['lemmas1_spacy'].append(lemmas1_spacy)
            self.features['lemmas2_spacy'].append(lemmas2_spacy)

            self.features['tokens1_sw'].append(tokens1_sw)
            self.features['tokens2_sw'].append(tokens2_sw)
            self.features['lemmas1_sw'].append(lemmas1_sw)
            self.features['lemmas2_sw'].append(lemmas2_sw)

            self.features['synets1_lemmas'] = keys1
            self.features['synets2_lemmas'] = keys2


    def extract_all(self):
        return pd.DataFrame({
            'NE NLTK': similarity(self.features['NE1'], self.features['NE2']),
            'Tokens Jac. Sim.': similarity(self.features['tokens1'], self.features['tokens2']),
            'Tokens (stop-words) Jac. Sim.': similarity(self.features['tokens1_sw'], self.features['tokens2_sw']),
            'Lemmas Jac. Sim.': similarity(self.features['lemmas1'], self.features['lemmas2']),
            'Lemmas (stop-words) Jac. Sim.': similarity(self.features['lemmas1_sw'], self.features['lemmas2_sw']),
            'Lemmas (Spacy) Jac. Sim.': similarity(self.features['lemmas1_spacy'], self.features['lemmas2_spacy']),

            'Unigrams Jac. Sim.': similarity(self.features['word_ngrams1_1'], self.features['word_ngrams2_1']),
            'Bigrams Jac. Sim.': similarity(self.features['word_ngrams1_2'], self.features['word_ngrams2_2']),
            'Trigrams Jac. Sim.': similarity(self.features['word_ngrams1_3'], self.features['word_ngrams2_3']),
            'Fourgrams Jac. Sim.': similarity(self.features['word_ngrams1_4'], self.features['word_ngrams2_4']),
            'Unigrams (stop-words) Jac. Sim.': similarity(self.features['word_ngrams1_sw_1'], self.features['word_ngrams2_sw_1']),
            'Bigrams (stop-words) Jac. Sim.': similarity(self.features['word_ngrams1_sw_2'], self.features['word_ngrams2_sw_2']),
            'Trigrams (stop-words) Jac. Sim.': similarity(self.features['word_ngrams1_sw_3'], self.features['word_ngrams2_sw_3']),
            'Fourgrams (stop-words) Jac. Sim.': similarity(self.features['word_ngrams1_sw_4'], self.features['word_ngrams2_sw_4']),

            'Char Bigrams Jac. Sim.': similarity(self.features['char_ngrams1_2'], self.features['char_ngrams2_2']),
            'Char Trigrams Jac. Sim.': similarity(self.features['char_ngrams1_3'], self.features['char_ngrams2_3']),
            'Char Fourgrams Jac. Sim.': similarity(self.features['char_ngrams1_4'], self.features['char_ngrams2_4']),
            'Char Bigrams (stop-words) Jac. Sim.': similarity(self.features['char_ngrams1_sw_2'], self.features['char_ngrams2_sw_2']),
            'Char Trigrams (stop-words) Jac. Sim.': similarity(self.features['char_ngrams1_sw_3'], self.features['char_ngrams2_sw_3']),
            'Char Fourgrams (stop-words) Jac. Sim.': similarity(self.features['char_ngrams1_sw_4'], self.features['char_ngrams2_sw_4']),

            'Overlap Unigram': similarity(self.features['word_ngrams1_1'], self.features['word_ngrams2_1'], 'overlap'),
            'Overlap Bigrams': similarity(self.features['word_ngrams1_2'], self.features['word_ngrams2_2'], 'overlap'),
            'Overlap Trigrams': similarity(self.features['word_ngrams1_3'], self.features['word_ngrams2_3'], 'overlap'),
            'Overlap Unigrams (stop-words)': similarity(self.features['word_ngrams1_sw_1'], self.features['word_ngrams2_sw_1'], 'overlap'),
            'Overlap Bigrams (stop-words)': similarity(self.features['word_ngrams1_sw_2'], self.features['word_ngrams2_sw_2'], 'overlap'),
            'Overlap Trigrams (stop-words)': similarity(self.features['word_ngrams1_sw_3'], self.features['word_ngrams2_sw_3'], 'overlap'),
            
            'Lesk Jac. Sim.': similarity(self.features['lesk1'], self.features['lesk2']),

            'NE NLTK': similarity(self.features['NE1'], self.features['NE2'], 'dice_empty'),
            'Tokens Dice Sim.': similarity(self.features['tokens1'], self.features['tokens2'], 'dice_empty'),
            'Tokens (stop-words) Dice Sim.': similarity(self.features['tokens1_sw'], self.features['tokens2_sw'], 'dice_empty'),
            'Lemmas Dice Sim.': similarity(self.features['lemmas1'], self.features['lemmas2'], 'dice_empty'),
            'Lemmas (stop-words) Dice Sim.': similarity(self.features['lemmas1_sw'], self.features['lemmas2_sw'], 'dice_empty'),
            'Lemmas (Spacy) Dice Sim.': similarity(self.features['lemmas1_spacy'], self.features['lemmas2_spacy'], 'dice_empty'),

            'Unigrams Dice Sim.': similarity(self.features['word_ngrams1_1'], self.features['word_ngrams2_1'], 'dice_empty'),
            'Bigrams Dice Sim.': similarity(self.features['word_ngrams1_2'], self.features['word_ngrams2_2'], 'dice_empty'),
            'Trigrams Dice Sim.': similarity(self.features['word_ngrams1_3'], self.features['word_ngrams2_3'], 'dice_empty'),
            'Fourgrams Dice Sim.': similarity(self.features['word_ngrams1_4'], self.features['word_ngrams2_4'], 'dice_empty'),
            'Unigrams (stop-words) Dice Sim.': similarity(self.features['word_ngrams1_sw_1'], self.features['word_ngrams2_sw_1'], 'dice_empty'),
            'Bigrams (stop-words) Dice Sim.': similarity(self.features['word_ngrams1_sw_2'], self.features['word_ngrams2_sw_2'], 'dice_empty'),
            'Trigrams (stop-words) Dice Sim.': similarity(self.features['word_ngrams1_sw_3'], self.features['word_ngrams2_sw_3'], 'dice_empty'),
            'Fourgrams (stop-words) Dice Sim.': similarity(self.features['word_ngrams1_sw_4'], self.features['word_ngrams2_sw_4'], 'dice_empty'),

            'Char Bigrams Dice Sim.': similarity(self.features['char_ngrams1_2'], self.features['char_ngrams2_2'], 'dice_empty'),
            'Char Trigrams Dice Sim.': similarity(self.features['char_ngrams1_3'], self.features['char_ngrams2_3'], 'dice_empty'),
            'Char Fourgrams Dice Sim.': similarity(self.features['char_ngrams1_4'], self.features['char_ngrams2_4'], 'dice_empty'),
            'Char Bigrams (stop-words) Dice Sim.': similarity(self.features['char_ngrams1_sw_2'], self.features['char_ngrams2_sw_2'], 'dice_empty'),
            'Char Trigrams (stop-words) Dice Sim.': similarity(self.features['char_ngrams1_sw_3'], self.features['char_ngrams2_sw_3'], 'dice_empty'),
            'Char Fourgrams (stop-words) Dice Sim.': similarity(self.features['char_ngrams1_sw_4'], self.features['char_ngrams2_sw_4'], 'dice_empty'),
            
            'Lesk Jac. Sim.': similarity(self.features['lesk1'], self.features['lesk2']),

            'Longest Common Subsequence': self.features['lcs_subsequence'],
            'Longest Common Subsequence (stop-words)': self.features['lcs_subsequence_sw'],
            'Longest Common Substring': self.features['lcs_substring'],
            'Longest Common Substring (stop-words)': self.features['lcs_substring_sw'],

            'Leacock-Chodorow Sim.': self.features['lch_sim'],
            'Path Sim.': self.features['wup_sim'],
            'Wu-Palmer Sim.': self.features['lin_sim'],
            'Lin Sim.': self.features['path_sim'],


            'Levenshtein Distance': self.features['levenshtein'],
            ## 'Hamming Distance': self.ham_dist,
            
            '# of Verb Tags': self.features['verbs_diff'],
            '# of Noun Tags': self.features['nouns_diff'],
            '# of Adjective Tags': self.features['adjs_diff'],
            '# of Adverb Tags': self.features['advs_diff'],

            'WAWO': self.features['WAWO'],
            'WWO': self.features['WWO'],

            'Vector Space Sentence': self.features['vector_space_sentence'],
            'Vector Space Sentence IC': self.features['vector_space_sentence_ic'],

            'Numeric Feature: Log': self.features['number_log'],
            'Numeric Feature: Intersection ': self.features['number_intersection'],
            'Numeric Feature Bool': self.features['number_bool'],

            'Function Word Similarity': self.features['function_word_similarity'],

        })
