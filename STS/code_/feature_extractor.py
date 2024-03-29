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
from nltk.parse.corenlp import CoreNLPDependencyParser
import neuralcoref

nltk.download('words')
nltk.download('omw-1.4')
nltk.download('wordnet_ic')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('conll2000')

# parser for the Stanford Dependency Parser, you need to be running it!
parser = CoreNLPDependencyParser(url='http://localhost:9000/')

# set the local for unicode
setlocale(LC_NUMERIC, 'en_US.UTF-8')

# initialize spacy and neuralcoref
nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

# set corpus
semcor_ic = wordnet_ic.ic('ic-semcor.dat')
brown_ic = wordnet_ic.ic('ic-brown.dat')

# initialize stememr
ps = PorterStemmer()

# get stopwords
stopw = set(nltk.corpus.stopwords.words('english')) 

# tags for wordnet
tags = {'NN': wordnet.NOUN,
        'VB': wordnet.VERB,
        'JJ': wordnet.ADJ,
        'RB': wordnet.ADV}

# initialize wordnet lemmatizer
wnl = WordNetLemmatizer()

###########################
# PREPROCESSING FUNCTIONS #
###########################

def preprocess(sentence):
    sentence = unidecode(sentence)                    # converts to ascii everything
    sentence = re.sub(r"(-\b|\b-|/)", "", sentence)   # remove hyphens and forward slashes
    sentence = re.sub(r"$US", "$", sentence)          # normalize dollar values, no other money symbol appears in the train dataset
    sentence = re.sub(r"<\.?(.*?)>", r"\1", sentence) # matches the interior string of the format <XYZ> and returns XYZ
    
    # substitute to full word
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
            clean_tokens.append(token.lower())              # convert to lower case if numeric
        else:
            try:
                # if the token is numeric, remove the commas separating thousands
                # convert it to float, and back to string
                numeric_str = str(round(atof(token), 1))
                if numeric_str[-2:] == '.0':                # if it is .0, put it as integer
                    clean_tokens.append(numeric_str[:-2])   
                else:
                    clean_tokens.append(numeric_str)        # append the float as string
            except:
                pass
    return clean_tokens

#############################
# FUNCTIONS TO GET FEATURES #
#############################

def tokenize(sentence, sw=False):
    tokens = nltk.word_tokenize(sentence)                       # tokenize the sentence
    pairs = nltk.pos_tag(tokens)                                # get the POS-tag of the tokens
    if sw:                                                      
        tokens = [w for w in tokens if not w.lower() in stopw]  # remove stopwords

    tokens = preprocess_tokenized(tokens)                       # apply the preprocess of numbers and to lower
    return set(tokens), pairs, tokens

def lemmatize(sentences_POS, sw=False):
    lemmas = [pos_wn(pair) for pair in sentences_POS]           # obtain the lemmas
    if sw:
        lemmas = [w for w in lemmas if not w.lower() in stopw]  # remove stopwords
    lemmas = preprocess_tokenized(lemmas)                       # apply the preprocess of numbers and to lower
    return set(lemmas)

def spacy_lemmatize(sentence):
    doc = nlp(sentence)                                         # lemmatize using spacy
    return set([token.lemma_.lower() for token in doc if token.lemma_ not in stopw]) # basic filtering of stopwords

def pos_wn(pair):
    # Function that given a pair of word and POS-tag, will lemmatize it using Wordnet
    word = pair[0].lower()
    tag = tags.get(pair[1][:2].upper())
    if tag:
        word = wnl.lemmatize(word, pos=tag)
    return word

def stemming(sentence):
    # function that given a tokenized sentence, stems it
    return set([ps.stem(w) for w in sentence])

def NE_nltk(pair):
    chunks = nltk.ne_chunk(pair, binary=False)  # chunk it 
    triads = nltk.tree2conlltags(chunks)        # get the triads with the token, pos tag, and if its named entity
    text = []
    for triad in triads:
        token = re.sub(r'[^\w\s]', '', triad[0]).lower()  # remove symbols inside each token
        if token in stopw or re.match(r'^[_\W]+$', token) or token == '':  # if its stopword, non-alphanumeric token, or empty we skip
            continue
        if triad[2][0] == 'O': # if not named entity we append to text
            text.append(token)
        elif triad[2][0] == 'B': # if named entity we append to text as well
            text.append(token)
        else: # if the named entity continues, we concatenate it with the last string
            text[-1] += ' ' + token
    return set(text)

def NE_spacy(sentence):
  tokens = nlp(sentence) # lemmatize using Spacy to get the NE
  text = []
  with tokens.retokenize() as retokenizer: # extract the lemmas and retokenize to get the NE's
    token = [t for t in tokens]
    for ent in tokens.ents:
        retokenizer.merge(tokens[ent.start:ent.end], 
                          attrs={"LEMMA": " ".join([tokens[i].text for i in range(ent.start, ent.end)])})
  
  text = [] # small preprocessing before returning the object
  for token in tokens:
    token = re.sub(r'[^\w\s]', '', token.text).lower() # remove punctuaction inside each token
    if token in stopw or re.match(r'^[_\W]+$', token) or token == '': # if its stopword, non-alphanumeric token, or empty we skip
      continue
    text.append(token)
  return set(text)

def coreference(sentence):
    # function that uses neuralcoref to coreference, we lemmatize and tokenize to get features
    doc = nlp(sentence)
    s = doc._.coref_resolved                # solve the references

    _, pos, _ = tokenize(s)                 # get POS tags
    tokens, _, _ = tokenize(s, sw = True)   # Tokenize removing stopwords
    lemmas = lemmatize(pos, sw = True)      # From the POS-tag lemmatize

    return set(tokens), set(lemmas)


###########
# Synsets #
###########

def synset(POS, synsets = {}):
    # function that will obtain synsets from a vector of POS-tags
    keys = []
    for pair in POS:
        if pair[0] in stopw:
            pass
        tag = tags.get(pair[1])
        if tag:
            synset = wordnet.synsets(pair[0], tag) # get the synset from wordnet
            if synset:
                synsets[pair[0]] = (synset[0], synset[0].pos()) # if it exists, append the pair to the dict
                keys.append(pair[0])                            # return the pair that existed, as a key
    
    return synsets, keys

def word_ngrams(tokens_list, n):
    # function that gets n-grams from words

    # we tried filtering the digits, but works worse:
    #tokens_list = set([w for w in tokens_list if not w.replace('.', '', 1).isdigit()]) # remove digits from ngrams
    return set(nltk.ngrams(tokens_list, n))

def char_ngrams(tokens_list, n):
    # function that gets n-grams from characters

    # we tried filtering the digits, but works worse:
    #tokens_list = set([w for w in tokens_list if not w.replace('.', '', 1).isdigit()]) # remove digits from ngrams
    return set(re.findall(fr"(?=([^\W_]{{{n}}}))", ' '.join(tokens_list)))

def triplet_parser(sentence):
    # Function that will get the governor, dependee, and dependent from the Stanford Dependency Parser
    parse, = parser.raw_parse(sentence)
    triples = []
    for governor, dep, dependent in parse.triples():
        if dep == 'punct' or governor[0].lower() in stopw: # small preprocessing to remove punctuation and to convert to lower and stopwords
            continue
        triples.append(( (governor[0].lower(), governor[1]), dep, (dependent[0].lower(), dependent[1])))
    return set(triples)

def longest_common_subsequence(t1, t2):
    # function that finds the longest common subsequence from two token lists
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
    # function that finds the longest common SUBSTRING from two token lists
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
    # function that finds computes the LESK of a sentence from a sentence and POS tags 
    synsets = [nltk.wsd.lesk(sentence, pair[0], pos=tags.get(pair[1][:2].upper())) for pair in pairs if
                   pair[0].lower() not in stopw]  # skip if the word is stopword else use lesk
    synsets = [syn.name() for syn in synsets if syn]  # we ignore the words without meaning
    return set(synsets)


def syntactic_role_sim(synsets, keys1, keys2, method='lch'):
    # function that computes the syntactic role similarities from the synsets and keys found
    sim = []
    for w1, w2 in list(itertools.product(keys1, keys2)):
        if w1 == w2: # if they are the same, append 1
            sim.append(1)
            continue

        syn1, tag1 = synsets[w1] # get the synset of the word we are currently observing
        syn2, tag2 = synsets[w1]

        if tag1 != tag2: # if the tags are not the same, do not compare, skip
            continue

        try:
            if method == 'lch':
                s = syn1.lch_similarity(syn2)
                if s: 
                    sim.append(s/syn1.lch_similarity(syn1)) # we normalize lch with the same synset
                else: 
                    sim.append(0)                           # if the lch does not exist append a 0
            elif method == 'wup':                           
                s = syn1.wup_similarity(syn2)               # compute WU-Palmer similarity
                sim.append(s) if s else sim.append(0)
            elif method == 'path':
                s = syn1.path_similarity(syn2)              # Compute Path similarity
                sim.append(s) if s else sim.append(0)
            elif method == 'lin':
                if tag1 in ['n', 'v']:                      # we can only do if the tags are 'n' or 'v' due to the corpus
                    s = syn1.lin_similarity(syn2, brown_ic) # compute Lin Similarity
                    sim.append(s) if s else sim.append(0)
                else:
                    sim.append(0)
        except:
            sim.append(0)

    # if similarities were found, return the mean
    if len(sim) > 0:
        return sum(sim)/len(sim)
    else:
        return 0

def wawo_score(synset, keys1, keys2):
    # Wordnet Augmented Word score
    pwn = 0
    for w1 in keys1:        
        if w1 in keys2:         # if word is found in the other keys, add 1
            pwn += 1
        else:                   # else, add depending the max of what is found as information content, for each of keys2
            syn1 = synset[w1][0]
            pwn += max([syn1.path_similarity(synset[w2][0], brown_ic) for w2 in keys2])
    return pwn

def wordnet_augmented_word_overlap(synset, keys1, keys2):
    # Wordnet Augmented Word Score
    # Feature from the paper "TakeLab: Systems for Measuring Semantic Text Similarity"
    if len(keys2) == 0 or len(keys1) == 0:
        return 0
    pwn1 = wawo_score(synset, keys1, keys2)/len(keys2)
    pwn2 = wawo_score(synset, keys2, keys1)/len(keys1)
    return 2*pwn1*pwn2/(pwn1 + pwn2) # harmonic mean


def weighted_word_overlap(synset, keys1, keys2):
    # Weighted Word Overlap
    # Feature from the paper "TakeLab: Systems for Measuring Semantic Text Similarity"
    s1_s2 = set(keys1).intersection(set(keys2))             # get the intersection of keys
    numerator = 0
    if len(s1_s2) == 0:                                     # if the intersection is zero we stop
        return 0
    
    # get the information content of each case, since it can be almost zero frequency, the value returned can be very large. 
    # Thus, we filter those that are too large, as the values were useless.
    # we do it three times for the intersection, elements from one, and elements from the other.
    numerator = sum([information_content(synset[w][0], brown_ic) for w in s1_s2 if synset[w][1] in ['v', 'n'] and information_content(synset[w][0], brown_ic) < 1e100])
    if numerator == 0:
        return 0

    wwc1 = sum([information_content(synset[w][0], brown_ic) for w in keys1 if synset[w][1] in ['v', 'n'] and information_content(synset[w][0], brown_ic) < 1e100])
    if wwc1 == 0:
        return 0

    wwc2 = sum([information_content(synset[w][0], brown_ic) for w in keys2 if synset[w][1] in ['v', 'n'] and information_content(synset[w][0], brown_ic) < 1e100])
    if wwc2 == 0:
        return 0

    # once ended, compute the harmonic mean    
    wwc1 = numerator / wwc1
    wwc2 = numerator / wwc2
    return 2*wwc1*wwc2/(wwc1 + wwc2) 


def cosine(v1, v2):
    # computes the cosine similarity from 2 vectors. Computed as:
    # cosine  = ( V1 * V2 ) / ||V1|| x ||V2||
    denom = (norm(v1) * norm(v2))
    if denom == 0:
        return 0
    return float(dot(v1, v2) / denom)

def vector_space_sentence(synset, keys1, keys2):
    # function that computes the vector space sentence similarity
    # it does so in two ways
    # computes a bag of words for all the words that appear and sum the frequencies
    # 1. use cosine on this previous element
    # 2. use the information content to scale it, and use the cosine on this
    s1_s2 = set(keys1).union(set(keys2))

    # initialize the bag of words
    u1 = [0] * len(s1_s2)
    u2 = [0] * len(s1_s2)

    u1_ic = [0] * len(s1_s2)
    u2_ic = [0] * len(s1_s2)

    for i, w in enumerate(s1_s2):
        if w in keys1:      
            u1[i] += 1              # if key appears, add it to the BOW

            if synset[w][1] in ['v', 'n']:  # if the POS tag is V or N, corpus limitation, add information content to vector
                ic = information_content(synset[w][0], brown_ic)
                if ic > 1e100:              # if its too large means the frequency is close to 0, we rather just add 0
                    u1_ic[i] += 0
                else:
                    u1_ic[i] += ic


        # same for the other pair 
        if w in keys2:
            u2[i] += 1

            if synset[w][1] in ['v', 'n']:
                ic = information_content(synset[w][0], brown_ic)
                if ic > 1e100:
                    u2_ic[i] += 0
                else:
                    u2_ic[i] += ic
    
    # return the cosine similarity for each of the 2 different
    return cosine(u1, u2), cosine(u1_ic, u2_ic)


def function_word_similarity(tokens1, tokens2):
    # Function that computes the amount of 'function words' that appear. It does so by observing their frequency
    # function words are basically stopwords. We create a bag of stopwords and count their frequency
    x1 = [0] * len(stopw)
    x2 = [0] * len(stopw)
    for i, w in enumerate(stopw):
        if w in tokens1:
            x1[i] += 1
        if w in tokens2:
            x2[i] += 1
        
    if sum(x1) == 0 or sum(x2) == 0:
        return 0
    # there are different ways to compute the similarity, we use the Pearson Correlation
    return pearsonr(x1, x2)[0]

##################
# OTHER FEATURES #
##################

def number_features(tokens1, tokens2):
    # Function that extracts the "numeric" features from the sentences
    # it compares the strings in three different ways

    # extract only numeric elements
    n1 = set([w for w in tokens1 if w.replace('.', '', 1).isdigit()])
    n2 = set([w for w in tokens2 if w.replace('.', '', 1).isdigit()])

    # log the difference between the lengths
    number_log = math.log(1 + len(n1) + len(n2))

    # compute the intersection
    if (len(n1) + len(n2) == 0):
        number_intersection = 0
    else:
        number_intersection = 2 * len(n1.intersection(n2))/(len(n1) + len(n2))
    
    # see if one is subset of another
    number_bool = int(n1.issubset(n2) or n2.issubset(n1))

    return number_log, number_intersection, number_bool

def num_POS(pos1, pos2, POS):
    # function that comperes the amount of different POS-tag elements we find form one sentence and another
    count_v1 = len([v for v in pos1 if v[1].startswith(POS)]) # extract how many words have function = POS
    count_v2 = len([v for v in pos2 if v[1].startswith(POS)])
    if count_v1 == 0 and count_v2 == 0 or count_v1 == count_v2:
        return 1
    return 1 - (abs(count_v1 - count_v2) / (count_v1 + count_v2))

################
# SIMILARITIES #
################

def levenshtein_distance(l1, l2):
    # computes the levenshtein distance between two sets of lemmas
    sent1 = ' '.join(l1)
    sent2 = ' '.join(l2)
    return nltk.edit_distance(sent1, sent2) / max(len(sent1), len(sent2))

def hamming(l1, l2):
    # computes the levenshtein distance between two sets of lemmas
    sent1 = ' '.join(l1)
    sent2 = ' '.join(l2)
    hamming_distance = hamming(sent1, sent2)
    return hamming_distance / max(len(sent1), len(sent2))

def jaccard_empty(set1, set2):
    # computes the Jaccard similarity, if one has no elements, we return 0. we consider empty vs something not be similar
    if len(set1) == 0 or len(set2) == 0:
        return 0
    return jaccard_distance(set1, set2)

def overlap(set1, set2):
    # computes the overlap similarity, if the intersection is zero we return 0. No overlap
    denom = len(set1.intersection(set2))
    if denom == 0:
        return 0
    return 2*pow(len(set1)/denom + len(set2)/denom, -1)

def dice_empty(set1, set2):
    # computes the Dice similarity, if the sum is zero we return 0
    if len(set1) + len(set2) == 0:
        return 0
    return 2*len(set1.intersection(set2))/(len(set1) + len(set2))

def similarity(set1, set2, sim_func = 'jaccard_empty'):
    # function that given 2 sets, will compute the similarity we defined
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
            'porter1': [], 'porter2': [],

            'coref_tokens1': [], 'coref_tokens2': [],
            'coref_lemmas1': [], 'coref_lemmas2': [],

            'NE1': [], 'NE2': [],
            'NE1_spacy': [], 'NE2_spacy': [],

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

            'triplet_parser1': [], 'triplet_parser2': [],

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

            'number_tokens1': [], 'number_tokens2': [],
            'number_tokens1_sw': [], 'number_tokens2_sw': [],

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

            # Porter Stemmer
            self.features['porter1'].append(stemming(tokens_list1_sw))
            self.features['porter2'].append(stemming(tokens_list2_sw))

            # Coreference
            coref_tokens1, coref_lemmas1 = coreference(s1)
            coref_tokens2, coref_lemmas2 = coreference(s2)

            # Named entities
            self.features['NE1'].append(NE_nltk(pos1))
            self.features['NE2'].append(NE_nltk(pos2))
            self.features['NE1_spacy'].append(NE_spacy(s1))
            self.features['NE2_spacy'].append(NE_spacy(s2))

            # LESK
            self.features['lesk1'].append(lesk(s1, pos1))
            self.features['lesk2'].append(lesk(s2, pos2))

            # Dependency Parser (NEEDS STANFORD PARSER TO BE RUNNING)
            self.features['triplet_parser1'].append(triplet_parser(s1))
            self.features['triplet_parser2'].append(triplet_parser(s2))

            # Longest common subsequence
            self.features['lcs_subsequence'].append(longest_common_subsequence(tokens1, tokens2))
            self.features['lcs_subsequence_sw'].append(longest_common_subsequence(tokens1_sw, tokens2_sw))
            
            # Longest common substring
            self.features['lcs_substring'].append(longest_common_substring(tokens1, tokens2))
            self.features['lcs_substring_sw'].append(longest_common_substring(tokens1_sw, tokens2_sw))

            # Function word Frequency Similarity
            self.features['function_word_similarity'].append(function_word_similarity(tokens1, tokens2))

            # Extract Synsets
            syns, keys1 = synset(pos1)
            syns, keys2 = synset(pos2, syns)

            # Syntactic role similarity
            self.features['lch_sim'].append(syntactic_role_sim(syns, keys1, keys2, method='lch'))
            self.features['wup_sim'].append(syntactic_role_sim(syns, keys1, keys2, method='wup'))
            self.features['lin_sim'].append(syntactic_role_sim(syns, keys1, keys2, method='lin'))
            self.features['path_sim'].append(syntactic_role_sim(syns, keys1, keys2, method='path'))

            # WordNet-Augmented Word Overlap and Weighted Word Overlap
            self.features['WAWO'].append(wordnet_augmented_word_overlap(syns, keys1, keys2))
            self.features['WWO'].append(weighted_word_overlap(syns, keys1, keys2))

            # Levenshtein sim for lemmas
            self.features['levenshtein'].append(levenshtein_distance(lemmas1, lemmas2))

            # Vector Space Sentence Similarity using information content and without
            vss, vss_ic = vector_space_sentence(syns, keys1, keys2)
            self.features['vector_space_sentence'].append(vss)
            self.features['vector_space_sentence_ic'].append(vss_ic)

            # Numeric string Featurs
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

            # Stylistic dimensions
            self.features['verbs_diff'].append(num_POS(pos1, pos2, 'V'))
            self.features['nouns_diff'].append(num_POS(pos1, pos2, 'N'))
            self.features['adjs_diff'].append(num_POS(pos1, pos2, 'J'))
            self.features['advs_diff'].append(num_POS(pos1, pos2, 'R'))

            # Store the tokens, lemmas, etc. to compute the similarities
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

            self.features['coref_tokens1'].append(coref_tokens1)
            self.features['coref_tokens2'].append(coref_tokens2)
            self.features['coref_lemmas1'].append(coref_lemmas1)
            self.features['coref_lemmas2'].append(coref_lemmas2)


    def extract_all(self):
        return pd.DataFrame({
            'NE NLTK Jac.': similarity(self.features['NE1'], self.features['NE2']),
            'NE NLTK Dice': similarity(self.features['NE1'], self.features['NE2'], 'dice_empty'),
            'NE Spacy Jac.': similarity(self.features['NE1_spacy'], self.features['NE2_spacy']),
            'NE Spacy Dice': similarity(self.features['NE1_spacy'], self.features['NE2_spacy'], 'dice_empty'),

            'Tokens Jac.': similarity(self.features['tokens1'], self.features['tokens2']),
            'Tokens (stop-words) Jac.': similarity(self.features['tokens1_sw'], self.features['tokens2_sw']),
            'Tokens Dice': similarity(self.features['tokens1'], self.features['tokens2'], 'dice_empty'),
            'Tokens (stop-words) Dice': similarity(self.features['tokens1_sw'], self.features['tokens2_sw'], 'dice_empty'),

            'Lemmas Jac.': similarity(self.features['lemmas1'], self.features['lemmas2']),
            'Lemmas (stop-words) Jac.': similarity(self.features['lemmas1_sw'], self.features['lemmas2_sw']),
            'Lemmas (Spacy) Jac.': similarity(self.features['lemmas1_spacy'], self.features['lemmas2_spacy']),
            'Lemmas Dice': similarity(self.features['lemmas1'], self.features['lemmas2'], 'dice_empty'),
            'Lemmas (stop-words) Dice': similarity(self.features['lemmas1_sw'], self.features['lemmas2_sw'], 'dice_empty'),
            'Lemmas (Spacy) Dice': similarity(self.features['lemmas1_spacy'], self.features['lemmas2_spacy'], 'dice_empty'),

            'Stemming Jac.': similarity(self.features['porter1'], self.features['porter2']),
            'Stemming Dice': similarity(self.features['porter1'], self.features['porter2'], 'dice_empty'),

            'Lesk Jac.': similarity(self.features['lesk1'], self.features['lesk2']),
            'Lesk Dice': similarity(self.features['lesk1'], self.features['lesk2'], 'dice_empty'),

            'Coref. Tokens Jac.': similarity(self.features['coref_tokens1'], self.features['coref_tokens2']),
            'Coref. Tokens Dice': similarity(self.features['coref_tokens1'], self.features['coref_tokens2'], 'dice_empty'),

            'Coref. Lemmas Jac.': similarity(self.features['coref_tokens1'], self.features['coref_tokens2']),
            'Coref. Lemmas Dice': similarity(self.features['coref_lemmas1'], self.features['coref_lemmas2'], 'dice_empty'),

            'Unigrams Jac.': similarity(self.features['word_ngrams1_1'], self.features['word_ngrams2_1']),
            'Bigrams Jac.': similarity(self.features['word_ngrams1_2'], self.features['word_ngrams2_2']),
            'Trigrams Jac.': similarity(self.features['word_ngrams1_3'], self.features['word_ngrams2_3']),
            'Fourgrams Jac.': similarity(self.features['word_ngrams1_4'], self.features['word_ngrams2_4']),
            'Unigrams (stop-words) Jac.': similarity(self.features['word_ngrams1_sw_1'], self.features['word_ngrams2_sw_1']),
            'Bigrams (stop-words) Jac.': similarity(self.features['word_ngrams1_sw_2'], self.features['word_ngrams2_sw_2']),
            'Trigrams (stop-words) Jac.': similarity(self.features['word_ngrams1_sw_3'], self.features['word_ngrams2_sw_3']),
            'Fourgrams (stop-words) Jac.': similarity(self.features['word_ngrams1_sw_4'], self.features['word_ngrams2_sw_4']),

            'Unigrams Dice': similarity(self.features['word_ngrams1_1'], self.features['word_ngrams2_1'], 'dice_empty'),
            'Bigrams Dice': similarity(self.features['word_ngrams1_2'], self.features['word_ngrams2_2'], 'dice_empty'),
            'Trigrams Dice': similarity(self.features['word_ngrams1_3'], self.features['word_ngrams2_3'], 'dice_empty'),
            'Fourgrams Dice': similarity(self.features['word_ngrams1_4'], self.features['word_ngrams2_4'], 'dice_empty'),
            'Unigrams (stop-words) Dice': similarity(self.features['word_ngrams1_sw_1'], self.features['word_ngrams2_sw_1'], 'dice_empty'),
            'Bigrams (stop-words) Dice': similarity(self.features['word_ngrams1_sw_2'], self.features['word_ngrams2_sw_2'], 'dice_empty'),
            'Trigrams (stop-words) Dice': similarity(self.features['word_ngrams1_sw_3'], self.features['word_ngrams2_sw_3'], 'dice_empty'),
            'Fourgrams (stop-words) Dice': similarity(self.features['word_ngrams1_sw_4'], self.features['word_ngrams2_sw_4'], 'dice_empty'),

            'Char Bigrams Jac.': similarity(self.features['char_ngrams1_2'], self.features['char_ngrams2_2']),
            'Char Trigrams Jac.': similarity(self.features['char_ngrams1_3'], self.features['char_ngrams2_3']),
            'Char Fourgrams Jac.': similarity(self.features['char_ngrams1_4'], self.features['char_ngrams2_4']),
            'Char Bigrams (stop-words) Jac.': similarity(self.features['char_ngrams1_sw_2'], self.features['char_ngrams2_sw_2']),
            'Char Trigrams (stop-words) Jac.': similarity(self.features['char_ngrams1_sw_3'], self.features['char_ngrams2_sw_3']),
            'Char Fourgrams (stop-words) Jac.': similarity(self.features['char_ngrams1_sw_4'], self.features['char_ngrams2_sw_4']),

            'Char Bigrams Dice': similarity(self.features['char_ngrams1_2'], self.features['char_ngrams2_2'], 'dice_empty'),
            'Char Trigrams Dice': similarity(self.features['char_ngrams1_3'], self.features['char_ngrams2_3'], 'dice_empty'),
            'Char Fourgrams Dice': similarity(self.features['char_ngrams1_4'], self.features['char_ngrams2_4'], 'dice_empty'),
            'Char Bigrams (stop-words) Dice': similarity(self.features['char_ngrams1_sw_2'], self.features['char_ngrams2_sw_2'], 'dice_empty'),
            'Char Trigrams (stop-words) Dice': similarity(self.features['char_ngrams1_sw_3'], self.features['char_ngrams2_sw_3'], 'dice_empty'),
            'Char Fourgrams (stop-words) Dice': similarity(self.features['char_ngrams1_sw_4'], self.features['char_ngrams2_sw_4'], 'dice_empty'),

            'Overlap Unigram': similarity(self.features['word_ngrams1_1'], self.features['word_ngrams2_1'], 'overlap'),
            'Overlap Bigrams': similarity(self.features['word_ngrams1_2'], self.features['word_ngrams2_2'], 'overlap'),
            'Overlap Trigrams': similarity(self.features['word_ngrams1_3'], self.features['word_ngrams2_3'], 'overlap'),
            'Overlap Unigrams (stop-words)': similarity(self.features['word_ngrams1_sw_1'], self.features['word_ngrams2_sw_1'], 'overlap'),
            'Overlap Bigrams (stop-words)': similarity(self.features['word_ngrams1_sw_2'], self.features['word_ngrams2_sw_2'], 'overlap'),
            'Overlap Trigrams (stop-words)': similarity(self.features['word_ngrams1_sw_3'], self.features['word_ngrams2_sw_3'], 'overlap'),

            'Triplet Parser Jac.': similarity(self.features['triplet_parser1'], self.features['triplet_parser2']),
            'Triplet Parser Dice.': similarity(self.features['triplet_parser1'], self.features['triplet_parser2'], 'dice_empty'),
            'Triplet Parser Overlap': similarity(self.features['triplet_parser1'], self.features['triplet_parser2'], 'overlap'),

            # Features that we have already calculated a metric
            'Longest Common Subsequence': self.features['lcs_subsequence'],
            'Longest Common Subsequence (stop-words)': self.features['lcs_subsequence_sw'],
            'Longest Common Substring': self.features['lcs_substring'],
            'Longest Common Substring (stop-words)': self.features['lcs_substring_sw'],

            'Leacock-Chodorow': self.features['lch_sim'],
            'Path': self.features['wup_sim'],
            'Wu-Palmer': self.features['lin_sim'],
            'Lin': self.features['path_sim'],

            'Levenshtein Distance': self.features['levenshtein'],
            
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


    def extract_lexic(self):
        return pd.DataFrame({
            'Tokens Jac.': similarity(self.features['tokens1'], self.features['tokens2']),
            'Tokens (stop-words) Jac.': similarity(self.features['tokens1_sw'], self.features['tokens2_sw']),
            'Tokens Dice': similarity(self.features['tokens1'], self.features['tokens2'], 'dice_empty'),
            'Tokens (stop-words) Dice': similarity(self.features['tokens1_sw'], self.features['tokens2_sw'], 'dice_empty'),

            'Lemmas Jac.': similarity(self.features['lemmas1'], self.features['lemmas2']),
            'Lemmas (stop-words) Jac.': similarity(self.features['lemmas1_sw'], self.features['lemmas2_sw']),
            'Lemmas (Spacy) Jac.': similarity(self.features['lemmas1_spacy'], self.features['lemmas2_spacy']),
            'Lemmas Dice': similarity(self.features['lemmas1'], self.features['lemmas2'], 'dice_empty'),
            'Lemmas (stop-words) Dice': similarity(self.features['lemmas1_sw'], self.features['lemmas2_sw'], 'dice_empty'),
            'Lemmas (Spacy) Dice': similarity(self.features['lemmas1_spacy'], self.features['lemmas2_spacy'], 'dice_empty'),

            'Stemming Jac.': similarity(self.features['porter1'], self.features['porter2']),
            'Stemming Dice': similarity(self.features['porter1'], self.features['porter2'], 'dice_empty'),

            'Coref. Tokens Jac.': similarity(self.features['coref_tokens1'], self.features['coref_tokens2']),
            'Coref. Tokens Dice': similarity(self.features['coref_tokens1'], self.features['coref_tokens2'], 'dice_empty'),

            'Coref. Lemmas Jac.': similarity(self.features['coref_tokens1'], self.features['coref_tokens2']),
            'Coref. Lemmas Dice': similarity(self.features['coref_lemmas1'], self.features['coref_lemmas2'], 'dice_empty'),

            'Unigrams Jac.': similarity(self.features['word_ngrams1_1'], self.features['word_ngrams2_1']),
            'Bigrams Jac.': similarity(self.features['word_ngrams1_2'], self.features['word_ngrams2_2']),
            'Trigrams Jac.': similarity(self.features['word_ngrams1_3'], self.features['word_ngrams2_3']),
            'Fourgrams Jac.': similarity(self.features['word_ngrams1_4'], self.features['word_ngrams2_4']),
            'Unigrams (stop-words) Jac.': similarity(self.features['word_ngrams1_sw_1'], self.features['word_ngrams2_sw_1']),
            'Bigrams (stop-words) Jac.': similarity(self.features['word_ngrams1_sw_2'], self.features['word_ngrams2_sw_2']),
            'Trigrams (stop-words) Jac.': similarity(self.features['word_ngrams1_sw_3'], self.features['word_ngrams2_sw_3']),
            'Fourgrams (stop-words) Jac.': similarity(self.features['word_ngrams1_sw_4'], self.features['word_ngrams2_sw_4']),

            'Unigrams Dice': similarity(self.features['word_ngrams1_1'], self.features['word_ngrams2_1'], 'dice_empty'),
            'Bigrams Dice': similarity(self.features['word_ngrams1_2'], self.features['word_ngrams2_2'], 'dice_empty'),
            'Trigrams Dice': similarity(self.features['word_ngrams1_3'], self.features['word_ngrams2_3'], 'dice_empty'),
            'Fourgrams Dice': similarity(self.features['word_ngrams1_4'], self.features['word_ngrams2_4'], 'dice_empty'),
            'Unigrams (stop-words) Dice': similarity(self.features['word_ngrams1_sw_1'], self.features['word_ngrams2_sw_1'], 'dice_empty'),
            'Bigrams (stop-words) Dice': similarity(self.features['word_ngrams1_sw_2'], self.features['word_ngrams2_sw_2'], 'dice_empty'),
            'Trigrams (stop-words) Dice': similarity(self.features['word_ngrams1_sw_3'], self.features['word_ngrams2_sw_3'], 'dice_empty'),
            'Fourgrams (stop-words) Dice': similarity(self.features['word_ngrams1_sw_4'], self.features['word_ngrams2_sw_4'], 'dice_empty'),

            'Char Bigrams Jac.': similarity(self.features['char_ngrams1_2'], self.features['char_ngrams2_2']),
            'Char Trigrams Jac.': similarity(self.features['char_ngrams1_3'], self.features['char_ngrams2_3']),
            'Char Fourgrams Jac.': similarity(self.features['char_ngrams1_4'], self.features['char_ngrams2_4']),
            'Char Bigrams (stop-words) Jac.': similarity(self.features['char_ngrams1_sw_2'], self.features['char_ngrams2_sw_2']),
            'Char Trigrams (stop-words) Jac.': similarity(self.features['char_ngrams1_sw_3'], self.features['char_ngrams2_sw_3']),
            'Char Fourgrams (stop-words) Jac.': similarity(self.features['char_ngrams1_sw_4'], self.features['char_ngrams2_sw_4']),

            'Char Bigrams Dice': similarity(self.features['char_ngrams1_2'], self.features['char_ngrams2_2'], 'dice_empty'),
            'Char Trigrams Dice': similarity(self.features['char_ngrams1_3'], self.features['char_ngrams2_3'], 'dice_empty'),
            'Char Fourgrams Dice': similarity(self.features['char_ngrams1_4'], self.features['char_ngrams2_4'], 'dice_empty'),
            'Char Bigrams (stop-words) Dice': similarity(self.features['char_ngrams1_sw_2'], self.features['char_ngrams2_sw_2'], 'dice_empty'),
            'Char Trigrams (stop-words) Dice': similarity(self.features['char_ngrams1_sw_3'], self.features['char_ngrams2_sw_3'], 'dice_empty'),
            'Char Fourgrams (stop-words) Dice': similarity(self.features['char_ngrams1_sw_4'], self.features['char_ngrams2_sw_4'], 'dice_empty'),

            'Overlap Unigram': similarity(self.features['word_ngrams1_1'], self.features['word_ngrams2_1'], 'overlap'),
            'Overlap Bigrams': similarity(self.features['word_ngrams1_2'], self.features['word_ngrams2_2'], 'overlap'),
            'Overlap Trigrams': similarity(self.features['word_ngrams1_3'], self.features['word_ngrams2_3'], 'overlap'),
            'Overlap Unigrams (stop-words)': similarity(self.features['word_ngrams1_sw_1'], self.features['word_ngrams2_sw_1'], 'overlap'),
            'Overlap Bigrams (stop-words)': similarity(self.features['word_ngrams1_sw_2'], self.features['word_ngrams2_sw_2'], 'overlap'),
            'Overlap Trigrams (stop-words)': similarity(self.features['word_ngrams1_sw_3'], self.features['word_ngrams2_sw_3'], 'overlap'),

            # Features that we have already calculated a metric
            'Longest Common Subsequence': self.features['lcs_subsequence'],
            'Longest Common Subsequence (stop-words)': self.features['lcs_subsequence_sw'],
            'Longest Common Substring': self.features['lcs_substring'],
            'Longest Common Substring (stop-words)': self.features['lcs_substring_sw'],

            'Levenshtein Distance': self.features['levenshtein'],
            
            'Vector Space Sentence': self.features['vector_space_sentence'],

        })


    def extract_syntactic(self):
        return pd.DataFrame({
            'Lesk Jac.': similarity(self.features['lesk1'], self.features['lesk2']),
            'Lesk Dice': similarity(self.features['lesk1'], self.features['lesk2'], 'dice_empty'),

            'Triplet Parser Jac.': similarity(self.features['triplet_parser1'], self.features['triplet_parser2']),
            'Triplet Parser Dice.': similarity(self.features['triplet_parser1'], self.features['triplet_parser2'], 'dice_empty'),
            'Triplet Parser Overlap': similarity(self.features['triplet_parser1'], self.features['triplet_parser2'], 'overlap'),

            'Leacock-Chodorow': self.features['lch_sim'],
            'Path': self.features['wup_sim'],
            'Wu-Palmer': self.features['lin_sim'],
            'Lin': self.features['path_sim'],
           
            'WAWO': self.features['WAWO'],
            'WWO': self.features['WWO'],

            'Vector Space Sentence IC': self.features['vector_space_sentence_ic'],
        })