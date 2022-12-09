import re
import nltk
from nltk.metrics import jaccard_distance
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet, wordnet_ic
from nltk import ne_chunk
from nltk.stem import WordNetLemmatizer
nltk.download('maxent_ne_chunker')
nltk.download('conll2000')
from scipy.spatial.distance import hamming
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


def tokenize(sentences, sw=False):
    sentences_tokens = []
    sentences_pairs = []
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)  # tokenize
        pairs = nltk.pos_tag(tokens)  # get the pos of the tokens
        if sw:
            tokens = [regex.sub('', w.lower()) for w in tokens]
        else:
            tokens = [regex.sub('', w.lower()) for w in tokens if
                      not w.lower() in stopw]  # remove stopwords and symbols from each word
        tokens = [w for w in tokens if w]  # remove empty elements
        sentences_tokens.append(set(tokens))
        sentences_pairs.append(pairs)
    return sentences_tokens, sentences_pairs


def lemmatize(sentences_POS, sw=False):
    lemmas = []
    for sentence in sentences_POS:
        lemmatized = [pos_wn(pair) for pair in sentence]
        if sw:
            lemmatized = [regex.sub('', w.lower()) for w in lemmatized]
        else:
            lemmatized = [regex.sub('', w.lower()) for w in lemmatized if
                          not w.lower() in stopw]  # remove stopwords and symbols from each word
        lemmatized = [w for w in lemmatized if w]  # remove empty elements
        lemmas.append(set(lemmatized))
    return lemmas


def jaccard_empty(set1, set2):
    if len(set1) != 0 and len(set2) != 0:
        return jaccard_distance(set1, set2)
    else:
        return 0


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


def lesk(sentences, POS):
    syns = []
    for sentence, pairs in zip(sentences, POS):
        synsets = [nltk.wsd.lesk(sentence, pair[0], pos=tags.get(pair[1][:2].upper())) for pair in pairs if
                   pair[0].lower() not in stopw]  # skip if the word is stopword else use lesk
        synsets = [syn.name() for syn in synsets if syn]  # we ignore the words without meaning
        syns.append(set(synsets))
    return syns


def NE_basic(pos):
    texts = []
    for pair in pos:
        chunks = nltk.ne_chunk(pair, binary=False)  # chunk it # https://www.youtube.com/watch?v=zDnPFxnALBg
        triads = nltk.tree2conlltags(chunks)  # get the triads with the token, pos tag, and if its named entity
        text = []
        for triad in triads:
            token = re.sub(r'[^\w\s]', '', triad[0]).lower()  # remove punctuaction inside each token
            if token in stopw or re.match(r'^[_\W]+$',
                                          token) or token == '':  # if its stopword, non-alphanumeric token, or empty we skip
                continue
            if triad[2][0] == 'O':  # if not named entity we append to text
                text.append(token)
            else:  # if named entity we append to text as well
                text.append(triad[2][0])
        texts.append(text)
    return texts


def syntactic_role_sim(pos1, pos2, method='lch'):
    similarities = []
    for pairs1, pairs2 in zip(pos1, pos2):
        syns1 = [wordnet.synsets(pair[0], tags.get(pair[1])) for pair in pairs1]
        syns2 = [wordnet.synsets(pair[0], tags.get(pair[1])) for pair in pairs2]
        for syn1 in syns1:
            sim = []
            if len(syn1) > 0:
                s1 = syn1[0]
            else:
                s1 = ''
            if isinstance(s1, nltk.corpus.reader.wordnet.Synset):
                for syn2 in syns2:
                    if len(syn2) > 0:
                        s2 = syn2[0]
                    else:
                        s2 = ''
                    if isinstance(s2, nltk.corpus.reader.wordnet.Synset):
                        try:
                            if method == 'lch':
                                sim.append(s1.lch_similarity(s2))
                            elif method == 'wup':
                                sim.append(s1.wup_similarity(s2))
                            elif method == 'path':
                                sim.append(s1.path_similarity(s2))
                            elif method == 'lin':
                                sim.append(s1.lin_similarity(s2, semcor_ic))
                        except:
                            sim.append(0)

        if len(sim) > 0:
            similarities.append(max(sim))
        else:
            similarities.append(0)

    return similarities


def levenshtein_distance(l1, l2):
    sent1 = ' '.join(l1)
    sent2 = ' '.join(l2)
    return nltk.edit_distance(sent1, sent2) / max(len(sent1), len(sent2))


def sorensen_dice(l1, l2):
    bigrams_l1 = list(nltk.ngrams(' '.join(l1), 2))
    bigrams_l2 = list(nltk.ngrams(' '.join(l2), 2))
    common_bigrams = list(set(bigrams_l1).intersection(bigrams_l2))
    if len(bigrams_l1) != 0 or len(bigrams_l2) != 0:
        s = 2 * (len(common_bigrams)) / (len(bigrams_l1) + len(bigrams_l2))
    else:
        s = 0
    return s


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


class Features:

    def __init__(self, data):
        self.pair1 = data['Sentence 1']
        self.pair2 = data['Sentence 2']

        self.tokens1, self.pos1 = tokenize(self.pair1)
        self.tokens2, self.pos2 = tokenize(self.pair2)
        self.tokens1_sw, _ = tokenize(self.pair1, sw=True)  # with stopwords
        self.tokens2_sw, _ = tokenize(self.pair2, sw=True)  # with stopwords
        self.NE1 = NE_basic(self.pos1)
        self.NE2 = NE_basic(self.pos2)

        self.lemmas1 = lemmatize(self.pos1)
        self.lemmas2 = lemmatize(self.pos2)
        self.lemmas1_sw = lemmatize(self.pos1, sw=True)
        self.lemmas2_sw = lemmatize(self.pos2, sw=True)

    def tokens(self):
        self.jac_tokens = [(1 - jaccard_distance(t[0], t[1])) for t in zip(self.tokens1, self.tokens2)]
        self.jac_tokens_sw = [(1 - jaccard_distance(t[0], t[1])) for t in zip(self.tokens1_sw, self.tokens2_sw)]

    def lemmas(self):
        self.jac_lemmas = [(1 - jaccard_distance(t[0], t[1])) for t in zip(self.lemmas1, self.lemmas2)]
        self.jac_lemmas_sw = [(1 - jaccard_distance(t[0], t[1])) for t in zip(self.lemmas1_sw, self.lemmas2_sw)]

    def lesk(self):
        l1 = lesk(self.pair1, self.pos1)
        l2 = lesk(self.pair2, self.pos2)
        self.jac_lesk = [1 - jaccard_empty(t[0], t[1]) for t in zip(l1, l2)]

    def n_grams(self):
        self.jac_2grams = [
            (1 - jaccard_empty(set([g for g in nltk.ngrams(' '.join(t[0]), 2)]),
                               set([g for g in nltk.ngrams(' '.join(t[1]), 2)]))) for
            t in zip(self.tokens1, self.tokens2)]
        self.jac_3grams = [
            (1 - jaccard_empty(set([g for g in nltk.ngrams(' '.join(t[0]), 3)]),
                               set([g for g in nltk.ngrams(' '.join(t[1]), 3)]))) for
            t in zip(self.tokens1, self.tokens2)]
        self.jac_4grams = [
            (1 - jaccard_empty(set([g for g in nltk.ngrams(' '.join(t[0]), 4)]),
                               set([g for g in nltk.ngrams(' '.join(t[1]), 4)]))) for
            t in zip(self.tokens1, self.tokens2)]
        self.jac_2grams_sw = [
            (1 - jaccard_empty(set([g for g in nltk.ngrams(' '.join(t[0]), 2)]),
                               set([g for g in nltk.ngrams(' '.join(t[1]), 2)]))) for
            t in zip(self.tokens1_sw, self.tokens2_sw)]
        self.jac_3grams_sw = [
            (1 - jaccard_empty(set([g for g in nltk.ngrams(' '.join(t[0]), 3)]),
                               set([g for g in nltk.ngrams(' '.join(t[1]), 3)]))) for
            t in zip(self.tokens1_sw, self.tokens2_sw)]
        self.jac_4grams_sw = [
            (1 - jaccard_empty(set([g for g in nltk.ngrams(' '.join(t[0]), 4)]),
                               set([g for g in nltk.ngrams(' '.join(t[1]), 4)]))) for
            t in zip(self.tokens1_sw, self.tokens2_sw)]

    def NE(self):
        self.NE_basic = [(1 - jaccard_distance(set(n[0]), set(n[1]))) for n in zip(self.NE1, self.NE2)]

    def syntatic_role(self):
        self.lch_sim = syntactic_role_sim(self.pos1, self.pos2, method='lch')
        self.wup_sim = syntactic_role_sim(self.pos1, self.pos2, method='wup')
        self.lin_sim = syntactic_role_sim(self.pos1, self.pos2, method='lin')
        self.path_sim = syntactic_role_sim(self.pos1, self.pos2, method='path')

    def lcs_subsequence(self):
        self.lcs_subsequence = [longest_common_subsequence(t[0], t[1]) for t in
                                zip(self.tokens1, self.tokens2)]
        self.lcs_subsequence_sw = [longest_common_subsequence(t[0], t[1]) for t in
                                   zip(self.tokens1_sw, self.tokens2_sw)]

    def lcs_substring(self):
        self.lcs_substring = [longest_common_substring(t[0], t[1]) for t in
                              zip(self.tokens1, self.tokens2)]
        self.lcs_substring_sw = [longest_common_substring(t[0], t[1]) for t in
                                 zip(self.tokens1_sw, self.tokens2_sw)]

    def levenshtein_dist(self):
        self.leven_dist = [levenshtein_distance(l[0], l[1]) for l in
                           zip(self.lemmas1, self.lemmas2)]

    def sorensen_dice_coef(self):
        self.sd_coefficient = [sorensen_dice(l[0], l[1]) for l in
                               zip(self.lemmas1, self.lemmas2)]

    def hamming_distance(self):
        self.ham_dist = [hamming(l[0], l[1]) for l in
                         zip(self.lemmas1, self.lemmas2)]

    def num_tags(self):
        self.verbs_diff = [num_verbs(p[0], p[1]) for p in
                           zip(self.pos1, self.pos2)]
        self.nouns_diff = [num_nouns(p[0], p[1]) for p in
                           zip(self.pos1, self.pos2)]
        self.adjs_diff = [num_adjs(p[0], p[1]) for p in
                          zip(self.pos1, self.pos2)]
        self.advs_diff = [num_advs(p[0], p[1]) for p in
                          zip(self.pos1, self.pos2)]

    def extract_all(self):
        self.NE()
        # self.syntatic_role()
        # self.hamming_distance()
        self.num_tags()
        self.sorensen_dice_coef()
        self.tokens()
        self.lemmas()
        self.n_grams()
        self.lcs_substring()
        self.lcs_subsequence()
        self.lesk()
        self.levenshtein_dist()

        return pd.DataFrame(
            {'NE basic': self.NE_basic,
             'Tokens Jac. Sim.': self.jac_tokens,
             'Tokens (stop-words) Jac. Sim.': self.jac_tokens_sw,
             'Lemmas Jac. Sim.': self.jac_lemmas,
             'Lemmas (stop-words) Jac. Sim.': self.jac_lemmas_sw,
             'Bigrams Jac. Sim.': self.jac_2grams,
             'Bigrams (stop-words) Jac. Sim.': self.jac_2grams_sw,
             'Trigrams Jac. Sim.': self.jac_3grams,
             'Trigrams (stop-words) Jac. Sim.': self.jac_3grams_sw,
             'Fourgrams Jac. Sim.': self.jac_4grams,
             'Fourgrams (stop-words) Jac. Sim.': self.jac_4grams_sw,
             'Longest Common Subsequence': self.lcs_subsequence,
             'Longest Common Subsequence (stop-words)': self.lcs_subsequence_sw,
             'Longest Common Substring': self.lcs_substring,
             'Longest Common Substring (stop-words)': self.lcs_substring_sw,
             'Lesk Jac. Sim.': self.jac_lesk,
             # 'Leacock-Chodorow Sim.': self.lch_sim,
             # 'Path Sim.': self.path_sim,
             # 'Wu-Palmer Sim.': self.wup_sim,
             # 'Lin Sim.': self.lin_sim,
             'Levenshtein Distance': self.leven_dist,
             'Sorensen-Dice Coefficient (lemmas without stop-words)': self.sd_coefficient,
             # 'Hamming Distance': self.ham_dist,
             '# of Verb Tags': self.verbs_diff,
             '# of Noun Tags': self.nouns_diff,
             '# of Adjective Tags': self.adjs_diff,
             '# of Adverb Tags': self.advs_diff
             })
