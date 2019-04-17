################################################################################
# IMPORTS                                                                      #
################################################################################


import os
import numpy as np
import pandas as pd
import pickle
from zipfile import ZipFile
from collections import Counter
from nltk import ngrams
from nltk.stem import SnowballStemmer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from string import punctuation
import re
from math import isnan


################################################################################
# PARAMETERS                                                                   #
################################################################################


# define input data locations
DATA_DIR = './data'
CASE_DIR = './data/cases'
CASE_METADATA = 'case_metadata.csv'

# define output data locations
DF_2SLS_FILENAME = '2SLS.pkl'
CASE_WORD_LIST = 'case_word_list.pkl'
X_REVERSED_FILENAME = 'X_reversed.npy'
Y_REVERSED_FILENAME = 'y_reversed.npy'
X_LOG_CITES_FILENAME = 'X_log_cites.npy'
Y_LOG_CITES_FILENAME = 'y_log_cites.npy'

# parameters
N_GRAM_LENGTH = 3
NUM_FEATURES_MOST_COMMON = 10000


################################################################################
# CUSTOM TOKEN EXTENSION                                                       #
################################################################################


from spacy.tokens import Token

def get_filtered_text(token):
    return token._._filtered_text

def set_filtered_text(token, value):
    token._._filtered_text = value

# user-facing attribute used for getting and setting
Token.set_extension('filtered_text',
                    getter=get_filtered_text,
                    setter=set_filtered_text)


################################################################################
# DATASET LOADING FUNCTIONS                                                    #
################################################################################


def get_exercise_1_and_5_dataset():
    X = np.load(os.path.join(DATA_DIR, X_REVERSED_FILENAME))
    y = np.load(os.path.join(DATA_DIR, Y_REVERSED_FILENAME))
    return X, y


def get_exercise_2_dataset():
    X = np.load(os.path.join(DATA_DIR, X_LOG_CITES_FILENAME))
    y = np.load(os.path.join(DATA_DIR, Y_LOG_CITES_FILENAME))
    return X, y


def get_exercise_3_dataframe():
    df = pd.read_pickle(os.path.join(DATA_DIR, DF_2SLS_FILENAME))
    return df


def get_exercise_4_list():
    with open(os.path.join(DATA_DIR, CASE_WORD_LIST), 'rb') as f:
        case_word_list = pickle.load(f)
    return case_word_list


################################################################################
# DATASET CREATION FUNCTIONS                                                   #
################################################################################


def create_datasets():
    _create_exercise_3_dataset()
    _create_exercise_1_2_4_5_datasets()

def _create_exercise_3_dataset():

    # read the metadata
    df = pd.read_csv(os.path.join(DATA_DIR, CASE_METADATA))

    # only keep datapoints which are fully defined
    df = df[(df.notnull()).sum(axis=1) == 5]

    # X: compute number of citations per case
    df['cites_case'] = np.exp(df['log_cites'])

    # Z: compute average citations to cases of each judge
    df2 = df[['judge_id', 'cites_case']].groupby(['judge_id']).mean()
    df2 = df2.rename(columns={'cites_case': 'avg_cites_case'})

    # combine data sources
    for current_judge_id, num_avg_cites in df2.iterrows():
        df.loc[df.judge_id == current_judge_id, 'avg_cites_judge'] = \
            float(num_avg_cites)

    # create reduced data frame
    df = df[['case_reversed', 'cites_case', 'avg_cites_judge', 'year']]

    # save dataframe
    df.to_pickle(os.path.join(DATA_DIR, DF_2SLS_FILENAME))


class CounterMessage(object):

    def __init__(self):
        self.counter = 1
        self.last_action = ''

    def update(self, max_cnt, action):
        # reset counter for every new action
        if self.last_action != action:
            self.counter = 1
            self.last_action = action
            print('')
        # print status message
        print('\r{}/{}: {}.'.format(self.counter, max_cnt, action),
              end=' ' * 10, flush=True)
        self.counter += 1


def _create_exercise_1_2_4_5_datasets():

    # create message counter
    counter = CounterMessage()

    # open cases zip file
    zfile = ZipFile('data/cases.zip')
    caseids = []
    raw_texts = {}
    years = {}

    # randomly shuffle files
    members = zfile.namelist()
    NUM_CASES = len(members)

    for case in members:
        year, caseid = case[:-4].split('_')
        with zfile.open(case) as f:
            raw_text = f.read().decode()
        raw_texts[caseid] = raw_text
        years[caseid] = int(year)
        caseids.append(caseid)
        counter.update(NUM_CASES, 'opened')

    # do NLP
    nlp = spacy.load('en')
    spacy_documents = {}
    for caseid in caseids:
        spacy_documents[caseid] = nlp(raw_texts[caseid])
        counter.update(NUM_CASES, 'nlp-processed')

    # create punctuation remover
    punctuation_remover = str.maketrans('', '', punctuation)
    # create lemmatizer
    lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)


    def filter_and_transform(token):
        # get the token's word(s)
        word = ''.join(token.text)
        # replace newlines with spaces
        word = word.replace('\r', ' ').replace('\n', ' ')
        # remove punctuation
        word = word.translate(punctuation_remover)
        # replace multiple subsequent spaces with one space
        word = re.sub(' +', ' ', word)
        # check that word still has some text (not just one char or space)
        if len(word) <= 1:
            return False, (word, token.pos_)
        # normalize numbers (28, 28th, 1st, ...)
        if any(char.isdigit() for char in word):
            return False, (word, token.pos_)
        # lemmatize the word
        lemmas = lemmatizer(word, token.pos)[0]
        # try to lemmatize the word
        if isinstance(lemmas, (list,)) and len(lemmas) > 0:
            # pick the first option if several lemmas were found
            word = lemmas[0]
        else:
            # no lemma was found (just keep the original word)
            word = word
        # convert the word to lowercase
        word = word.lower()
        # remove stopwords
        if word in STOP_WORDS:
            return False, (word, token.pos_)
        # finally, return the filtered word and type
        return True, (word, token.pos_)

    case_ngrams = {}
    all_ngrams = []
    all_case_tokens = []

    stemmer = SnowballStemmer('english')

    # n-gram cases
    for caseid in caseids:
        spacy_document = spacy_documents[caseid]
        # process each sentence separately
        # (we don't want n-grams to overlap sentences)
        case_tokens = []
        case_noun_ngrams = []
        for sentence in spacy_document.sents:
            sentence_tokens = []
            # filter the tokens in the case
            for token in sentence:
                take_token, filtered_token = filter_and_transform(token)
                if take_token:
                    sentence_tokens.append(filtered_token)
            # append list of processed tokens for this document
            tl = [stemmer.stem(t[0]) for t in sentence_tokens]
            case_tokens += (tl)
            # create list to keep track of all n-grams ending in a noun
            # in this sentence
            case_sentence_noun_ngrams = []
            # iterate over all ngrams that can be built out of the tokens
            for ngram in ngrams(sentence_tokens, N_GRAM_LENGTH):
                # check if the last word is a noun
                if ngram[N_GRAM_LENGTH-1][1] == 'NOUN':
                    # if so, add that n-gram
                    curr_ngram = (ngram[0][0],
                                  ngram[1][0],
                                  ngram[2][0])
                    # stores all n-grams for this sentence of this case
                    case_sentence_noun_ngrams.append(curr_ngram)
                    # stores all n-grams for all cases
                    all_ngrams.append(curr_ngram)
            # save all n-grams appearing in sentence
            case_noun_ngrams += case_sentence_noun_ngrams
        # save list of all n-grams for this case
        case_ngrams[caseid] = case_noun_ngrams
        counter.update(NUM_CASES, 'n-grammed')
        # save list of all appearing tokens for this case
        all_case_tokens.append(case_tokens)

    # save the processed case tokens
    with open(os.path.join(DATA_DIR, CASE_WORD_LIST), 'wb') as f:
        pickle.dump(all_case_tokens, f)

    # load metadata into dictionary
    metadata = {}
    case_metadata = pd.read_csv(os.path.join(DATA_DIR, CASE_METADATA)).values
    for caseid, case_reversed, judge_id, year, log_cites in case_metadata:
        metadata[caseid] = {
            'reversed': case_reversed,
            'judge_id': judge_id,
            'year_meta': year,
            'log_cites': log_cites
        }

    # create list of values
    X_reversed = []
    X_log_cites = []
    y_reversed = []
    y_log_cites = []

    # determine most common n_grams
    most_common = Counter(all_ngrams).most_common(NUM_FEATURES_MOST_COMMON)

    # featurize according to most common n_grams
    for caseid in caseids:
        # count the n_gram frequencies of the current case
        current_case_ngrams = Counter(case_ngrams[caseid])
        # create feature vector
        features = np.zeros(len(most_common))
        # for each most common ngram
        for i in range(len(most_common)):
            ngram = most_common[i][0]
            # check if it appears in the cases's n_grams
            if ngram in current_case_ngrams:
                # if it appears, add the number of appearances as a feature
                features[i] = current_case_ngrams[ngram]
        # create feature vectors and targets for reversed
        X_reversed.append(features)
        y_reversed.append(metadata[caseid]['reversed'])
        # create feature vectors and targets for log_cites
        if not isnan(metadata[caseid]['log_cites']):
            X_log_cites.append(features)
            y_log_cites.append(metadata[caseid]['log_cites'])
        counter.update(NUM_CASES, 'featurized')

    # convert to numpy array
    X_reversed = np.array(X_reversed)
    X_log_cites = np.array(X_log_cites)
    y_reversed = np.array(y_reversed)
    y_log_cites = np.array(y_log_cites)

    # standardize maintain sparsity by not subtracting the mean
    X_reversed = X_reversed / np.std(X_reversed, axis=0)
    X_log_cites = X_log_cites / np.std(X_log_cites, axis=0)

    # remove potential zero std columns (constant features)
    # this happens if we have no metadata for the only rows
    # where the trigram appears
    X_log_cites = X_log_cites.transpose()
    X_log_cites = X_log_cites[~np.isnan(X_log_cites).any(axis=1)]
    X_log_cites = X_log_cites.transpose()

    # remove columns with zero variance of X_log_cites
    col_mask = list(set(list(np.argwhere(np.isnan(X_log_cites))[:,1])))

    # save the data
    np.save(os.path.join(DATA_DIR, X_REVERSED_FILENAME), X_reversed)
    np.save(os.path.join(DATA_DIR, X_LOG_CITES_FILENAME), X_log_cites)

    np.save(os.path.join(DATA_DIR, Y_REVERSED_FILENAME), y_reversed)
    np.save(os.path.join(DATA_DIR, Y_LOG_CITES_FILENAME), y_log_cites)
