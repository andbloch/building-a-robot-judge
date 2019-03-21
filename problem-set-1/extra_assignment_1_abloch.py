# IMPORTS ######################################################################


import pandas as pd
import numpy as np


# EXERCISE FUNCTIONS ###########################################################


def cosine_similarities(data_1, data_2):
    """
    Takes two panda data frames which each are the term frequency matrices of
    the articles of some corpus w.r.t its vocabulary and returns the document
    similarities between the articles from corpus 1 and corpus 2 w.r.t. the
    union of the vocabularies of the corpuses.
    :param data_1: the data frame with the articles and term freq. of corpus 1
    :param data_2: the data frame with the articles and term freq. of corpus 2
    :return: the document similarities between the articles from corpus 1 and 2
    """

    # create copies of arguments
    data_1 = data_1.__deepcopy__()
    data_2 = data_2.__deepcopy__()

    # extract the vocabularies
    V_1 = set(data_1.columns.tolist())
    V_2 = set(data_2.columns.tolist())

    # compute the intersection of the vocabularies
    V_3 = list(V_1 & V_2)

    # define pointers (shorthand) to the data matrices
    D_1 = data_1.values
    D_2 = data_2.values

    # normalize the data matrices
    D_1 = D_1 / np.linalg.norm(D_1, axis=-1)[:, np.newaxis]
    D_2 = D_2 / np.linalg.norm(D_2, axis=-1)[:, np.newaxis]

    # write back the normalized data matrices into data frames
    # otherwise changes won't be reflected once we select data again
    for i in range(0, len(data_1.index.tolist())):
        for j in range(0, len(data_1.columns.tolist())):
            data_1.iloc[i,j] = D_1[i, j]
    for i in range(0, len(data_2.index.tolist())):
        for j in range(0, len(data_2.columns.tolist())):
            data_2.iloc[i,j] = D_2[i, j]

    # extract reduced data matrices
    # IMPORTANT: This is more efficient than the proposed matrix-multiplication
    # Matrix-Mult: O(n^3)
    # Column-Selection: O(n^2)
    D_1 = data_1[V_3].values
    D_2 = data_2[V_3].values

    # compute cosine-similarities
    D_3 = np.matmul(D_1, D_2.T)

    # create similarity data frame
    similarities = pd.DataFrame(
        index=pd.Index(data_1.index.tolist(), name='article_1_id'),
        columns=pd.Index(data_2.index.tolist(), name='article_2_id')
    )

    # store similarities in resulting similarity data frame
    for i in range(0, len(data_1.index.tolist())):
        for j in range(0, len(data_2.index.tolist())):
            similarities.iloc[i,j] = D_3[i, j]

    return similarities


# BUILDING OF EXAMPLE DATASETS #################################################


# specify number of documents in each corpus
n_1 = 5  # number of documents in first corpus
n_2 = 9  # number of documents in second corpus

# create words for both dictionaries
columns_1 = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
columns_2 = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

# derive feature space sizes of each corpus
m_1 = len(columns_1)
m_2 = len(columns_2)

# create article IDs for each corpus
corpus_1_article_ids = []
corpus_2_article_ids = []

# create data frames
data_1 = pd.DataFrame(
    index=pd.Index(corpus_1_article_ids, name='article_id'),
    columns=pd.Index(columns_1, name='words')
)
data_2 = pd.DataFrame(
    index=pd.Index(corpus_2_article_ids, name='article_id'),
    columns=pd.Index(columns_2, name='words')
)

# fill data frames with documents (and their feature frequencies)
for i in range(0, n_1):
    data_1.loc[i] = np.random.randint(10, size=m_1).astype(float).tolist()
for i in range(0, n_2):
    data_2.loc[i] = np.random.randint(10, size=m_2).astype(float).tolist()

# print example datasets
print('example corpus 1:')
print(data_1)
print('')
print('example corpus 2:')
print(data_2)
print('')


# FINAL SCENARIO ILLUSTRATION ##################################################


similarities = cosine_similarities(data_1, data_2)

# print computed similarities
print('computed similarities:')
print(similarities)


# EXPLANATION OF SOLUTION ######################################################


# The cosine of the angle θ(u,v) between two documents u and v is computed
# as follows:
#                   <u,v>
# cos(θ(u,v)) = -------------
#               ||u|| * ||v||
# This is exactly the cosine-distance between two documents u and v.

# Now, if the corpuses where the documents u and v have different vocabularies
# the question we need to ask ourselves is: shall we compute the common
# dictionary as
# - the intersection, or
# - the union
# of the dictionaries of the two corpora?

# Answer: Let's analyze the formula for the cosine distance
# - the dot product <u,v> is computed more efficiently if we just consider
#   the intersection of the dictionaries (as otherwise we have just a lot
#   of 0-factors if a word is just in the vocabulary of one corpus.
# - however, we may think about how we want to scale the vectors. There are
#   two options:
#   (1) normalizing w.r.t. the union of the vocabularies:
#       Then we rescale a vector to unit norm w.r.t. all of the words that it
#       actually contains. This doesn't distort the norm.
#   (2) normalizing w.r.t. the intersection of the vocabularies
#       Then we rescale a vector to unit norm only w.r.t. the term frequencies
#       of the words that appear in both vocabularies. This may lead to
#       greater cosine similarities, as the coefficients of a document may be
#       increased if we drop the scaling factors that would have belonged
#       to words that only appear in its original vocabulary (but not in the
#       intersection)
#   In general, option (2) makes more sense, because then the cosine similarity
#   still captures the following: Let's consider two articles a_1 and a_2 which
#   have exactly the same term frequencies for the intersection of their
#   vocabularies. Additionally, a_1 contains some term frequencies over a
#   terms that only appear in the corpus of a_1. Then, by choosing option (1)
#   or (2) we'd have the following outcomes:
#   (1) Then they would have a cosine similarity of 1 (as they are absolutely
#       collinear (in the intersection vocabulary)
#   (2) Then they would have a a cosine similarity of less than 1, as they
#       are collinear in the intersection vocabulary dimensions, but not
#       in the vocabulary dimensions that correspond to words that only appear
#       in the corpus of a_1.
#   Now, option (2) seems more reasonable, when we'd like to capture that
#   difference, especially when the terms that do not appear in both corpuses
#   have an important meaning.
#   Now, option (1) seems more reasonable, if both corpuses contain articles
#   about the same topic, and one corpus has just decided to only use a
#   subset of the words to count the term frequencies. Then, one might say
#   that these additional words usually would be the same for the other article
#   (extrapolated based on co-ocurrences). Hence we don't need to consider this
#   extra complexity.
#   Now, option (2) seems overall more reasonable as it provides a more complete
#   comparison measure. That's why I chose it. One may think about providing
#   this option as an extra parameter, but this wasn't requested.
