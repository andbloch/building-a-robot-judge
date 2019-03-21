# IMPORTS ######################################################################


import pandas as pd
import numpy as np


# EXERCISE FUNCTIONS ###########################################################

def get_transformation(data_1, data_2):

    # extract the columns of the data frames
    cols_1 = data_1.columns.tolist()
    cols_2 = data_2.columns.tolist()

    # compute the intersection of the vocabularies
    V_3 = list(set(cols_1) & set(cols_2))

    # determine relevant dimensions
    m_1 = data_1.shape[1]
    m_2 = data_2.shape[1]
    m_3 = len(V_3)

    # creates a transformation matrix
    def create_transf_matrix(m_prev, cols_prev):

        # initialize transformation matrix
        T = np.zeros(shape=(m_prev, m_3), dtype=float)

        # for each word in the common vocabulary
        for i, word in enumerate(V_3):

            # get index of that word in previous vocabulary (always exists!)
            j = cols_1.index(word)

            # create corresponding entry in transformation matrix to copy
            # that count to the right place
            T[j, i] = 1.0

        return T

    T_1 = create_transf_matrix(m_1, cols_1)
    T_2 = create_transf_matrix(m_2, cols_2)

    return T_1, T_2


def cosine_sim(T_1, T_2, data_1, data_2):

    # use transformations to transform articles to same feature-space
    D_1 = np.matmul(data_1.values, T_1)
    D_2 = np.matmul(data_2.values, T_2)

    # normalize the articles
    D_1 /= np.linalg.norm(D_1, axis=1)[:, np.newaxis]
    D_2 /= np.linalg.norm(D_2, axis=1)[:, np.newaxis]

    # compute cosine-similarities in new feature space
    D_3 = np.matmul(D_1, D_2.T)

    # create similarity data frame
    similarities = pd.DataFrame(
        index=pd.Index(data_1.index.tolist(), name='article_1_id'),
        columns=pd.Index(data_2.index.tolist(), name='article_2_id')
    )

    # store cosine-similarities in data frame
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


T_1, T_2 = get_transformation(data_1, data_2)
similarities = cosine_sim(T_1, T_2, data_1, data_2)

# print computed similarities
print('computed similarities:')
print(similarities)