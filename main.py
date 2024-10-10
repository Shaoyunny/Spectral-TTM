# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import gensim.corpora as corpora
from preprocess import TextPreprocessor
from Spectral_TTM import TTM_Gibbs_Sampling_opti
from Spectral_TTM import compute_moments_matrix, compute_moments_tensor, svd_decomposition, tensor_matrix_multiply, tensor_spectral_algorithm
from evaluation import Spectral_TTM_calculate_perplexity, Spectral_TTMPMF_MAE
from Spectral_TTMPMF import TTM_predict_ALS
import warnings

warnings.filterwarnings("ignore")

def main():
    # Load training data
    train_data = pd.read_csv("/home/suibe/wmk/zsy/dataset/AMAZON_FASHION_5_deal5_train.csv", encoding='utf-8')
    train_data_text = train_data['reviewText']
    data = train_data_text.values.tolist()

    # Preprocess text data
    preprocessor = TextPreprocessor()
    data_words = list(preprocessor.sent_to_words(data))
    data_words_nostops = preprocessor.remove_stopwords(data_words)
    data_lemmatized = preprocessor.lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    
    # Create dictionary and corpus for topic modeling
    id2word = corpora.Dictionary(data_lemmatized)
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]
    n_doc = len(data_lemmatized)
    texts_data, n_word = preprocessor.check_words(data_lemmatized)

    # Calculate second and third order moments
    unique_words = list(set(word for doc in data_lemmatized for word in doc))
    E_2 = np.zeros((n_word, n_word))
    E_3 = np.zeros((n_word, n_word, n_word))
    word_to_index = {word: i for i, word in enumerate(unique_words)}

    # Compute normalized moments
    normalized_second_order_moments_matrices = compute_moments_matrix(data_lemmatized, word_to_index)
    normalized_third_order_moments_tensors = compute_moments_tensor(data_lemmatized, word_to_index)

    for matrix in normalized_second_order_moments_matrices:
        E_2 += matrix
    M_2 = E_2 / n_doc

    for tensor in normalized_third_order_moments_tensors:
        E_3 += tensor
    M_3 = E_3 / n_doc

    # Create user-item rating matrix
    rating_matrix = pd.pivot_table(train_data, values='overall', index='reviewerID', columns='asin', fill_value=0)
    rating_matrix = np.array(rating_matrix.values)
    user_num, item_num = rating_matrix.shape
    rating_matrix_T = rating_matrix.T
    is_rating = np.where(rating_matrix != 0, 1, 0)

    # Set parameters for the model
    max_iter1 = 100
    L = 100  # Outer iterations
    N = 50   # Inner iterations
    k1, p1, q1 = 40, 40, 7
    lamb = [1]

    # SVD decomposition and tensor spectral algorithm
    U_k, S_k, V_k = svd_decomposition(M_2, k1)
    S_sqrt = np.diag(np.sqrt(S_k))
    W = U_k @ np.diag(S_sqrt)
    T = tensor_matrix_multiply(M_3, W, W, W)
    alpha = tensor_spectral_algorithm(T, k1, L, N)

    # Initialize parameters for Gibbs sampling
    beta = np.ones((k1, n_word)) * 0.001
    xi = np.ones((p1, 1)) * 0.001
    rho = np.ones((q1, 1)) * 0.001
    t1 = TTM_Gibbs_Sampling_opti(k1, p1, q1, n_word, user_num, item_num, n_doc, alpha.astype(float), beta, xi, rho, train_data['userid'], train_data['itemid'], train_data_text.values.tolist(), max_iter1)
    
    # Get feature matrix and run Gibbs sampling
    t1.get_feature_matrix()
    t1.initial_matrix()
    a = t1.TTM_Gibbs()
    E1, H1, theta1, phi1 = t1.calculate_parameter(a)

    # Calculate perplexity and MAE
    per1 = Spectral_TTM_calculate_perplexity(theta1, phi1, E1, H1, user_num, train_data_text.values.tolist(), item_num, k1, p1, q1)
    print(per1)
    pre1 = TTM_predict_ALS(E1, H1, k1, p1, q1, max_iter1, lamb, user_num, item_num, rating_matrix)
    MAE1 = Spectral_TTMPMF_MAE(pre1, user_num, item_num, rating_matrix)
    print(MAE1)

if __name__ == "__main__":
    main()
