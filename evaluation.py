### Perplexity Calculation
def Spectral_TTM_calculate_perplexity(theta, phi, E, H, user_num, text_data, item_num, K, P, Q):
    prob_list = []  # Store probabilities
    for i in range(user_num):
        for j in range(item_num):
            for vocab_index in text_data[i]:
                prob = 0
                for p in range(P):
                    for q in range(Q):
                        for k in range(K):
                            prob += theta[k][p][q] * E[p][i] * H[q][j] * phi[vocab_index][k]
                prob_list.append(prob)
    N = len(prob_list)
    TTM_perplexity = 1
    for prob in prob_list:
        TTM_perplexity *= 1 / pow(prob, 1 / N)
    return TTM_perplexity

### MAE Calculation
def Spectral_TTMPMF_MAE(predict, user_num, item_num, rating_matrix):
    MAE, count = 0, 0
    for i in range(user_num):
        for j in range(item_num):
            if rating_matrix[i][j] > 0:  # Only consider non-zero ratings
                wucha = abs(rating_matrix[i][j] - predict[i][j])
                MAE += wucha
                count += 1
    return MAE / count  # Return mean absolute error

