import numpy as np
from functools import reduce

def TTM_predict_ALS(E, H, k1, m, n, max_iter, lamb, user_num, item_num, rating_matrix):
    # Create a binary matrix indicating which ratings are present
    is_rating = np.where(rating_matrix != 0, 1, 0)
    rating_matrix_T = rating_matrix.T

    for i in range(len(lamb)):
        l1 = lamb[i]
        for j in range(len(lamb)):
            l2 = lamb[j]
            user_persona = E.T  # Transpose of user latent features
            item_nature = H.T    # Transpose of item latent features
            
            # Initialize A matrix with random values
            A = np.random.normal(loc=0, scale=1, size=(m, n))
            I1, I2, I3 = np.identity(m), np.identity(n), np.identity(m * n)
            F = np.zeros((user_num * item_num, user_num * item_num))
            R = rating_matrix.reshape(1, user_num * item_num)  # Flatten the rating matrix
            
            for iter_num in range(max_iter):  # Iterate for a fixed number of times
                # Update user latent features
                for user_index in range(user_num):  # Iterate over each user
                    Fi = np.zeros((item_num, item_num))
                    start_index = user_index * item_num
                    end_index = (user_index + 1) * item_num
                    for index in range(item_num):
                        Fi[index][index] = is_rating[user_index][index]
                    F[start_index:end_index, start_index:end_index] = Fi
                    
                    # Calculate updated user latent features
                    matrix_1 = [A, H, Fi, H.T, A.T]
                    matrix_2 = [A, H, Fi, rating_matrix[user_index]]
                    ui = np.dot(np.linalg.inv(reduce(np.dot, matrix_1) + l1 * I1), 
                                 (reduce(np.dot, matrix_2) + l1 * user_persona[user_index]))
                    user_persona[user_index] = ui.T  # Update user persona

                E = user_persona.T  # Update the E matrix with transposed user personas

                # Update item latent features
                for item_index in range(item_num):
                    Fj = np.zeros((user_num, user_num))
                    for index in range(user_num):
                        Fj[index][index] = is_rating[index][item_index]
                    matrix_3 = [A.T, E, Fj, E.T, A]
                    matrix_4 = [A.T, E, Fj, rating_matrix_T[item_index].T]
                    vj = np.dot(np.linalg.inv(reduce(np.dot, matrix_3) + l2 * I2), 
                                 (reduce(np.dot, matrix_4) + l2 * item_nature[item_index].T))
                    item_nature[item_index] = vj.T  # Update item nature

                H = item_nature.T  # Update the H matrix with transposed item natures

                # Create gamma matrix for factorization
                gamma_matrix = np.zeros((m * n, user_num * item_num))
                for user_index in range(user_num):
                    for p in range(m):
                        for item_index in range(item_num):
                            for q in range(n):
                                gamma_matrix[n * p + q][item_num * user_index + item_index] = E[p][user_index] * H[q][item_index]

                matrix_5 = [gamma_matrix, F, gamma_matrix.T] 
                tmp = np.linalg.inv(reduce(np.dot, matrix_5) + I3) 
                matrix_6 = [tmp, gamma_matrix, F, R.T] 
                A1 = reduce(np.dot, matrix_6) 

                # Update A matrix with the calculated values
                for p in range(m):
                    for q in range(n):
                        A[p][q] = A1[n * p + q] 

                # Compute the predicted scores
                matri = [E.T, A, H] 
                predict_score = reduce(np.dot, matri) 
                return predict_score  # Return the predicted scores
