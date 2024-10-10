import numpy as np
import random
from collections import Counter
from itertools import combinations, permutations

def svd_decomposition(matrix, k):
    """Perform SVD decomposition and return the top k components."""
    U, S, V = np.linalg.svd(matrix) 
    U_k = U[:, :k] 
    S_k = np.diag(S[:k]) 
    V_k = V[:k, :] 
    return U_k, S_k, V_k

def tensor_matrix_multiply(tensor, matrix1, matrix2, matrix3):
    """Multiply a tensor with three matrices using tensordot."""
    result = np.tensordot(tensor, matrix1, axes=([0], [0])) 
    result = np.tensordot(result, matrix2, axes=([0], [0]))
    result = np.tensordot(result, matrix3, axes=([0], [0])) 
    return result

def compute_moments_matrix(docs, word_to_idx):
    """Calculate and normalize second-order moments as matrices."""
    matrices = []
    for doc in docs:
        matrix = np.zeros((len(word_to_idx), len(word_to_idx)))
        for word_pair, count in Counter(combinations(doc, 2)).items():
            idx1, idx2 = word_to_idx[word_pair[0]], word_to_idx[word_pair[1]]
            matrix[idx1, idx2] = count / (len(doc) * (len(doc) - 1) / 2)  # Normalize
            matrix[idx2, idx1] = matrix[idx1, idx2]  # Undirected
        matrices.append(matrix)
    return matrices

def compute_moments_tensor(docs, word_to_idx):
    """Calculate and normalize third-order moments as tensors."""
    tensors = []
    for doc in docs:
        tensor = np.zeros((len(word_to_idx), len(word_to_idx), len(word_to_idx)))
        for word_triple, count in Counter(combinations(doc, 3)).items():
            idx1, idx2, idx3 = word_to_idx[word_triple[0]], word_to_idx[word_triple[1]], word_to_idx[word_triple[2]]
            tensor[idx1, idx2, idx3] = count / (len(doc) * (len(doc) - 1) * (len(doc) - 2) / 6)
            # Update all permutations for undirected tensor
            for perm in permutations([idx1, idx2, idx3]):
                tensor[perm] = tensor[idx1, idx2, idx3]
        tensors.append(tensor)
    return tensors

def tensor_spectral_algorithm(M, K, L, N):
    """Run the tensor spectral algorithm to compute alpha values."""
    phi_values = np.random.rand(L, K, K) 
    for i in range(L):
        for j in range(K):
            phi_values[i, j, :] /= np.linalg.norm(phi_values[i, j, :]) 
    for i in range(L):
        for j in range(N):
            phi_j = phi_values[i]
            tensor_product = np.einsum('ijk,jl,kl->il', M, phi_j, phi_j) 
            for k in range(K):
                tensor_product[k, :] /= np.linalg.norm(tensor_product[k, :]) 
            phi_values[i] = tensor_product 
    max_value = -np.inf 
    i_star = -1 
    for i in range(L):
        phi = phi_values[i]
        value = np.einsum('ijk,jk,kk->', M, phi, phi) 
        if value > max_value:
            max_value = value 
            i_star = i  # Use the best phi as initial value for further iterations 
    phi_hat = phi_values[i_star] 
    for j in range(N):
        tensor_product = np.einsum('ijk,jl,kl->il', M, phi_hat, phi_hat) 
        for k in range(K):
            tensor_product[k, :] /= np.linalg.norm(tensor_product[k, :]) 
        phi_hat = tensor_product  # Calculate lambda_hat
    lambda_hat = np.einsum('ijk,jk,kk->i', M, phi_hat, phi_hat) 
    # Calculate alpha_0 and alpha 
    alpha_0 = np.sum(lambda_hat ** 2)
    alpha = 4 * alpha_0 * (alpha_0 + 1) / (alpha_0 + 2) ** 2 * lambda_hat ** 2 
    return alpha

class TTM_Gibbs_Sampling_opti:
    def __init__(self, K, P, Q, T, I, J, M, alpha, beta, xi, rho, user_data, item_data, text_data, max_iter):
        """Initialize the Gibbs Sampling for the topic tensor model."""
        self.K = K
        self.P = P
        self.Q = Q
        self.T = T
        self.I = I
        self.J = J
        self.M = M
        self.alpha = alpha
        self.beta = beta
        self.xi = xi
        self.rho = rho
        self.user_data = user_data
        self.item_data = item_data
        self.text_data = text_data
        self.max_iter = max_iter

        # Pre-compute sum of alpha and beta for efficiency
        self.sum_alpha = sum(self.alpha)
        self.sum_beta = np.sum(self.beta, axis=1)

        # Initialize matrices
        self.get_feature_matrix()
        self.initial_matrix()

    def get_feature_matrix(self):
        """Randomly assign persona, nature, and topic to each text."""
        self.persona = [[random.randint(0, self.P - 1) for _ in text] for text in self.text_data]
        self.nature = [[random.randint(0, self.Q - 1) for _ in text] for text in self.text_data]
        self.topic = [[random.randint(0, self.K - 1) for _ in text] for text in self.text_data]

    def initial_matrix(self):
        """Initialize user-persona, item-nature, and topic distributions."""
        self.user_persona = np.zeros((self.P, self.I))
        self.user_persona_vector = np.zeros(self.I)
        self.item_nature = np.zeros((self.Q, self.J))
        self.item_nature_vector = np.zeros(self.J)
        self.persona_nature_topic = np.zeros((self.K, self.P, self.Q))
        self.persona_nature_topic_matrix = np.zeros((self.P, self.Q))
        self.topic_vocab = np.zeros((self.T, self.K))
        self.topic_vocab_vector = np.zeros(self.K)

        for i, text in enumerate(self.text_data):
            for j, v in enumerate(text):
                z, x, y = self.topic[i][j], self.persona[i][j], self.nature[i][j]
                user_id, item_id = self.user_data[i], self.item_data[i]

                # Update counts
                np.add.at(self.user_persona, (x, user_id), 1)
                self.user_persona_vector[user_id] += 1
                np.add.at(self.item_nature, (y, item_id), 1)
                self.item_nature_vector[item_id] += 1
                np.add.at(self.persona_nature_topic, (z, x, y), 1)
                np.add.at(self.persona_nature_topic_matrix, (x, y), 1)
                np.add.at(self.topic_vocab, (v, z), 1)
                self.topic_vocab_vector[z] += 1

    def posterior_prob(self, persona, nature, topic, vocab, user, item):
        """Compute posterior probabilities for persona, nature, and topic."""
        persona_probs = [
            (self.persona_nature_topic[topic][p][nature] + self.alpha[topic]) / 
            (self.persona_nature_topic_matrix[p][nature] + self.sum_alpha) * 
            (self.user_persona[p][user] + self.xi[p]) for p in range(self.P)
        ]

        nature_probs = [
            (self.persona_nature_topic[topic][persona][q] + self.alpha[topic]) / 
            (self.persona_nature_topic_matrix[persona][q] + self.sum_alpha) * 
            (self.item_nature[q][item] + self.rho[q]) for q in range(self.Q)
        ]

        topic_probs = [
            (self.topic_vocab[vocab][k] + self.beta[k][vocab]) / 
            (self.topic_vocab_vector[k] + self.sum_beta[k]) * 
            (self.persona_nature_topic[k][persona][nature] + self.alpha[k]) for k in range(self.K)
        ]

        return persona_probs, nature_probs, topic_probs

    def search_posterior(self, prob_list):
        """Sample a new assignment based on the computed probabilities."""
        sum_prob = sum(prob_list)
        judge = random.uniform(0, sum_prob)
        res = 0
        for index, prob in enumerate(prob_list):
            res += prob
            if judge <= res:
                return index

    def TTM_Gibbs(self):
        """Perform Gibbs sampling iterations."""
        Gibbs_result = []

        for iter_num in range(self.max_iter):
            for i, text in enumerate(self.text_data):
                for j, v in enumerate(text):
                    old_persona, old_nature, old_topic = self.persona[i][j], self.nature[i][j], self.topic[i][j]
                    user_id, item_id = self.user_data[i], self.item_data[i]

                    # Decrement counts
                    np.add.at(self.user_persona, (old_persona, user_id), -1)
                    self.user_persona_vector[user_id] -= 1
                    np.add.at(self.item_nature, (old_nature, item_id), -1)
                    self.item_nature_vector[item_id] -= 1
                    np.add.at(self.persona_nature_topic, (old_topic, old_persona, old_nature), -1)
                    np.add.at(self.persona_nature_topic_matrix, (old_persona, old_nature), -1)
                    np.add.at(self.topic_vocab, (v, old_topic), -1)
                    self.topic_vocab_vector[old_topic] -= 1

                    # Compute posterior probabilities
                    posterior_prob_persona, posterior_prob_nature, posterior_prob_topic = self.posterior_prob(old_persona, old_nature, old_topic, v, user_id, item_id)

                    # Sample new values
                    new_persona = self.search_posterior(posterior_prob_persona)
                    new_nature = self.search_posterior(posterior_prob_nature)
                    new_topic = self.search_posterior(posterior_prob_topic)

                    # Increment counts with new values
                    np.add.at(self.user_persona, (new_persona, user_id), 1)
                    self.user_persona_vector[user_id] += 1
                    np.add.at(self.item_nature, (new_nature, item_id), 1)
                    self.item_nature_vector[item_id] += 1
                    np.add.at(self.persona_nature_topic, (new_topic, new_persona, new_nature), 1)
                    np.add.at(self.persona_nature_topic_matrix, (new_persona, new_nature), 1)
                    np.add.at(self.topic_vocab, (v, new_topic), 1)
                    self.topic_vocab_vector[new_topic] += 1

                    # Update assignments
                    self.persona[i][j], self.nature[i][j], self.topic[i][j] = new_persona, new_nature, new_topic

            if iter_num >= self.max_iter-50:
                Gibbs_result.append([
                    self.user_persona, self.user_persona_vector, 
                    self.item_nature, self.item_nature_vector, 
                    self.persona_nature_topic, self.persona_nature_topic_matrix,
                    self.topic_vocab, self.topic_vocab_vector
                ])

        return Gibbs_result
    def calculate_parameter(self, Gibbs_result):
        self.epsilon = np.zeros((self.P, self.I))
        self.eta = np.zeros((self.Q, self.J))
        self.theta = np.zeros((self.K, self.P, self.Q))
        self.phi = np.zeros((self.T, self.K))
        for i in range(self.I):
            for p in range(self.P):
                self.epsilon[p][i] = (Gibbs_result[0][0][p][i]+self.xi[p])/(Gibbs_result[0][1][i]+sum(self.xi))
        for j in range(self.J):
            for q in range(self.Q):
                self.eta[q][j] = (Gibbs_result[0][2][q][j]+self.rho[q])/(Gibbs_result[0][3][j]+sum(self.rho))
        for p in range(self.P):
            for q in range(self.Q):
                for k in range(self.K):
                    self.theta[k][p][q] = (Gibbs_result[0][4][k][p][q]+self.alpha[k])/(Gibbs_result[0][5][p][q]+sum(self.alpha))
        for k in range(self.K):
            for t in range(self.T):
                self.phi[t][k] = (Gibbs_result[0][6][t][k]+self.beta[k][t])/(Gibbs_result[0][7][k]+sum(self.beta[0]))
        return self.epsilon, self.eta, self.theta, self.phi

