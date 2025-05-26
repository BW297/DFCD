import numpy as np


def calculate_doa_k_block(mas_level, q_matrix, r_matrix, k, block_size=50):
    n_students, _ = mas_level.shape
    numerator = 0
    denominator = 0
    question_hask = np.where(q_matrix[:, k] != 0)[0].tolist()
    for start in range(0, n_students, block_size):
        end = min(start + block_size, n_students)
        delta_matrix_block = mas_level[start:end, k].reshape(-1, 1) > mas_level[start:end, k].reshape(1, -1)
        r_matrix_block = r_matrix[start:end, :]
        for j in question_hask:
            row_vector = (r_matrix_block[:, j].reshape(1, -1) != -1).astype(int)
            columen_vector = (r_matrix_block[:, j].reshape(-1, 1) != -1).astype(int)
            mask = row_vector * columen_vector
            delta_r_matrix = r_matrix_block[:, j].reshape(-1, 1) > r_matrix_block[:, j].reshape(1, -1)
            I_matrix = r_matrix_block[:, j].reshape(-1, 1) != r_matrix_block[:, j].reshape(1, -1)
            numerator_ = np.logical_and(mask, delta_r_matrix)
            denominator_ = np.logical_and(mask, I_matrix)
            numerator += np.sum(delta_matrix_block * numerator_)
            denominator += np.sum(delta_matrix_block * denominator_)
    if denominator == 0:
        DOA_k = 0
    else:
        DOA_k = numerator / denominator
    return DOA_k
