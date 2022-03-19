import numpy as np


def compute_vec_pxx(x_signal: np.ndarray, y_signal: np.ndarray) -> float:

    numerator = 0.0
    mean_x = np.mean(x_signal, axis=0)
    mean_y = np.mean(y_signal, axis=0)
    rms_x = 0.0
    rms_y = 0.0

    for x, y in zip(x_signal, y_signal):

        x_diff = x - mean_x
        y_diff = y - mean_y
        numerator += np.dot(x_diff, y_diff)
        rms_x += np.linalg.norm(x_diff) ** 2
        rms_y += np.linalg.norm(y_diff) ** 2

    rms_x **= 0.5
    rms_y **= 0.5
    denominator = rms_x * rms_y

    return numerator / denominator


def rolling_pcc(x_signal: np.ndarray, y_signal: np.ndarray, window_length: int = 100) -> np.ndarray:

    length = np.shape(x_signal)[0]

    pcc_vals = np.zeros(length - window_length)

    for index in np.arange(0, length - window_length):

        pcc_vals[index] = compute_vec_pxx(x_signal[index:index+window_length], y_signal[index:index+window_length])

    return pcc_vals
