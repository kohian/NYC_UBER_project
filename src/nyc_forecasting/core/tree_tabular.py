import numpy as np



def make_tabular_next_step_dataset(
    feature_array: np.ndarray,
    target_array: np.ndarray,
    input_len: int,
    flatten_sequences: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    total = len(feature_array) - input_len

    for idx in range(total):
        x_seq = feature_array[idx : idx + input_len]
        y_next = target_array[idx + input_len]
        xs.append(x_seq.reshape(-1) if flatten_sequences else x_seq)
        ys.append(y_next)

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
