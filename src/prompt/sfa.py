# Symbolic Fourier Approximation
# %%
from glob import glob
import numpy as np
from pyts.approximation import SymbolicFourierApproximation as SFA

from data_processing import load_npy_file

def stack_iq(batch):
    # batch shape: (N, 2, L)
    return np.hstack([batch[:, 0, :], batch[:, 1, :]])

# %%
if __name__ == '__main__':
    signal_paths = list(filter(lambda x: 'embedding' not in x, glob('../../data/RadioML/*/train/*/*.npy')))

    X_train = np.array([load_npy_file(sig) for sig in signal_paths])

# %%
    train_signals = np.hstack([X_train[:, :, 0], X_train[:, :, 1]])
    n_timestamps = train_signals.shape[1]   # 2048 for 1024-sample IQ


    sfa = SFA(
        n_coefs=24,
        n_bins=64,
        strategy='normal',           
        anova=True,                  
        norm_mean=True, norm_std=True,
    )

    X_train_words = sfa.fit_transform(train_signals)
    # X_test_words  = sfa.transform(X_test_1d)
    print("Example word sequence for one sample:", X_train_words[0])
    print(len(X_train_words[0]))

# %%
