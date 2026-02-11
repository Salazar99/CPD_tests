import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

WINDOW_SIZE = 1
STEP = 1
MAX_LAG = 1
N_CLUSTERS = 1

def autocorrelation(x, max_lag):
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode="full")
    corr = corr[corr.size // 2 :]
    corr = corr / corr[0]  
    return corr[:max_lag]

def sliding_windows(signal, window_size, step):
    windows = []
    indices = []
    for i in range(0, len(signal) - window_size, step):
        windows.append(signal[i : i + window_size])
        indices.append(i)
    return np.array(windows), np.array(indices)


def build_autocorr_features(signal, window_size, step, max_lag):
    windows, indices = sliding_windows(signal, window_size, step)
    features = np.array([autocorrelation(w, max_lag) for w in windows])
    return features, indices



#features, indices = build_autocorr_features(
#        signal,
#        window_size=WINDOW_SIZE,
#        step=STEP,
#        max_lag=MAX_LAG
#    )
#scaler = StandardScaler()
#X = scaler.fit_transform(features)
#kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0)
#labels = kmeans.fit_predict(X)

