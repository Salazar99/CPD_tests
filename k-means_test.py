import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load the data
df = pd.read_csv('./tests/Engine_timing/Engine_timing.csv', sep=';', decimal=',')

# 2. Preprocessing (Standardizing physical state variables)
features = ['ThrottleAngle','LoadTorque','EngineSpeed']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Automatic "Elbow" Detection
# Calculate WCSS (Inertia) for different values of k
k_range = range(2, 9)
wcss = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Heuristic: Find the k point furthest from the line connecting start and end points
def find_elbow(k_values, wcss_values):
    p1 = np.array([k_values[0], wcss_values[0]])
    p2 = np.array([k_values[-1], wcss_values[-1]])
    
    distances = []
    for i in range(len(k_values)):
        p0 = np.array([k_values[i], wcss_values[i]])
        # Distance calculation from point to line
        dist = np.abs(np.cross(p2-p1, p1-p0)) / np.linalg.norm(p2-p1)
        distances.append(dist)
    return k_values[np.argmax(distances)]

optimal_k = find_elbow(list(k_range), wcss)
print(f"Optimal number of segments detected: {optimal_k}")

# 4. Apply Segmentation using the Optimal K
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['segment'] = kmeans_final.fit_predict(X_scaled)

# 5. Save the labeled data
df.to_csv('auto_segmented_results.csv', index=False)

# 6. Visualization of the Automated Result
plt.figure(figsize=(14, 10))
subset = df.head(1500)
colors = plt.cm.get_cmap('tab10', optimal_k)

features_to_plot = ['ThrottleAngle','LoadTorque','EngineSpeed']
for idx, feature in enumerate(features_to_plot, 1):
    plt.subplot(len(features_to_plot), 1, idx)
    for i in range(optimal_k):
        mask = subset['segment'] == i
        plt.scatter(subset.index[mask], subset.loc[mask, feature], 
                    label=f'Segment {i}', s=12, color=colors(i))
    plt.ylabel(feature)
    plt.legend(loc='upper right')
    if idx == 1:
        plt.title(f'Automated Multivariate Segmentation (k={optimal_k})')
    if idx == len(features_to_plot):
        plt.xlabel('Index')

plt.tight_layout()
plt.show()