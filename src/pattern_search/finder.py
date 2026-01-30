from tqdm import tqdm
import numpy as np
from numpy import ceil
from sympy import sign


def hash(prev,curr) -> int:
    A = 6364136223846793005	
    M = 2**64

    return (int(prev) * A + int(curr) + 1) % M

def aggregate_and_filter(hashes):
    patterns = {}
    #Remove small-length windows and keep only a representative of the pattern
    #Check all the windows that are centered around a closer value and keep only a representative of the group
    for element in tqdm(hashes.values(), desc="Aggregating and filtering patterns"):
        if element['number_of_occurrences'] > 2 and len(element['windows']) > 1:
            centers = []
            for window in element['windows']:
                center_value = window[0] + (window[1] // 2)
                centers.append(center_value)
            
            centers.sort()
            clusters = []
            current_cluster = [centers[0]]

            for i in range(1, len(centers)):
                if centers[i] - current_cluster[-1] <= 1:  # Adjust the threshold as needed
                    current_cluster.append(centers[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [centers[i]]
            clusters.append(current_cluster)

            representative_centers = [cluster[len(cluster) // 2] for cluster in clusters]
            new_windows = []
            for rep_center in representative_centers:
                new_windows.append((rep_center - (element['dimension'] // 2), element['dimension']))  
            element['cluster_centers'] = representative_centers
            element['windows'] = new_windows
            
            patterns[id(element)] = element
    
    for p in tqdm(patterns.values(), desc="Comparing patterns"):
        for w in patterns.values():
            if len(p['cluster_centers']) == len(w['cluster_centers']) and id(p) != id(w):
                match_bits = np.abs(np.array(p['cluster_centers']) - np.array(w['cluster_centers'])) <= 5
                if np.sum(match_bits) == len(p['cluster_centers']):
                    if p['dimension'] > w['dimension']:
                        patterns[id(w)]['windows'] = [] 
                    else: 
                        patterns[id(p)]['windows'] = []
    
    new_patterns = {}
    for element in patterns.values():
        if len(element['windows']) > 0:
            new_patterns[id(element)] = element
                             
    return new_patterns

def delta(trace, start_idx, n):
    """
    Calculates the raw difference between the current point 
    and the point n steps in the past.
    """
    # Safety check: if we are at the very start, delta is 0
    if start_idx < n:
        # Optional: return trace[start_idx] - trace[0] if you want partials
        return 0 
    
    delta = trace[start_idx] - trace[start_idx - n]
    
    return delta
    
"""Find patterns in the trace using a sliding window approach.
    The values used to compute the hash are derived from the delta between consecutive trace values.
    This allows to identify patterns that share similar topological features, regardless of their absolute values.
    Args:
        k: The size of the sliding window.
        trace: The trace data to search within.
"""
def sw_finder(k, trace):
    hashes = {}
    for start in tqdm(range(1, len(trace) - k + 1), desc="Hashing Signal"):
        rolling_hash = 0
        current = 0 
        
        while current <= k and (start + current) < len(trace):
            delta_sng = sign(delta(trace, start + current, current))
            
            rolling_hash = hash(rolling_hash, delta_sng)
            current += 1 
            #if(start == 0 or start == 20):
            #    print(f"Rolling hash at position {start} with window size {current}: {rolling_hash}")
            #    print(f"delta: {delta}, curr_val: {curr_val}")

            if rolling_hash in hashes:
                hashes[rolling_hash]['number_of_occurrences'] += 1
                if current != hashes[rolling_hash]['dimension']:
                    print(f"Warning: Hash collision detected for hash {rolling_hash} at position {start}. Previous dimension: {hashes[rolling_hash]['dimension']}, New dimension: {current}")
                #print(f"Pattern found: {start} to {start + k}  with hash {rolling_hash}")
                if(current >= 0.1*len(trace)):
                    hashes[rolling_hash]['windows'].append((start, current))
            else:
                hashes[rolling_hash] = {
                    'dimension': current,
                    'pattern': trace[start:start + current],
                    'number_of_occurrences': 1,
                    'windows': []
                }
                if(current >= 0.1*len(trace)):
                    hashes[rolling_hash]['windows'].append((start, current))
                
                
    print(f"Total unique patterns found: {len(hashes)}")
    print("Patterns details:")
    for h, details in hashes.items():
        print(f"Hash: {h}, Dimension: {details['dimension']}, Occurrences: {details['number_of_occurrences']}")

    return aggregate_and_filter(hashes)