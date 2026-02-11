from blinker import signal
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from src.trace_container.trace_obj import Trace
import numpy as np
import src.Segmentation.CPD as cpd_viz
import matplotlib.pyplot as plt
import src.pattern_search.finder as pattern_finder
from patternMatching import WINDOW_SIZE, autocorrelation
from patternMatching import sliding_windows
from patternMatching import build_autocorr_features


if __name__ == "__main__":
    tracepath1 = "./tests/pattern_test/test_1.csv"
    tracepath2 = "./tests/Engine_timing/Engine_timing.csv"
    
    # Visualization of raw trace data from CSV file
    print('#######################STATISTICS FOR TEST TRACE######################')
    test_trace = Trace(tracepath1)
    test_trace.display_head()
    stats = test_trace.get_statistics()
    print(stats)
    Signal = test_trace.get_column("amplitude")
    print('#######################STATISTICS FOR ENGINE TRACE######################')
    engine_trace = Trace(tracepath2)
    engine_trace.display_head()
    stats = engine_trace.get_statistics()
    print(stats)
    #CPD visualization
    Throttle = engine_trace.get_column("ThrottleAngle")
    Torque = engine_trace.get_column("LoadTorque")
    Speed = engine_trace.get_column("EngineSpeed")
    
    
    N_CLUSTERS = 10
    STEP = 50 
    MAX_LAG = 10
    features, indices = build_autocorr_features(
            Signal,
            window_size=WINDOW_SIZE,
            step=STEP,
            max_lag=MAX_LAG
        )
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0)
    labels = kmeans.fit_predict(X)
    
    print(f"Number of patterns found: {len(set(labels))}")
    for pattern in set(labels):
        pattern_indices = indices[labels == pattern]
        print(f"Pattern {pattern}: {len(pattern_indices)} occurrences")
        test_trace.plot_signal_with_windows("amplitude", pattern_indices, window_size=WINDOW_SIZE, step=STEP)
        test_trace.plot_high_density_summary("amplitude", pattern_indices, window_size=WINDOW_SIZE, step=STEP, filename=f"amplitude_pattern_{pattern}_summary.pdf")
    #n_bkps = 10
    #cpd_viz.detect_changepoints_l2_known(signal, n_bkps)
 
    #cpd_viz.detect_changepoints_dynp(signal, n_bkps)
    
    #cpd_viz.detect_changepoints_pelt(Throttle, pen=5)
    
    #cpd_viz.detect_changepoints_pelt(Torque, pen=5)
    
    #cpd_viz.detect_changepoints_pelt(Speed, pen=100000)
    
    #cpd_viz.detect_changepoints_binary_segmentation(signal, n_bkps)
    
    #cpd_viz.detect_changepoints_kernel(signal, n_bkps, kernel="linear")
    
    #cpd_viz.detect_changepoints_l2(signal)
    
    patterns = pattern_finder.sw_finder(len(Signal)//4, Signal)
    print(f"#Patterns found:{len(patterns)}")
    test_trace.plot_signal_with_windows("amplitude", patterns)
    test_trace.plot_high_density_summary("amplitude", patterns, filename="amplitude_summary.pdf")
 
    patterns = pattern_finder.sw_finder(len(Speed)//6, Speed)
    print(f"#Patterns found in Engine Speed:{len(patterns)}")
    # engine_trace.plot_signal_with_windows("EngineSpeed", patterns)
    #engine_trace.save_pattern_plot("EngineSpeed", patterns, filename="EngineSpeed_patterns.pdf")
    engine_trace.plot_high_density_summary("EngineSpeed", patterns, filename="EngineSpeed_summary.pdf")