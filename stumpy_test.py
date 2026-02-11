import numpy as np
from src.trace_container.trace_obj import Trace
import stumpy
import matplotlib.pyplot as plt

if __name__ == "__main__":
    tracepath1 = "./tests/pattern_test/test_1.csv"
    tracepath2 = "./tests/Engine_timing/Engine_timing.csv"
    tracepath3 = "./tests/Fuel_control/Fuel_Control.csv"
    tracepath4 = "./tests/WT/WT_Trace_Date_repetition.csv"
    
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
    #CPD visualization#
    # Throttle = engine_trace.get_column("ThrottleAngle")
    # Torque = engine_trace.get_column("LoadTorque")
    # Speed = engine_trace.get_column("EngineSpeed")
    print('#######################STATISTICS FOR FUEL CONTROL TRACE######################')
    fuel_control_trace = Trace(tracepath3)
    fuel_control_trace.display_head()
    stats = fuel_control_trace.get_statistics()
    print(stats)
    #CPD visualization#
    #time;air_fuel_ratio;speed;map;ego;throttle;fuel
    # Air_fuel_ratio = fuel_control_trace.get_column("air_fuel_ratio")
    # Speed = fuel_control_trace.get_column("speed")
    # Map = fuel_control_trace.get_column("map")
    # Ego = fuel_control_trace.get_column("ego")
    # Throttle = fuel_control_trace.get_column("throttle")
    # Fuel = fuel_control_trace.get_column("fuel")
    print('#######################STATISTICS FOR WT TRACE######################')
    wt_trace = Trace(tracepath4)
    wt_trace.display_head()
    stats = wt_trace.get_statistics()
    print(stats)
    
    
    
    
    ############ Multivariate TRACE ############
    engine_trace = engine_trace.get_columns(["ThrottleAngle", "LoadTorque", "EngineSpeed"])
    fuel_control_trace = fuel_control_trace.get_columns(["air_fuel_ratio", "speed", "map", "ego", "throttle", "fuel"])
    wt_trace_multivariate = wt_trace.get_columns(["z","d","rain","a"])  # Replace with actual column names
    wt_trace_univariate = wt_trace.get_column("rain")  # Replace with actual column names
    ###################################################### CLASP segmentation ######################################################
    
    ######### MULTIVARIATE ###############
    your_time_series = np.random.rand(10000)
    window_size = 500  # Approximately, how many data points might be found in a pattern

    matrix_profile = stumpy.stump(wt_trace_univariate, m=window_size)

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    # 1. Plot the Original Time Series
    ax[0].plot(wt_trace_univariate, color='navy', alpha=0.7)
    ax[0].set_title('Raw Time Series', fontsize=14)
    ax[0].set_ylabel('Value')

    # 2. Plot the Matrix Profile (Distance Profile)
    # We plot the first column: matrix_profile[:, 0]
    ax[1].plot(matrix_profile[:, 0], color='crimson')
    ax[1].set_title(f'Matrix Profile (m={window_size})', fontsize=14)
    ax[1].set_ylabel('Distance')
    ax[1].set_xlabel('Index')

    # Optional: Highlight the global minimum (the best motif pair)
    min_idx = np.argmin(matrix_profile[:, 0])
    ax[1].axvline(x=min_idx, color='black', linestyle='--', alpha=0.5)
    ax[1].annotate('Best Motif', xy=(min_idx, matrix_profile[min_idx, 0]), 
                xytext=(min_idx+100, matrix_profile[min_idx, 0]+2),
                arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.show()



    subseq_len = 250
    correct_arc_curve, regime_locations = stumpy.fluss(matrix_profile[:, 1],
                                                    L=subseq_len,
                                                    n_regimes=2
                                                    )
    

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # Plot the original data
    ax[0].plot(wt_trace_univariate)
    ax[0].set_title('Raw Time Series')

    # Mark the regime change
    for loc in regime_locations:
        ax[0].axvline(x=loc, color='r', linestyle='--')

    # Plot the Arc Curve
    ax[1].plot(correct_arc_curve)
    ax[1].set_title('FLUSS Arc Curve')
    ax[1].set_ylim(0, 1)

    plt.show()