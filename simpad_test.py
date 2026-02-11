from src.trace_container.trace_obj import Trace
import numpy as np
import matplotlib.pyplot as plt
import os
from src.mSIMPAD.SIMPAD import SSPDetector as ssp

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
    
    ##1-Engine control##
    #clasp = BinaryClaSPSegmentation()
    #clasp.fit_predict(engine_trace)
    # 
    ##Plotting the segmentation results
    #clasp.plot(heading="Segmentation of Engine control signals", ts_name="ACC", file_path="Engine_segmentation.png")
    # 
    # 
    #2-Fuel control##
    # clasp = BinaryClaSPSegmentation()
    # clasp.fit_predict(fuel_control_trace)
    # 
    # Plotting the segmentation results
    # clasp.plot(heading="Segmentation of Fuel control signals", ts_name="ACC", file_path="Fuel_control_segmentation.png")
    
    
    #3-WT trace univariate##
    
    detection = ssp.SIMPAD(wt_trace_univariate, l=500, m=100)
    fig, axes = plt.subplots(2, 1, figsize=(12,3), dpi=300, sharex=True)
    axes[0].plot(wt_trace_univariate)
    axes[1].plot(detection)
    plt.show()
    plt.close(fig)
