from src.trace_container.trace_obj import Trace
import src.Segmentation.CPD as cpd_viz
import matplotlib.pyplot as plt

if __name__ == "__main__":
    tracepath = "./tests/Engine_timing/Engine_timing.csv"
    
    # Visualization of raw trace data from CSV file
    trace = Trace(tracepath)
    trace.display_head()
    trace.plot_column("EngineSpeed", title="Engine Speed Over Time")
    trace.plot_columns(["ThrottleAngle", "LoadTorque","EngineSpeed"], title="All Engine Parameters")
    stats = trace.get_statistics()
    print(stats)
    
    #CPD visualization
    Throttle = trace.get_column("ThrottleAngle")
    Torque = trace.get_column("LoadTorque")
    Speed = trace.get_column("EngineSpeed")
    
    n_bkps = 10
    #cpd_viz.detect_changepoints_l2_known(signal, n_bkps)
 
    #cpd_viz.detect_changepoints_dynp(signal, n_bkps)
    
    #cpd_viz.detect_changepoints_pelt(Throttle, pen=5)
    
    #cpd_viz.detect_changepoints_pelt(Torque, pen=5)
    
    cpd_viz.detect_changepoints_pelt(Speed, pen=100000)
    
    #cpd_viz.detect_changepoints_binary_segmentation(signal, n_bkps)
    
    #cpd_viz.detect_changepoints_kernel(signal, n_bkps, kernel="linear")
    
    #cpd_viz.detect_changepoints_l2(signal)
    