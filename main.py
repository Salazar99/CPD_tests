from src.trace_visualizer.viewer import TraceVisualizer

if __name__ == "__main__":
    # Example usage
    visualizer = TraceVisualizer("./tests/Engine_timing/Engine_timing.csv")
    visualizer.display_head()
    visualizer.plot_column("EngineSpeed", title="Engine Speed Over Time")
    visualizer.plot_columns(["ThrottleAngle", "LoadTorque","EngineSpeed"], title="All Engine Parameters")
    stats = visualizer.get_statistics()
    print(stats)