import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt


class Trace:
    """A class to visualize trace data from CSV files."""
    
    def __init__(self, csv_file: str):
        """
        Initialize the visualizer with a CSV file.
        
        Args:
            csv_file: Path to the CSV file to load
        """
        self.csv_file = Path(csv_file)
        self.df = None
        self.load_data()
    
    def load_data(self) -> None:
        """Load the CSV file into a pandas DataFrame."""
        if not self.csv_file.exists():
            raise FileNotFoundError(f"File not found: {self.csv_file}")
        self.df = pd.read_csv(self.csv_file, sep=';',decimal=',')
        #cast all column to float 
        for col in self.df.columns:
            self.df[col] = self.df[col].astype(float)
        
        print(f"Loaded {len(self.df)} rows from {self.csv_file}")
    
    def display_head(self) -> None:
        """Display the first n rows of the dataframe."""
        print(self.df.head().to_string() + "\n" + self.df.dtypes.to_string())
    
    def plot_column(self, column: str, title: str = None) -> None:
        """Plot a single column as a line chart."""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")
        plt.figure(figsize=(10, 6))
        x_data = self.df["time"] if "time" in self.df.columns else range(len(self.df))
        plt.plot(x_data, self.df[column])
        plt.title(title or f"Trace: {column}")
        plt.xlabel("time" if "time" in self.df.columns else "Index")
        plt.ylabel(column)
        plt.grid(True)
        plt.show()
    
    def plot_columns(self, columns: list, title: str = None) -> None:
        """Plot multiple columns on the same chart."""
        plt.figure(figsize=(10, 6))
        x_data = self.df["time"] if "time" in self.df.columns else range(len(self.df))
        for col in columns:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe")
            plt.plot(x_data, self.df[col], label=col)
        plt.title(title or "Trace Visualization")
        plt.xlabel("time" if "time" in self.df.columns else "Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def get_statistics(self) -> pd.DataFrame:
        """Return descriptive statistics for numeric columns."""
        return self.df.describe()
    
    def get_column(self, column: str) -> np.ndarray:
        """Return a specific column from the dataframe as a numpy array."""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")
        return self.df[column].to_numpy()
    
    def get_columns(self, columns: list) -> np.ndarray:
        """Return specific columns from the dataframe as a numpy array."""
        for column in columns:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in dataframe")
        return self.df[columns].to_numpy()
    
    def save_pattern_plot(self, signal, pattern_dict, filename="pattern_analysis.pdf"):
        """
        Saves subplots to a file. Supports .pdf, .svg (vector) or .png, .jpg (raster).
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np

        unique_hashes = list(pattern_dict.keys())
        n_subplots = len(unique_hashes)

        if n_subplots == 0:
            print("No patterns found to plot.")
            return

        # 1. Color mapping based on unique lengths
        all_lengths = sorted(list(set(length for m in pattern_dict.values() for _, length in m['windows'])))
        colors = cm.get_cmap('tab10', len(all_lengths)) # tab10 is great for distinct categories
        length_to_color = {length: colors(i) for i, length in enumerate(all_lengths)}

        # Adjust height dynamically: 3 inches per subplot
        fig, axes = plt.subplots(n_subplots, 1, figsize=(14, 3 * n_subplots), 
                                 squeeze=False, sharex=True)
        axes = axes.flatten()

        signal_data = self.get_column(signal)

        for i, rolling_hash in enumerate(unique_hashes):
            ax = axes[i]
            metadata = pattern_dict[rolling_hash]
            windows = metadata['windows']

            ax.plot(signal_data, color='black', linewidth=0.7, alpha=0.3, label='Signal')

            for start, length in windows:
                p_color = length_to_color[length]
                ax.axvspan(start, start + length, facecolor=p_color, 
                           alpha=0.4, edgecolor='red', linewidth=0.5)

            ax.set_title(f"Hash: {rolling_hash} | Length: {windows[0][1]} | Count: {len(windows)}", loc='left')
            ax.set_ylabel("Amp")
            ax.grid(True, linestyle='--', alpha=0.3)

        plt.xlabel("Time Index")

        # Use tight_layout with padding to prevent label clipping in the saved file
        plt.tight_layout(pad=2.0)

        # Save the file
        # For vector graphics, use .pdf or .svg
        # For high-res raster, use .png with dpi=300
        plt.savefig(filename, format=None, bbox_inches='tight')
        print(f"Plot successfully saved to: {filename}")

        # Close the plot to free up memory (very important for large automated runs)
        plt.close(fig)
    
    def plot_signal_with_windows(self, signal, pattern_dict):
        
    
       
        """
        Subplots separated by Pattern Hash, with colors determined by window length.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np

        unique_hashes = list(pattern_dict.keys())
        n_subplots = len(unique_hashes)

        if n_subplots == 0:
            return

        # 1. Create a color mapping based on all unique lengths present in the dict
        all_lengths = sorted(list(set(length for m in pattern_dict.values() for _, length in m['windows'])))
        # Generate a distinct color for each unique length
        colors = cm.get_cmap('viridis', len(all_lengths))
        length_to_color = {length: colors(i) for i, length in enumerate(all_lengths)}

        fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 4 * n_subplots), 
                                 squeeze=False, sharex=True)
        axes = axes.flatten()

        signal_data = self.get_column(signal)

        for i, rolling_hash in enumerate(unique_hashes):
            ax = axes[i]
            metadata = pattern_dict[rolling_hash]
            windows = metadata['windows']
            current_length = windows[0][1]

            # Pick the color assigned to this specific length
            pattern_color = length_to_color[current_length]

            ax.plot(signal_data, color='black', linewidth=1, alpha=0.5, label='Signal')

            for start, length in windows:
                ax.axvspan(start, start + length, facecolor=pattern_color, 
                           alpha=0.4, edgecolor='red')

            # Title includes the Hash and the Length to clarify the color coding
            ax.set_title(f"Hash: {rolling_hash} | Length: {current_length} | Occurrences: {len(windows)}")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.2)

        plt.xlabel("Time Index")
        plt.tight_layout()
        plt.show()
        
    def plot_high_density_summary(self, signal, pattern_dict, filename="summary.pdf"):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    
        signal_data = self.get_column(signal)
        unique_hashes = list(pattern_dict.keys())
        
        # Create two subplots: one for signal, one for the "barcode" map
        fig, (ax_sig, ax_map) = plt.subplots(2, 1, figsize=(15, 8), 
                                             gridspec_kw={'height_ratios': [3, 1]},
                                             sharex=True)
    
        ax_sig.plot(signal_data, color='black', alpha=0.3, linewidth=0.5)
        
        # Generate colors for each hash
        colors = cm.get_cmap('tab20', len(unique_hashes))
        
        for i, h in enumerate(unique_hashes):
            windows = pattern_dict[h]['windows']
            # Plot markers on the signal
            for start, length in windows:
                ax_sig.axvspan(start, start+length, color=colors(i), alpha=0.1)
                
                # Plot "barcode" ticks in the map subplot
                ax_map.vlines(start, i, i+0.8, colors=colors(i), linewidth=1)
    
        ax_map.set_yticks(range(len(unique_hashes)))
        ax_map.set_yticklabels([f"H:{str(h)[:6]}..." for h in unique_hashes])
        ax_map.set_title("Pattern Occurrence Map (Barcode View)")
        ax_sig.set_title("Signal with Pattern Overlays")
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()