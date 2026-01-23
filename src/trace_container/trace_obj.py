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