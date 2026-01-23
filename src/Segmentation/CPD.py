from blinker import signal
import ruptures as rpt
import numpy as np
import matplotlib.pyplot as plt

def detect_changepoints_dynp(signal, n_bkps, pen=None, epsilon=None):
    """Detect changepoints using Dynamic Programming algorithm."""
    print("detecting dynp changepoints")
    algo = rpt.Dynp(min_size=2, jump=1).fit(signal)
    bkps = algo.predict(n_bkps=n_bkps)
    rpt.display(signal, bkps)
    plt.show()
    print("plotting dynp changepoints")
    
def detect_changepoints_pelt(signal, pen=1):
    """Detect changepoints using PELT algorithm."""
    print("detecting pelt changepoints")
    algo = rpt.Pelt(min_size=2, jump=1).fit(signal)
    bkps = algo.predict(pen=pen)
    rpt.display(signal, bkps)
    plt.show()
    print("plotting pelt changepoints")
    

def detect_changepoints_binary_segmentation(signal, n_bkps):
    """Detect changepoints using Binary Segmentation algorithm."""
    print("detecting binary segmentation changepoints")
    algo = rpt.Binseg(min_size=2, jump=1).fit(signal)
    bkps = algo.predict(n_bkps=n_bkps)
    rpt.display(signal, bkps)
    plt.show()
    print("plotting binary segmentation changepoints")
    

def detect_changepoints_kernel(signal, n_bkps, kernel="linear"):
    """Detect changepoints using Kernel Change Point algorithm."""
    print("detecting kernel changepoints")
    algo = rpt.KernelCPD(kernel=kernel, min_size=2, jump=1).fit(signal)
    bkps = algo.predict(n_bkps=n_bkps)
    rpt.display(signal, bkps)
    plt.show()
    print("plotting kernel changepoints")

def detect_changepoints_l2(signal):
    sigma = 5.0
    """Detect changepoints using L2 (Least Squares) algorithm."""
    print("detecting l2 changepoints")
    algo = rpt.Binseg(model="l2", min_size=2, jump=1).fit(signal)
    bkps = algo.predict(pen=np.log(3 * len(signal) * sigma**2))
    rpt.display(signal, bkps)
    plt.show()
    print("plotting l2 changepoints")
    
def detect_changepoints_l2_known(signal, n_bkps):
    sigma = 5.0
    """Detect changepoints using L2 (Least Squares) algorithm."""
    print("detecting l2 changepoints with known number of breakpoints")
    algo = rpt.Binseg(model="l2", min_size=2, jump=1).fit(signal)
    bkps = algo.predict(n_bkps=n_bkps)
    rpt.display(signal, bkps)
    plt.show()
    print("plotting l2 changepoints with known number of breakpoints")