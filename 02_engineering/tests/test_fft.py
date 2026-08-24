import numpy as np
import pytest
import sys
import os

# Add src to path to allow running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fft_processor import generate_signals

def test_signal_generation_shape():
    n_samples = 10
    fs = 1000
    X, y, frequencies, time = generate_signals(n_samples=n_samples, fs=fs)
    assert X.shape[0] == n_samples
    assert y.shape[0] == n_samples

def test_frequencies_are_positive():
    _, _, frequencies, _ = generate_signals(n_samples=5)
    assert np.all(frequencies > 0)
