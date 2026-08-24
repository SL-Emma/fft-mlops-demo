import numpy as np
import scipy.fft as fft

def generate_signals(n_samples=200, duration=1.0, fs=1000):
    """
    Generates synthetic time-series signals with FFT features.
    Class 0: Low frequencies (50Hz, 120Hz)
    Class 1: High frequencies (200Hz, 300Hz)
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    X = []
    y = []
    
    for _ in range(n_samples):
        is_class_1 = np.random.choice([0, 1])
        
        if is_class_1 == 0:
            f1, f2 = 50, 120
        else:
            f1, f2 = 200, 300
            
        noise = np.random.normal(0, 0.5, len(t))
        signal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t) + noise
        
        # Compute FFT
        yf = fft.fft(signal)
        xf = fft.fftfreq(len(t), 1/fs)
        
        # Positive frequencies only
        pos_mask = xf > 0
        magnitudes = np.abs(yf[pos_mask])
        
        X.append(magnitudes)
        y.append(is_class_1)
    
    return np.array(X), np.array(y), xf[pos_mask], t

def extract_features(signal, fs):
    """Helper to extract FFT features from a single signal"""
    yf = fft.fft(signal)
    xf = fft.fftfreq(len(signal), 1/fs)
    pos_mask = xf > 0
    return np.abs(yf[pos_mask]), xf[pos_mask]
