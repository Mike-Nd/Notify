import numpy as np
import pyaudio
import scipy.signal
from scipy.fft import fft
from scipy.signal.windows import hann
import threading
import time

# Dictionary mapping frequencies to musical notes
NOTE_FREQUENCIES = {
    'C0': 16.35, 'C#0': 17.32, 'D0': 18.35, 'D#0': 19.45,
    'E0': 20.60, 'F0': 21.83, 'F#0': 23.12, 'G0': 24.50,
    'G#0': 25.96, 'A0': 27.50, 'A#0': 29.14, 'B0': 30.87,
    'C1': 32.70, 'C#1': 34.65, 'D1': 36.71, 'D#1': 38.89,
    'E1': 41.20, 'F1': 43.65, 'F#1': 46.25, 'G1': 49.00,
    'G#1': 51.91, 'A1': 55.00, 'A#1': 58.27, 'B1': 61.74,
    'C2': 65.41, 'C#2': 69.30, 'D2': 73.42, 'D#2': 77.78,
    'E2': 82.41, 'F2': 87.31, 'F#2': 92.50, 'G2': 98.00,
    'G#2': 103.83, 'A2': 110.00, 'A#2': 116.54, 'B2': 123.47,
    'C3': 130.81, 'C#3': 138.59, 'D3': 146.83, 'D#3': 155.56,
    'E3': 164.81, 'F3': 174.61, 'F#3': 185.00, 'G3': 196.00,
    'G#3': 207.65, 'A3': 220.00, 'A#3': 233.08, 'B3': 246.94,
    'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13,
    'E4': 329.63, 'F4': 349.23, 'F#4': 369.99, 'G4': 392.00,
    'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88
}

class Tuner:
    def __init__(self):
        self.CHUNK = 2048  # Buffer size
        self.RATE = 44100  # Sample rate
        self.running = False
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None
        
    def start(self):
        """Start the tuner"""
        self.running = True
        
        # Open audio stream
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        # Start processing thread
        self.thread = threading.Thread(target=self._process_audio)
        self.thread.start()
        
    def _process_audio(self):
        """Process incoming audio data"""
        while self.running:
            try:
                # Read audio data
                data = np.frombuffer(self.stream.read(self.CHUNK), dtype=np.float32)
                
                # Apply window function
                windowed = data * hann(len(data))
                
                # Compute FFT
                fft_data = fft(windowed)
                freqs = np.fft.fftfreq(len(fft_data), 1.0/self.RATE)
                
                # Find dominant frequency
                peak_freq = self._get_peak_frequency(freqs, fft_data)
                
                if peak_freq > 20:  # Filter out very low frequencies
                    # Get musical note
                    note = self._freq_to_note(peak_freq)
                    # Calculate tuning
                    cents = self._get_cents_deviation(peak_freq, note)
                    
                    status = "♯" if cents > 5 else "♭" if cents < -5 else "✓"
                    print(f"\rNote: {note} {status} | Frequency: {peak_freq:.1f}Hz | Cents: {cents:+.0f}   ", end="")
                    
            except Exception as e:
                print(f"\nError: {e}")
                break
                
    def _get_peak_frequency(self, freqs, fft_data):
        """Find the dominant frequency"""
        # Look at positive frequencies only
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        magnitude = np.abs(fft_data[pos_mask])
        
        # Find peak
        peak_idx = magnitude.argmax()
        return freqs[peak_idx]
    
    def _freq_to_note(self, frequency):
        """Convert frequency to closest musical note"""
        # Find the note with the closest frequency
        return min(NOTE_FREQUENCIES.items(), 
                  key=lambda x: abs(frequency - x[1]))[0]
    
    def _get_cents_deviation(self, freq, note):
        """Calculate cents deviation from perfect pitch"""
        perfect_freq = NOTE_FREQUENCIES[note]
        return 1200 * np.log2(freq / perfect_freq)
    
    def stop(self):
        """Stop the tuner"""
        self.running = False
        if self.thread:
            self.thread.join()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

if __name__ == "__main__":
    tuner = Tuner()
    print("Starting tuner... (Press Ctrl+C to stop)")
    try:
        tuner.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping tuner...")
        tuner.stop()