import asyncio
from aiohttp import web
import json
import os
import numpy as np
from scipy.fft import fft
import math

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

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Web Audio Tuner</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            text-align: center;
            background-color: #333; /* Changed to dark theme */
        }
        .tuner {
            margin: 20px 0;
            padding: 20px;
            border: 1px solid #555; /* Changed to dark theme */
            border-radius: 10px;
        }
        .note {
            font-size: 48px;
            font-weight: bold;
            margin: 20px 0;
            color: #fff; /* Changed to dark theme */
        }
        .frequency {
            font-size: 24px;
            color: #ff7; /* Changed to dark theme */
        }
        .meter {
            width: 100%;
            height: 20px;
            background: #333; /* Changed to dark theme */
            margin: 20px auto;
            border-radius: 10px;
            position: relative;
        }
        .meter-value {
            height: 100%;
            background: #ff7; /* Changed to dark theme */
            border-radius: 10px;
            transition: width 0.2s;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            background: #ff7; /* Changed to dark theme */
            color: #333; /* Changed to dark theme */
            border: none;
            border-radius: 5px;
        }
        button:disabled {
            background: #555; /* Changed to dark theme */
        }
        canvas {
            width: 100%;
            height: 200px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="tuner">
        <div class="note">-</div>
        <div class="frequency">- Hz</div>
        <div class="meter">
            <div class="meter-value" style="width: 50%"></div>
        </div>
        <canvas id="waveform" width="100%" height="200"></canvas>
        <button id="startButton">Start Tuner</button>
    </div>

    <script>
        console.log('Script starting...');
        const noteFreqs = %s;
        let audioContext;
        let analyser;
        let dataArray;
        let isRunning = false;

        document.addEventListener('DOMContentLoaded', () => {
            console.log('DOM loaded');
            const startButton = document.getElementById('startButton');
            if (!startButton) {
                console.error('Start button not found!');
                return;
            }

            startButton.addEventListener('click', async () => {
                console.log('Button clicked');
                if (!isRunning) {
                    try {
                        await initAudio();
                        isRunning = true;
                        startButton.textContent = 'Stop Tuner';
                        analyze();
                    } catch (error) {
                        console.error('Error starting tuner:', error);
                        alert('Error starting tuner: ' + error.message);
                    }
                } else {
                    isRunning = false;
                    startButton.textContent = 'Start Tuner';
                }
            });
        });

        async function initAudio() {
            try {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 2048;
                
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const source = audioContext.createMediaStreamSource(stream);
                source.connect(analyser);
                
                dataArray = new Float32Array(analyser.frequencyBinCount);
            } catch (error) {
                console.error('Error initializing audio:', error);
                alert('Failed to initialize audio. Please make sure you have a microphone connected and have granted microphone permissions.');
                isRunning = false;
                document.getElementById('startButton').textContent = 'Start Tuner';
            }
        }

        function getNote(frequency) {
            return Object.entries(noteFreqs).reduce((closest, [note, freq]) => {
                return Math.abs(freq - frequency) < Math.abs(freq - closest.freq) 
                    ? {note, freq} 
                    : closest;
            }, {note: '', freq: Infinity});
        }

        function getCents(frequency, targetFreq) {
            return Math.floor(1200 * Math.log2(frequency / targetFreq));
        }

        function analyze() {
            if (!isRunning) return;

            analyser.getFloatTimeDomainData(dataArray);
            
            const frequency = detectPitch(dataArray, audioContext.sampleRate);
            
            if (frequency > 20) {  // Filter out very low frequencies
                const {note, freq: targetFreq} = getNote(frequency);
                const cents = getCents(frequency, targetFreq);
                
                document.querySelector('.note').textContent = note;
                document.querySelector('.frequency').textContent = `${frequency.toFixed(1)} Hz`;
                
                const meterValue = Math.max(0, Math.min(100, 50 + cents/2));
                document.querySelector('.meter-value').style.width = `${meterValue}%`;
            }

            drawWaveform();
            
            requestAnimationFrame(analyze);
        }

        function detectPitch(buffer, sampleRate) {
            const correlations = new Float32Array(buffer.length/2);
            
            for (let lag = 0; lag < correlations.length; lag++) {
                let sum = 0;
                for (let i = 0; i < correlations.length; i++) {
                    sum += buffer[i] * buffer[i + lag];
                }
                correlations[lag] = sum;
            }
            
            let maxCorrelation = 0;
            let maxLag = 0;
            
            for (let lag = 1; lag < correlations.length; lag++) {
                if (correlations[lag] > maxCorrelation) {
                    maxCorrelation = correlations[lag];
                    maxLag = lag;
                }
            }
            
            return sampleRate / maxLag;
        }

        function drawWaveform() {
            const canvas = document.getElementById('waveform');
            const ctx = canvas.getContext('2d');
            
            if (canvas.width !== canvas.clientWidth) {
                canvas.width = canvas.clientWidth;
            }
            if (canvas.height !== canvas.clientHeight) {
                canvas.height = canvas.clientHeight;
            }
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.beginPath();
            ctx.strokeStyle = '#ff7'; /* Changed to dark theme */
            ctx.lineWidth = 2;
            
            const sliceWidth = canvas.width / dataArray.length;
            let x = 0;
            
            for (let i = 0; i < dataArray.length; i++) {
                const y = (dataArray[i] + 1) * canvas.height / 2;
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
                
                x += sliceWidth;
            }
            
            ctx.stroke();
        }
    </script>
</body>
</html>
'''

async def handle_index(request):
    # Replace %s with a placeholder that won't conflict with HTML/CSS
    template = HTML_TEMPLATE.replace('%s', '###FREQUENCIES###')
    formatted_template = template.replace('###FREQUENCIES###', json.dumps(NOTE_FREQUENCIES))
    return web.Response(text=formatted_template, content_type='text/html')

app = web.Application()
app.router.add_get('/', handle_index)

if __name__ == '__main__':
    web.run_app(app, host='localhost', port=8080)
