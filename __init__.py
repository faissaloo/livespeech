#!/usr/bin/env python3
import pyaudio
import wave
import audioop
from .wavTranscriber import *
import numpy as np
from collections import deque
import os
import time
import math

LANG_CODE = 'en-US'  # Language to use

# Microphone stream config.
CHUNK = 16  # CHUNKS of bytes to read each time from mic 12 and 16 seemed gud
THRESHOLD = 15000  # The threshold intensity that defines silence
                  # and noise signal (an int. lower than THRESHOLD is silence).

SILENCE_LIMIT = 1  # Silence limit in seconds. The max ammount of seconds where
                   # only silence is recorded. When this time passes the
                   # recording finishes and the file is delivered.

PREV_AUDIO = 0.5  # Previous audio (in seconds) to prepend. When noise
                  # is detected, how much of previously recorded audio is
                  # prepended. This helps to prevent chopping the beggining
                  # of the phrase.

def listen_for_speech(**kwargs):
    threshold=kwargs.get('threshold', THRESHOLD)
    num_phrases=kwargs.get('num_phrases', -1)
    model_dir=kwargs.get('model_dir', 'models')

    # Event hooks
    on_load=kwargs.get('on_load', lambda *_: None)
    on_ready=kwargs.get('on_ready', lambda *_: None)
    on_record_start=kwargs.get('on_record_start', lambda *_: None)
    on_record_end=kwargs.get('on_record_end', lambda *_: None)
    on_transcription=kwargs.get('on_transcription', lambda *_: None)

    dso = load_model_from_dir(model_dir)
    #Open stream
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=CHUNK)

    on_load(dso, p, stream)

    audio_to_process = []
    cur_data = ''  # current chunk  of audio data
    rel = 16000/CHUNK
    slid_win = deque(maxlen=int(SILENCE_LIMIT * rel))
    #Prepend audio from 0.5 seconds before noise was detected
    prev_audio = deque(maxlen=int(PREV_AUDIO * rel))

    started = False
    n = num_phrases
    response = []

    on_ready()
    while (num_phrases == -1 or n > 0):
        cur_data = stream.read(CHUNK)
        slid_win.append(math.sqrt(abs(audioop.avg(cur_data, 4))))

        if(sum([x > THRESHOLD for x in slid_win]) > 0):
            if(not started):
                on_record_start()
                started = True
            audio_to_process.append(cur_data)
        elif (started is True):
            on_record_end(list(prev_audio) + audio_to_process)
            # Transcribe
            text_value = deepspeech_stt(dso, list(prev_audio) + audio_to_process)
            on_transcription(text_value)

            # Reset all
            started = False
            slid_win = deque(maxlen=int(SILENCE_LIMIT * rel))
            prev_audio = deque(maxlen=int(0.5 * rel))
            audio_to_process = []
            n -= 1
            on_ready()
        else:
            prev_audio.append(cur_data)

    stream.close()
    p.terminate()


def save_deepspeech_audio(data, p, filename):
    # writes data to WAV file
    data = ''.join(data)
    wf = wave.open(filename + '.wav', 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(data)
    wf.close()
    return filename + '.wav'


# dso = deepspeech object
def deepspeech_stt(dso, audio):
    # Run deepspeech on the chunk that just completed VAD
    audio = np.frombuffer(b''.join(audio), dtype=np.int16)
    output = stt(dso[0], audio, 16000)
    inference_time = output[1]
    return output[0]


def main_on_record_start():
    print("Recording started")

def main_on_record_end(audio):
    print("Recording complete")

def main_on_ready():
    print("Ready for input")

def main_on_transcription(text):
    print("You said: "+text)

if(__name__ == '__main__'):
    listen_for_speech(on_record_start = main_on_record_start,
        on_record_end = main_on_record_end,
        on_ready = main_on_ready,
        on_transcription=main_on_transcription)
