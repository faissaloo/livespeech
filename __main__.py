from livespeech import *

def main_on_record_start():
    print("Recording started")

def main_on_record_end(audio, p):
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
