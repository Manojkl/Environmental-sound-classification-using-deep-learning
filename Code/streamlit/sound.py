import logging
import pyaudio, wave, pylab
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
# from pygame import mixer
from scipy.io.wavfile import write
# from settings import settings
from settings import DURATION, DEFAULT_SAMPLE_RATE, MAX_INPUT_CHANNELS, \
                                    WAVE_OUTPUT_FILE, INPUT_DEVICE, CHUNK_SIZE
# from settings.settings 
from setup_logging import setup_logging

setup_logging()
logger = logging.getLogger('src.sound')




