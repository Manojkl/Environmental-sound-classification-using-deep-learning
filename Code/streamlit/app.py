import time, os
import logging
import streamlit as st
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from PIL import Image
# from settings import IMAGE_DIR, DURATION, WAVE_OUTPUT_FILE
from src.sound import sound
# from src.model import CNN
# from setup_logging import setup_logging 

def main():
    title = "Enviornmental sound classification using machine learning"
    st.title(title)
    image = Image.open('signal.png')
    st.image(image, use_column_width=True)

    if st.button('Record'):
        with st.spinner(f'Recording for {DURATION} seconds ....'):
            sound.record()
        st.success("Recording completed")

if __name__=='__main__':

    main()
