import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config  # Now Python should find config.py!

def build_model(vocab_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64),  # Reduce from 100
        LSTM(128, return_sequences=True),  # Reduce from 256
        Dropout(0.2),
        LSTM(128),  # Reduce from 256
        Dense(128, activation='relu'),  # Reduce from 256
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def load_vocab():
    with open("./data/processed_data.pkl", "rb") as f:
        dataset = pickle.load(f)

    # Extract all notes from every instrument in all songs
    all_notes = []
    for song in dataset:
        for instrument, notes in song.items():  # Extracting notes from dictionaries
            all_notes.extend(notes)

    note_to_int = {note: num for num, note in enumerate(sorted(set(all_notes)))}
    return note_to_int
