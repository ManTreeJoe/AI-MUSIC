import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config  # Now Python should find config.py!
from model import build_model, load_vocab


# Limit GPU memory allocation
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def prepare_sequences():
    with open("./data/processed_data.pkl", "rb") as f:
        data = pickle.load(f)

    note_to_int = load_vocab()
    sequences = []

    for song in data:
        all_notes = []  
        for instrument, notes in song.items():  # Extract only the notes, ignore instrument names
            all_notes.extend(notes)
        
        seq = [note_to_int[n] for n in all_notes if n in note_to_int]  # Ensure only known notes are used
        for i in range(0, len(seq) - config.SEQUENCE_LENGTH, 1):
            sequences.append(seq[i:i+config.SEQUENCE_LENGTH])

    sequences = pad_sequences(sequences, maxlen=config.SEQUENCE_LENGTH, padding='pre')
    X = sequences[:, :-1]
    y = sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=len(note_to_int))

    return X, y, len(note_to_int)
if __name__ == "__main__":
    X, y, vocab_size = prepare_sequences()
    model = build_model(vocab_size)
    model.fit(X, y, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
    model.save(config.MODEL_SAVE_PATH)
