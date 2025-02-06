import numpy as np
import tensorflow as tf
import pickle
import music21 as m21
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config  # Now Python should find config.py!
from model import load_vocab

def generate_music(seed, length=100):
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    note_to_int = load_vocab()
    int_to_note = {num: note for note, num in note_to_int.items()}
    
    input_seq = [note_to_int[n] for n in seed][-config.SEQUENCE_LENGTH+1:]
    generated = seed[:]
    
    for _ in range(length):
        padded_seq = pad_sequences([input_seq], maxlen=config.SEQUENCE_LENGTH-1, padding='pre')
        pred = model.predict(padded_seq)[0]
        next_note = int_to_note[np.argmax(pred)]
        generated.append(next_note)
        input_seq.append(np.argmax(pred))
        input_seq = input_seq[1:]
    
    return generated

def save_midi(notes, filename="generated_song.mid"):
    stream = m21.stream.Stream()
    for note_str in notes:
        if "." in note_str:
            chord_notes = [m21.note.Note(n) for n in note_str.split(".")]
            stream.append(m21.chord.Chord(chord_notes))
        else:
            stream.append(m21.note.Note(note_str))
    stream.write('midi', fp=f"./output/{filename}")

if __name__ == "__main__":
    user_seed = ["C4", "E4", "G4"]
    generated_notes = generate_music(user_seed)
    save_midi(generated_notes)
s