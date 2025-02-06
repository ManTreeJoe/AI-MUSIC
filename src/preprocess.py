import sys
import os
import pickle
import warnings
import music21 as m21

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config  # Now Python should find config.py!

# Suppress music21 warnings about missing instruments
warnings.simplefilter("ignore", m21.midi.translate.TranslateWarning)

def preprocess_midi(file_path):
    """Processes a MIDI file and extracts notes from all instrument parts."""
    try:
        midi = m21.converter.parse(file_path)
        notes_dict = {}

        for part in midi.parts:
            # Attempt to get the instrument name
            instrument_name = part.getInstrument().instrumentName or "Unknown_Instrument"

            notes = []
            for element in part.flatten().notes:  # Fixed incorrect attribute reference
                if isinstance(element, m21.note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, m21.chord.Chord):
                    notes.append('.'.join(str(n) for n in element.pitches))

            if notes:
                notes_dict[instrument_name] = notes

        return notes_dict if notes_dict else None  # Return None if no valid notes found

    except Exception as e:
        print(f"Skipping corrupt MIDI file: {file_path} - Error: {e}")
        return None

from tqdm import tqdm

def load_dataset():
    dataset = []
    files = [f for f in os.listdir(config.MIDI_DIR) if f.endswith(".mid")]
    for file in tqdm(files, desc='Processing MIDI files'):
        if file.endswith(".mid"):
            notes = preprocess_midi(os.path.join(config.MIDI_DIR, file))
            if notes:
                dataset.append(notes)

    with open("./data/processed_data.pkl", "wb") as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    load_dataset()
