#!/opt/homebrew/bin/python3
"""
Name: load_data.py
Purpose: To extract notes and relevant information from a MIDI file
"""

__author__ = "Ojas Chaturvedi"
__github__ = "github.com/ojas-chaturvedi"
__license__ = "MIT"

import argparse
import pretty_midi
import numpy as np


def extract_notes_with_offset(midi_file) -> list:
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    # Extract instrument, onset, offset, and pitch
    note_events = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            onset = note.start
            offset = note.end
            pitch = note.pitch
            note_events.append((instrument, onset, offset, pitch))
    # Sort by onset time
    note_events.sort(key=lambda x: x[1])

    return note_events


def prepare_data_for_mir_eval(note_events) -> tuple:
    intervals = []
    instruments = []
    pitches = []
    for instrument, onset, offset, pitch in note_events:
        intervals.append([onset, offset])
        instruments.append(instrument)
        pitches.append(pitch)
    # Convert to NumPy arrays
    intervals = np.array(intervals)
    instruments = np.array(instruments)
    pitches = np.array(pitches)

    return intervals, instruments, pitches


def main() -> None:
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Extract notes and relevant information from a MIDI file."
    )
    parser.add_argument("--path", required=True, help="Path to the MIDI file")
    parser.add_argument(
        "--output", required=False, help="Path to save the extracted notes"
    )
    args = parser.parse_args()

    # Extract notes from the MIDI file
    note_events = extract_notes_with_offset(args.path)

    # Prepare intervals and pitches for mir_eval
    intervals, instruments, pitches = prepare_data_for_mir_eval(note_events)

    # Optionally, save the extracted notes to a file if an output path is provided
    if args.output:
        with open(args.output, "w") as file:
            for note in note_events:
                file.write(f"{note[0].name}, {note[1]:.4f}, {note[2]:.4f}, {note[3]}\n")
    else:
        # Print the results
        print(f"Intervals: {intervals}")
        print(f"Instruments: {instruments}")
        print(f"Pitches: {pitches}")


if __name__ == "__main__":
    main()
