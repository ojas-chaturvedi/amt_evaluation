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
import yaml


def extract_notes_with_offset(midi_file) -> list:
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file)

    # Extract instrument, onset, offset, and pitch
    note_events = []
    for instrument in midi_data.instruments:

        if instrument.is_drum == True:
            continue

        for note in instrument.notes:
            onset = note.start
            offset = note.end
            pitch = note.pitch
            note_events.append(
                (get_instrument_family(instrument.program), onset, offset, pitch)
            )
    # Sort by onset time
    note_events.sort(key=lambda x: x[1])

    return note_events


def get_instrument_family(program_number: int) -> str:
    # Load the instrument data
    with open("instrument_families.yaml", "r") as file:
        instrument_data = yaml.safe_load(file)

    for family, instruments in instrument_data["instrument_families"].items():
        if isinstance(instruments, dict) and program_number in instruments:
            return family

    return "Unknown"


def prepare_data_for_mir_eval(note_events) -> tuple:
    intervals = []
    families = []
    pitches = []
    for family, onset, offset, pitch in note_events:
        intervals.append([onset, offset])
        families.append(family)
        pitches.append(pitch)
    # Convert to NumPy arrays
    intervals = np.array(intervals)
    families = np.array(families)
    pitches = np.array(pitches)

    return intervals, families, pitches


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

    # Prepare intervals, instrument families, and pitches for mir_eval
    intervals, instrument_families, pitches = prepare_data_for_mir_eval(note_events)

    # Optionally, save the extracted notes to a file if an output path is provided
    if args.output:
        with open(args.output, "w") as file:
            for note in note_events:
                file.write(f"{note[0].name}, {note[1]:.4f}, {note[2]:.4f}, {note[3]}\n")
    else:
        # Print the results
        print(f"Intervals: {intervals}")
        print(f"Instrument Families: {instrument_families}")
        print(f"Pitches: {pitches}")


if __name__ == "__main__":
    main()
