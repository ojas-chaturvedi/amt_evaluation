#!/opt/homebrew/bin/python3
"""
Name: compare_midi.py
Purpose: To compare 2 .mid files to score the transcribed output compared to the original sheet music
"""

__author__ = "Ojas Chaturvedi"
__github__ = "github.com/ojas-chaturvedi"
__license__ = "MIT"

import argparse
import transcription  # Improved version of mir_eval.transcription

from load_data import extract_notes_with_offset, prepare_data_for_mir_eval


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Compare two MIDI files for transcription accuracy."
    )
    parser.add_argument(
        "--reference", required=True, help="Path to the reference MIDI file"
    )
    parser.add_argument(
        "--transcription", required=True, help="Path to the transcribed MIDI file"
    )
    parser.add_argument("--output", required=False, help="Path to save the F1 score")
    args = parser.parse_args()

    # Extract and prepare notes from reference and transcribed MIDI files
    reference_notes = extract_notes_with_offset(args.reference)
    estimated_notes = extract_notes_with_offset(args.transcription)

    # Prepare intervals and pitches for both reference and estimated notes
    ref_intervals, ref_instrument_families, ref_pitches = prepare_data_for_mir_eval(
        reference_notes
    )
    est_intervals, est_instrument_families, est_pitches = prepare_data_for_mir_eval(
        estimated_notes
    )

    # Compare the transcriptions
    scores = transcription.evaluate(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        ref_instruments=ref_instrument_families,
        est_instruments=est_instrument_families,
    )

    # Optionally, save the F1 score to a file if an output path is provided
    if args.output:
        with open(args.output, "w") as file:
            for key, value in scores.items():
                file.write(f"{key}: {value}\n")
    else:
        # Print the results
        for key, value in scores.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
