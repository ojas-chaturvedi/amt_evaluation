#!/opt/homebrew/bin/python3
"""
Name: instrument_similarity.py
Purpose: To compute the similarity matrix of musical instrument families based on audio features
"""

__author__ = "Ojas Chaturvedi"
__github__ = "github.com/ojas-chaturvedi"
__license__ = "MIT"

import tensorflow_datasets as tfds
import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import yaml
from tqdm import tqdm

# Map numeric instrument family IDs to human-readable names
INSTRUMENT_MAPPING = {
    0: "bass",
    1: "brass",
    2: "flute",
    3: "guitar",
    4: "keyboard",
    5: "mallet",
    6: "organ",
    7: "reed",
    8: "string",
    9: "synth_lead",
    10: "vocal",
}


# Step 1: Extract audio features
def extract_audio_features(audio_waveform, sample_rate):
    mfcc = librosa.feature.mfcc(
        y=audio_waveform, sr=sample_rate, n_mfcc=13
    )  # Extract MFCC features
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio_waveform, sr=sample_rate
    )  # Extract spectral centroid
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio_waveform, sr=sample_rate
    )  # Extract spectral bandwidth

    # Concatenate features and take the mean across time
    features = np.concatenate(
        (
            np.mean(mfcc, axis=1),
            np.mean(spectral_centroid, axis=1),
            np.mean(spectral_bandwidth, axis=1),
        )
    )
    return features


# Step 2: Load and combine all splits of NSynth dataset
def load_nsynth_with_features():
    print("Loading NSynth dataset from TensorFlow Datasets...")
    splits = ["train", "valid", "test"]
    instrument_features = {}

    for split in splits:
        print(f"Processing {split} split...")
        nsynth = tfds.load("nsynth", split=split)
        for sample in tqdm(tfds.as_numpy(nsynth), desc=f"Processing {split}"):
            # Extract family ID from the instrument dictionary
            instrument_id = sample["instrument"]["family"]
            instrument_name = INSTRUMENT_MAPPING[
                instrument_id
            ]  # Map to human-readable name

            # Extract audio and compute features
            audio = sample["audio"]
            sample_rate = 16000  # NSynth audio is sampled at 16 kHz
            features = extract_audio_features(audio, sample_rate)

            # Group features by instrument
            if instrument_name not in instrument_features:
                instrument_features[instrument_name] = []
            instrument_features[instrument_name].append(features)

    # Average features per instrument
    averaged_features = {
        instrument: np.mean(np.array(features), axis=0)
        for instrument, features in instrument_features.items()
    }

    return averaged_features


# Step 3: Generate similarity matrix
def generate_similarity_matrix(features):
    instruments = list(features.keys())
    feature_matrix = np.array([features[instr] for instr in instruments])

    # Compute cosine similarity
    similarity = cosine_similarity(feature_matrix)

    # Convert to DataFrame for readability
    similarity_df = pd.DataFrame(similarity, index=instruments, columns=instruments)
    return similarity_df


# Step 4: Save the similarity matrix to YAML
def save_similarity_to_yaml(similarity_df, output_file="instrument_similarity.yaml"):
    similarity_dict = similarity_df.to_dict()
    with open(output_file, "w") as file:
        yaml.dump(similarity_dict, file)
    print(f"Similarity matrix saved to {output_file}")


def main():
    # Load NSynth dataset and compute features
    features = load_nsynth_with_features()

    # Generate similarity matrix
    similarity_df = generate_similarity_matrix(features)

    # Save the matrix to a YAML file
    save_similarity_to_yaml(similarity_df)

    print("Instrument similarity matrix generated successfully.")


if __name__ == "__main__":
    main()
