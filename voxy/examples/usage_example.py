"""
Example usage of the voxy module for voice cloning and speech synthesis.
"""

import os
import io
from pathlib import Path
import torch
import torchaudio
import numpy as np

# Import the voxy module
from voxy import create_speech_model, audio_to_text, cleanup_audio


def example_1_basic_usage():
    """Basic usage of the module."""
    print("\n=== Example 1: Basic Usage ===")

    # Create a speech model
    model = create_speech_model(model_type="csm")

    # Generate speech with default voice
    audio = model.generate_speech(
        text="Hello, this is a test of the CSM speech model.",
        output_path="output/basic_output.wav",
    )

    print(f"Generated audio saved to output/basic_output.wav")
    print(f"Audio shape: {audio.shape}, Sample rate: {model._generator.sample_rate}")


def example_2_voice_cloning():
    """Voice cloning example."""
    print("\n=== Example 2: Voice Cloning ===")

    # Create a speech model
    model = create_speech_model(model_type="csm")

    # Assuming you have a sample audio file
    audio_path = "sample_voice.wav"

    # Option 1: Provide your own transcript
    transcript = "This is a sample of my voice for cloning purposes."

    # Clone the voice
    voice_profile = model.clone_voice(audio_input=audio_path, transcript=transcript)

    print(f"Voice profile created: {voice_profile}")

    # Generate speech with the cloned voice
    audio = model.generate_speech(
        text="This is my cloned voice speaking. Isn't it amazing?",
        voice_profile=voice_profile,
        output_path="output/cloned_voice.wav",
    )

    print(f"Generated cloned voice audio saved to output/cloned_voice.wav")


def example_3_auto_transcription():
    """Auto-transcription example."""
    print("\n=== Example 3: Auto Transcription ===")

    # Assuming you have a sample audio file
    audio_path = "sample_voice.wav"

    # Transcribe the audio
    transcript = audio_to_text(audio_path)
    print(f"Transcribed text: {transcript}")

    # Create a speech model
    model = create_speech_model(model_type="csm")

    # Clone the voice with auto-transcription
    voice_profile = model.clone_voice(
        audio_input=audio_path,
        # transcript=None  # This will use auto-transcription
    )

    # Generate speech with the cloned voice
    audio = model.generate_speech(
        text="This voice was cloned using automatic transcription.",
        voice_profile=voice_profile,
        output_path="output/auto_transcribed_voice.wav",
    )


def example_4_flexible_inputs():
    """Demonstrate flexible input handling."""
    print("\n=== Example 4: Flexible Inputs ===")

    # Create a speech model
    model = create_speech_model(model_type="csm")

    # Load audio in different ways
    with open("sample_voice.wav", "rb") as f:
        audio_bytes = f.read()

    # Clone voice from bytes
    voice_profile1 = model.clone_voice(
        audio_input=audio_bytes, transcript="This is a sample from bytes."
    )

    # Clone voice from file object
    with open("sample_voice.wav", "rb") as f:
        voice_profile2 = model.clone_voice(
            audio_input=f, transcript="This is a sample from a file object."
        )

    # Generate speech from different text formats
    # From string
    audio1 = model.generate_speech(
        text="This is direct text input.",
        voice_profile=voice_profile1,
        output_path="output/direct_text.wav",
    )

    # From file path (assuming text file exists)
    with open("sample_text.txt", "w") as f:
        f.write("This text was loaded from a file.")

    audio2 = model.generate_speech(
        text="/sample_text.txt",  # Starts with / so treated as file path
        voice_profile=voice_profile1,
        output_path="output/file_text.wav",
    )

    # From stream
    text_stream = io.StringIO("This text was loaded from a stream.")
    audio3 = model.generate_speech(
        text=text_stream,
        voice_profile=voice_profile1,
        output_path="output/stream_text.wav",
    )

    print("Generated speech from various input formats")


if __name__ == "__main__":
    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Note: To run these examples, you need:
    # 1. The CSM model and dependencies installed
    # 2. A sample voice file named "sample_voice.wav"

    # Uncomment examples to run them
    try:
        # example_1_basic_usage()
        # example_2_voice_cloning()
        # example_3_auto_transcription()
        # example_4_flexible_inputs()
        print(
            "To run examples, uncomment them in the script and ensure you have the necessary files."
        )
    except Exception as e:
        print(f"Error running examples: {e}")
