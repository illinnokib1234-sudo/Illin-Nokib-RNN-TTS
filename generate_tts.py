"""
XTTS-v2 Voice Cloning and Text-to-Speech
Author: Illin Nokib

This script generates speech using Coqui XTTS-v2
with a user-provided voice reference.

The voice reference is NOT included in this repository.
"""

from pathlib import Path
import argparse

import torch
from TTS.api import TTS


MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


def load_model():
    """Load XTTS-v2 using GPU when available, otherwise CPU."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tts = TTS(MODEL_NAME).to(device)

    print("XTTS-v2 loaded successfully!")

    return tts


def generate_speech(tts, text, voice_reference, output_file):
    """Generate speech using the provided voice reference."""

    voice_reference = Path(voice_reference)
    output_file = Path(output_file)

    if not voice_reference.exists():
        raise FileNotFoundError(
            f"Voice reference not found: {voice_reference}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("Generating speech...")

    tts.tts_to_file(
        text=text,
        speaker_wav=str(voice_reference),
        language="en",
        file_path=str(output_file),
    )

    print(f"Audio generated successfully: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate voice-cloned speech using Coqui XTTS-v2."
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="Path to the voice reference WAV file."
    )

    parser.add_argument(
        "--text",
        required=True,
        help="Text to convert into speech."
    )

    parser.add_argument(
        "--output",
        default="output.wav",
        help="Output WAV file path."
    )

    args = parser.parse_args()

    tts = load_model()

    generate_speech(
        tts=tts,
        text=args.text,
        voice_reference=args.reference,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
