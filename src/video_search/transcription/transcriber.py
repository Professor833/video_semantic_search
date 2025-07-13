"""
Video transcription module using OpenAI Whisper.
Handles batch transcription of video files with proper error handling and memory management.
"""

import os
import gc
import sys
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any
import whisper
import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils import (
    Config,
    setup_logging,
    get_video_files,
    generate_video_id,
    format_timestamp,
    save_json,
    load_json,
    validate_video_file,
    format_file_size,
    get_file_size,
    ProgressTracker,
    estimate_processing_time,
)


class WhisperTranscriber:
    """OpenAI Whisper transcriber with memory management and error handling."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)
        self.model = None
        self.device = self._get_device()

        # Whisper configuration
        self.model_size = config.get("whisper.model_size", "base")
        self.language = config.get("whisper.language", "en")
        self.fp16 = config.get("whisper.fp16", False)

        # Transcription parameters
        self.temperature = config.get(
            "whisper.temperature", 0.0
        )  # Controls randomness; 0.0 = deterministic, higher = more diverse (range: 0.0–1.0, default: 0.0)
        self.beam_size = config.get(
            "whisper.beam_size", 10
        )  # Number of beams for beam search; higher = more accurate but slower (range: 1–10+, default: 5)
        self.best_of = config.get(
            "whisper.best_of", 5
        )  # Number of candidates to consider; higher = better results, more compute (range: 1–10+, default: 5)
        self.patience = config.get(
            "whisper.patience", 1.0
        )  # Beam search patience; higher = more thorough search (range: 1.0+, default: 1.0), in seconds
        self.length_penalty = config.get(
            "whisper.length_penalty", 1.0
        )  # Penalizes longer sequences; 1.0 = no penalty (range: 0.0–2.0, default: 1.0)
        self.suppress_tokens = config.get(
            "whisper.suppress_tokens", "-1"
        )  # Tokens to suppress in output; "-1" disables suppression (string, default: "-1")
        self.initial_prompt = config.get(
            "whisper.initial_prompt", "This video is about Python programming language."
        )  # Optional prompt to guide transcription (string or None, default: None)
        self.condition_on_previous_text = config.get(
            "whisper.condition_on_previous_text", True
        )  # Use previous text as context; True = more coherent (bool, default: True)
        self.compression_ratio_threshold = config.get(
            "whisper.compression_ratio_threshold", 2.4
        )  # Discards segments with high compression (possible junk); lower = stricter (range: 1.0–5.0, default: 2.4)
        self.logprob_threshold = config.get(
            "whisper.logprob_threshold", -1.0
        )  # Discards segments with low confidence; -1.0 disables (range: -1.0–0.0, default: -1.0)
        self.no_speech_threshold = config.get(
            "whisper.no_speech_threshold", 0.6
        )  # Threshold for detecting silence; lower = more likely to keep quiet segments (range: 0.0–1.0, default: 0.6)

        # Paths
        self.transcripts_dir = Path(
            config.get("paths.transcripts_dir", "./data/transcripts")
        )
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

    def _get_device(self) -> str:
        """Determine the best device for Whisper processing."""
        device_config = self.config.get("whisper.device", "auto")

        if device_config == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        else:
            return device_config

    def load_model(self) -> None:
        """Load Whisper model with proper error handling."""
        try:
            self.logger.info(
                f"Loading Whisper model '{self.model_size}' on device '{self.device}'"
            )

            # Clear any existing model from memory
            if self.model is not None:
                del self.model
                gc.collect()  # clear the cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Load model
            self.model = whisper.load_model(
                self.model_size, device=self.device, in_memory=True
            )

            self.logger.info(f"Successfully loaded Whisper model '{self.model_size}'")

        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe_video(
        self, video_path: str, force_retranscribe: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe a single video file.

        Args:
            video_path: Path to the video file
            force_retranscribe: If True, retranscribe even if transcript exists

        Returns:
            Dictionary containing transcription results
        """
        if not validate_video_file(video_path):
            raise ValueError(f"Invalid video file: {video_path}")

        video_id = generate_video_id(video_path)
        transcript_file = self.transcripts_dir / f"{video_id}.json"

        # Check if transcript already exists
        if transcript_file.exists() and not force_retranscribe:
            self.logger.info(f"Transcript already exists for {Path(video_path).name}")
            return load_json(str(transcript_file))

        try:
            self.logger.info(f"Transcribing video: {Path(video_path).name}")
            self.logger.info(
                f"File size: {format_file_size(get_file_size(video_path))}"
            )

            # Load model if not already loaded
            if self.model is None:
                self.load_model()

            # Prepare transcription options
            options = {
                "language": self.language,
                "temperature": self.temperature,
                "beam_size": self.beam_size,
                "best_of": self.best_of,
                "patience": self.patience,
                "length_penalty": self.length_penalty,
                "suppress_tokens": self.suppress_tokens,
                "initial_prompt": self.initial_prompt,
                "condition_on_previous_text": self.condition_on_previous_text,
                "compression_ratio_threshold": self.compression_ratio_threshold,
                "logprob_threshold": self.logprob_threshold,
                "no_speech_threshold": self.no_speech_threshold,
                "fp16": self.fp16,
                "verbose": False,
            }

            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}

            # Transcribe
            result = self.model.transcribe(video_path, **options)

            # Process results
            transcript_data = {
                "video_id": video_id,
                "video_path": video_path,
                "video_filename": Path(video_path).name,
                "model_size": self.model_size,
                "language": result.get("language", self.language),
                "duration": result.get("duration", 0),
                "text": result.get("text", ""),
                "segments": [],
            }

            # Process segments
            for segment in result.get("segments", []):
                segment_data = {
                    "id": segment.get("id"),
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": segment.get("text", "").strip(),
                    "tokens": segment.get("tokens", []),
                    "temperature": segment.get("temperature", 0),
                    "avg_logprob": segment.get("avg_logprob", 0),
                    "compression_ratio": segment.get("compression_ratio", 0),
                    "no_speech_prob": segment.get("no_speech_prob", 0),
                    "confidence": 1.0 - segment.get("no_speech_prob", 0),
                }
                transcript_data["segments"].append(segment_data)

            # Save transcript
            save_json(transcript_data, str(transcript_file))

            self.logger.info(f"Successfully transcribed {Path(video_path).name}")
            self.logger.info(
                f"Duration: {format_timestamp(transcript_data['duration'])}"
            )
            self.logger.info(f"Segments: {len(transcript_data['segments'])}")

            return transcript_data

        except Exception as e:
            self.logger.error(f"Failed to transcribe {Path(video_path).name}: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def transcribe_batch(
        self, video_files: List[str], force_retranscribe: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Transcribe multiple video files in batch.

        Args:
            video_files: List of video file paths
            force_retranscribe: If True, retranscribe even if transcripts exist

        Returns:
            List of transcription results
        """
        if not video_files:
            self.logger.warning("No video files provided for transcription")
            return []

        self.logger.info(f"Starting batch transcription of {len(video_files)} videos")

        # Estimate processing time
        estimated_minutes = estimate_processing_time(video_files, self.model_size)
        self.logger.info(f"Estimated processing time: {estimated_minutes:.1f} minutes")

        results = []
        progress = ProgressTracker(len(video_files), "Transcribing videos")

        for i, video_path in enumerate(video_files):
            try:
                # Memory management: clear cache periodically
                if i > 0 and i % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                result = self.transcribe_video(video_path, force_retranscribe)
                results.append(result)

                progress.update()

            except Exception as e:
                self.logger.error(f"Failed to transcribe {Path(video_path).name}: {e}")
                # Continue with other videos
                progress.update()
                continue

        successful_transcriptions = len([r for r in results if r is not None])
        self.logger.info(
            f"Batch transcription completed: {successful_transcriptions}/{len(video_files)} successful"
        )

        return results

    def get_existing_transcripts(self) -> List[str]:
        """Get list of existing transcript files."""
        return [str(f) for f in self.transcripts_dir.glob("*.json")]

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("Transcriber resources cleaned up")


def main():
    """Main function for running transcription as a standalone script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe videos using OpenAI Whisper"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default=None,
        help="Directory containing video files (overrides config)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        help="Whisper model size (overrides config)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force retranscription of existing files"
    )
    parser.add_argument(
        "--video", type=str, default=None, help="Transcribe a single video file"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = Config(args.config)
        config.ensure_directories()

        # Override config with command line arguments
        if args.videos_dir:
            config.config["paths"]["videos_dir"] = args.videos_dir
        if args.model_size:
            config.config["whisper"]["model_size"] = args.model_size

        # Initialize transcriber
        transcriber = WhisperTranscriber(config)

        if args.video:
            # Transcribe single video
            if not validate_video_file(args.video):
                print(f"Error: Invalid video file: {args.video}")
                return 1

            result = transcriber.transcribe_video(args.video, args.force)
            print(f"Transcription completed: {result['video_filename']}")
            print(f"Duration: {format_timestamp(result['duration'])}")
            print(f"Segments: {len(result['segments'])}")

        else:
            # Batch transcription
            videos_dir = config.get("paths.videos_dir", "./videos")
            video_files = get_video_files(videos_dir)

            if not video_files:
                print(f"No video files found in {videos_dir}")
                return 1

            print(f"Found {len(video_files)} video files")
            for vf in video_files:
                print(f"  - {Path(vf).name} ({format_file_size(get_file_size(vf))})")

            results = transcriber.transcribe_batch(video_files, args.force)

            print(f"\nTranscription completed:")
            print(f"  - Successful: {len(results)}/{len(video_files)}")
            print(f"  - Transcripts saved to: {transcriber.transcripts_dir}")

        # Cleanup
        transcriber.cleanup()

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
