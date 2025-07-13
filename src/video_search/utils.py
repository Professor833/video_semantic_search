"""
Utilities module for video semantic search tool.
Contains configuration loading, logging setup, and common helper functions.
"""

import os
import json
import logging
import yaml
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import colorlog
from datetime import datetime, timedelta


class Config:
    """Configuration manager for the video search tool."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'whisper.model_size')."""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_paths(self) -> Dict[str, str]:
        """Get all configured paths."""
        return self.config.get("paths", {})

    def ensure_directories(self) -> None:
        """Ensure all configured directories exist."""
        paths = self.get_paths()
        for path in paths.values():
            Path(path).mkdir(parents=True, exist_ok=True)


def setup_logging(config: Config) -> logging.Logger:
    """Set up logging configuration."""
    log_level = getattr(logging, config.get("logging.level", "INFO"))
    log_format = config.get(
        "logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create logger
    logger = logging.getLogger("video_search")
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with colors
    if config.get("logging.console_logging", True):
        console_handler = colorlog.StreamHandler()
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if config.get("logging.file_logging", True):
        logs_dir = Path(config.get("paths.logs_dir", "./logs"))
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = (
            logs_dir / f"video_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_video_files(videos_dir: str) -> List[str]:
    """Get list of video files from the videos directory."""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    videos_path = Path(videos_dir)

    if not videos_path.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    video_files = []
    for file_path in videos_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(str(file_path))

    return sorted(video_files)


def generate_video_id(video_path: str) -> str:
    """Generate a unique ID for a video based on its path and filename."""
    return hashlib.md5(video_path.encode()).hexdigest()[:12]


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS timestamp."""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def parse_timestamp(timestamp: str) -> float:
    """Parse HH:MM:SS timestamp to seconds."""
    parts = timestamp.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = map(float, parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes, seconds = map(float, parts)
        return minutes * 60 + seconds
    else:
        return float(parts[0])


def save_json(data: Any, file_path: str) -> None:
    """Save data to JSON file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(file_path: str) -> Any:
    """Load data from JSON file."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    return Path(file_path).stat().st_size


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.1f}{size_names[i]}"


def validate_video_file(video_path: str) -> bool:
    """Validate if a file is a valid video file."""
    if not Path(video_path).exists():
        return False

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    return Path(video_path).suffix.lower() in video_extensions


def clean_text(text: str) -> str:
    """Clean and normalize text for better search results."""
    import re

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\-\'"()]', "", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def estimate_processing_time(video_files: List[str], model_size: str = "base") -> float:
    """Estimate processing time for transcription based on video files and model size."""
    # Rough estimates based on model size (minutes per hour of video)
    time_multipliers = {
        "tiny": 0.5,
        "base": 1.0,
        "small": 2.0,
        "medium": 4.0,
        "large": 8.0,
    }

    total_size = sum(get_file_size(vf) for vf in video_files)
    # Rough estimate: 1GB ≈ 1 hour of video
    estimated_hours = total_size / (1024**3)

    multiplier = time_multipliers.get(model_size, 1.0)
    estimated_minutes = estimated_hours * 60 * multiplier

    return estimated_minutes


class ProgressTracker:
    """Simple progress tracker for long-running operations."""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()

    def update(self, increment: int = 1) -> None:
        """Update progress."""
        self.current += increment
        self._print_progress()

    def _print_progress(self) -> None:
        """Print current progress."""
        percentage = (self.current / self.total) * 100
        elapsed = datetime.now() - self.start_time

        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = str(eta).split(".")[0]  # Remove microseconds
        else:
            eta_str = "Unknown"

        print(
            f"\r{self.description}: {self.current}/{self.total} ({percentage:.1f}%) - ETA: {eta_str}",
            end="",
        )

        if self.current >= self.total:
            print()  # New line when complete
