"""
Embedding module for video semantic search.
Processes transcripts into segments, generates embeddings, and builds FAISS index.
"""

import gc
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from src.video_search.utils import (
    Config,
    setup_logging,
    load_json,
    save_json,
    clean_text,
    ProgressTracker,
    chunk_list,
)


class TranscriptSegmenter:
    """Segments transcripts into searchable chunks."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)

        # Segmentation parameters
        self.segment_length = config.get("transcript.segment_length", 30)  # seconds
        self.overlap = config.get("transcript.overlap", 5)  # seconds
        self.min_segment_length = config.get(
            "transcript.min_segment_length", 10
        )  # 10 seconds
        self.max_segment_length = config.get(
            "transcript.max_segment_length", 60
        )  # 60 seconds

    def segment_transcript(self, transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Segment a transcript into searchable chunks.

        Args:
            transcript: Transcript data from Whisper

        Returns:
            List of segment dictionaries
        """
        segments = []
        video_segments = transcript.get("segments", [])

        if not video_segments:
            self.logger.warning(
                f"No segments found in transcript for {transcript.get('video_filename', 'unknown')}"
            )
            return segments

        # Create time-based segments
        current_segment = {
            "video_id": transcript["video_id"],
            "video_path": transcript["video_path"],
            "video_filename": transcript["video_filename"],
            "start_time": 0,
            "end_time": 0,
            "text": "",
            "original_segments": [],
        }

        for whisper_segment in video_segments:
            segment_start = whisper_segment.get("start", 0)
            segment_end = whisper_segment.get("end", 0)
            segment_text = whisper_segment.get("text", "").strip()

            if not segment_text:
                continue

            # If this segment would make our current segment too long, finalize it
            if (
                current_segment["text"]
                and segment_end - current_segment["start_time"] > self.segment_length
            ):

                # Finalize current segment if it's long enough
                if (
                    current_segment["end_time"] - current_segment["start_time"]
                    >= self.min_segment_length
                    and current_segment["text"].strip()
                ):

                    current_segment["text"] = clean_text(current_segment["text"])
                    current_segment["duration"] = (
                        current_segment["end_time"] - current_segment["start_time"]
                    )
                    segments.append(current_segment.copy())

                # Start new segment with overlap
                overlap_start = max(0, current_segment["end_time"] - self.overlap)

                # Find segments that fall within the overlap period
                overlap_text = ""
                for orig_seg in current_segment["original_segments"]:
                    if orig_seg["end"] > overlap_start:
                        overlap_text += " " + orig_seg["text"]

                current_segment = {
                    "video_id": transcript["video_id"],
                    "video_path": transcript["video_path"],
                    "video_filename": transcript["video_filename"],
                    "start_time": overlap_start,
                    "end_time": segment_end,
                    "text": overlap_text.strip() + " " + segment_text,
                    "original_segments": [whisper_segment],
                }
            else:
                # Add to current segment
                if not current_segment["text"]:
                    current_segment["start_time"] = segment_start

                current_segment["end_time"] = segment_end
                current_segment["text"] += " " + segment_text
                current_segment["original_segments"].append(whisper_segment)

        # Add final segment
        if (
            current_segment["text"].strip()
            and current_segment["end_time"] - current_segment["start_time"]
            >= self.min_segment_length
        ):

            current_segment["text"] = clean_text(current_segment["text"])
            current_segment["duration"] = (
                current_segment["end_time"] - current_segment["start_time"]
            )
            segments.append(current_segment)

        self.logger.info(
            f"Segmented {transcript['video_filename']}: {len(segments)} segments"
        )
        return segments

    def segment_all_transcripts(
        self, transcript_files: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Segment all transcript files.

        Args:
            transcript_files: List of transcript file paths

        Returns:
            List of all segments from all transcripts
        """
        all_segments = []
        progress = ProgressTracker(len(transcript_files), "Segmenting transcripts")

        for transcript_file in transcript_files:
            try:
                transcript = load_json(transcript_file)
                segments = self.segment_transcript(transcript)
                all_segments.extend(segments)
                progress.update()

            except Exception as e:
                self.logger.error(f"Failed to segment {transcript_file}: {e}")
                progress.update()
                continue

        self.logger.info(f"Total segments created: {len(all_segments)}")
        return all_segments


class EmbeddingGenerator:
    """Generates embeddings for text segments using sentence-transformers."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)
        self.model = None

        # Model configuration
        self.model_name = config.get(
            "sentence_transformer.model_name", "all-MiniLM-L6-v2"
        )
        self.device = self._get_device()
        self.normalize_embeddings = config.get(
            "sentence_transformer.normalize_embeddings", True
        )
        self.batch_size = config.get("sentence_transformer.batch_size", 32)

        # Paths
        self.embeddings_dir = Path(
            config.get("paths.embeddings_dir", "./data/embeddings")
        )
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

    def _get_device(self) -> str:
        """Determine the best device for sentence transformer."""
        device_config = self.config.get("sentence_transformer.device", "auto")

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
        """Load sentence transformer model."""
        try:
            self.logger.info(
                f"Loading sentence transformer model '{self.model_name}' on device '{self.device}'"
            )

            # Clear any existing model from memory
            if self.model is not None:
                del self.model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Load model
            self.model = SentenceTransformer(self.model_name, device=self.device)

            self.logger.info(f"Successfully loaded sentence transformer model")

        except Exception as e:
            self.logger.error(f"Failed to load sentence transformer model: {e}")
            raise

    def generate_embeddings(self, segments: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate embeddings for text segments.

        Args:
            segments: List of segment dictionaries

        Returns:
            Numpy array of embeddings
        """
        if not segments:
            return np.array([])

        # Load model if not already loaded
        if self.model is None:
            self.load_model()

        # Extract texts
        texts = [segment["text"] for segment in segments]

        self.logger.info(f"Generating embeddings for {len(texts)} segments")

        try:
            # Generate embeddings in batches
            embeddings = []
            text_chunks = chunk_list(texts, self.batch_size)

            progress = ProgressTracker(len(text_chunks), "Generating embeddings")

            for chunk in text_chunks:
                # Generate embeddings for this chunk
                chunk_embeddings = self.model.encode(
                    chunk,
                    normalize_embeddings=self.normalize_embeddings,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                embeddings.append(chunk_embeddings)
                progress.update()

            # Concatenate all embeddings
            embeddings = np.vstack(embeddings)

            self.logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        segments: List[Dict[str, Any]],
        filename: str = "embeddings.npy",
    ) -> None:
        """Save embeddings and metadata."""
        embeddings_file = self.embeddings_dir / filename
        metadata_file = self.embeddings_dir / f"{Path(filename).stem}_metadata.json"

        # Save embeddings
        np.save(embeddings_file, embeddings)

        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "embedding_dimension": embeddings.shape[1],
            "num_segments": len(segments),
            "normalize_embeddings": self.normalize_embeddings,
            "segments": segments,
        }
        save_json(metadata, str(metadata_file))

        self.logger.info(f"Saved embeddings to {embeddings_file}")
        self.logger.info(f"Saved metadata to {metadata_file}")

    def load_embeddings(
        self, filename: str = "embeddings.npy"
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Load embeddings and metadata."""
        embeddings_file = self.embeddings_dir / filename
        metadata_file = self.embeddings_dir / f"{Path(filename).stem}_metadata.json"

        if not embeddings_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Embeddings or metadata file not found")

        embeddings = np.load(embeddings_file)
        metadata = load_json(str(metadata_file))

        return embeddings, metadata["segments"]

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("Embedding generator resources cleaned up")


class FAISSIndexBuilder:
    """Builds and manages FAISS index for fast similarity search."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)

        # FAISS configuration
        self.index_type = config.get("faiss.index_type", "IndexFlatIP")
        self.dimension = config.get("faiss.dimension", 384)
        self.nlist = config.get("faiss.nlist", 100)
        self.nprobe = config.get("faiss.nprobe", 10)

        # Paths
        self.index_dir = Path(config.get("paths.index_dir", "./data/index"))
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index = None

    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build FAISS (Facebook AI Similarity Search) index from embeddings.

        Args:
            embeddings: Numpy array of embeddings

        Returns:
            FAISS index
        """
        if embeddings.size == 0:
            raise ValueError("No embeddings provided")

        self.logger.info(f"Building FAISS index with {embeddings.shape[0]} vectors")
        self.logger.info(f"Embedding dimension: {embeddings.shape[1]}")
        self.logger.info(f"Index type: {self.index_type}")

        try:
            # Ensure embeddings are float32
            embeddings = embeddings.astype(np.float32)

            # Create index based on type
            if self.index_type == "IndexFlatIP":
                # Inner product (cosine similarity for normalized vectors)
                index = faiss.IndexFlatIP(embeddings.shape[1])

            elif self.index_type == "IndexFlatL2":
                # L2 distance (Exact search using Euclidean distance)
                index = faiss.IndexFlatL2(embeddings.shape[1])

            elif self.index_type == "IndexIVFFlat":
                # Inverted file index with flat quantizer (Approximate search using cosine similarity)
                quantizer = faiss.IndexFlatIP(embeddings.shape[1])
                index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], self.nlist)

                # Train the index
                self.logger.info("Training IVF index...")
                # Check if we have enough embeddings for training
                if embeddings.shape[0] < self.nlist:
                    raise ValueError(
                        f"Number of embeddings ({embeddings.shape[0]}) must be >= nlist ({self.nlist}) for IVF index training."
                    )
                index.train(embeddings)  # type: ignore
                index.nprobe = self.nprobe

            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")

            # Add vectors to index
            self.logger.info("Adding vectors to index...")
            index.add(embeddings)  # type: ignore

            self.logger.info(
                f"Successfully built FAISS index with {index.ntotal} vectors"
            )
            self.index = index

            return index

        except Exception as e:
            self.logger.error(f"Failed to build FAISS index: {e}")
            raise

    def save_index(self, index: faiss.Index, filename: str = "faiss_index.bin") -> None:
        """Save FAISS index to disk."""
        index_file = self.index_dir / filename

        try:
            faiss.write_index(index, str(index_file))
            self.logger.info(f"Saved FAISS index to {index_file}")

        except Exception as e:
            self.logger.error(f"Failed to save FAISS index: {e}")
            raise

    def load_index(self, filename: str = "faiss_index.bin") -> faiss.Index:
        """Load FAISS index from disk."""
        index_file = self.index_dir / filename

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_file}")

        try:
            index = faiss.read_index(str(index_file))
            self.logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
            self.index = index

            return index

        except Exception as e:
            self.logger.error(f"Failed to load FAISS index: {e}")
            raise

    def search(
        self, query_embeddings: np.ndarray, k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors in the index.

        Args:
            query_embeddings: Query embeddings
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None:
            raise ValueError("No index loaded")

        # Ensure query embeddings are float32
        query_embeddings = query_embeddings.astype(np.float32)

        # Search
        distances, indices = self.index.search(query_embeddings, k)  # type: ignore

        return distances, indices


def main():
    """Main function for running embedding generation as a standalone script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate embeddings and build FAISS index"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--transcripts-dir",
        type=str,
        default=None,
        help="Directory containing transcript files (overrides config)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of embeddings and index",
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = Config(args.config)
        config.ensure_directories()

        # Override config with command line arguments
        if args.transcripts_dir:
            config.config["paths"]["transcripts_dir"] = args.transcripts_dir

        # Get transcript files
        transcripts_dir = Path(
            config.get("paths.transcripts_dir", "./data/transcripts")
        )
        transcript_files = list(transcripts_dir.glob("*.json"))

        if not transcript_files:
            print(f"No transcript files found in {transcripts_dir}")
            return 1

        print(f"Found {len(transcript_files)} transcript files")

        # Initialize components
        segmenter = TranscriptSegmenter(config)
        embedder = EmbeddingGenerator(config)
        indexer = FAISSIndexBuilder(config)

        # Check if embeddings already exist
        embeddings_file = embedder.embeddings_dir / "embeddings.npy"
        if embeddings_file.exists() and not args.force:
            print("Embeddings already exist. Use --force to regenerate.")
            embeddings, segments = embedder.load_embeddings()
        else:
            # Segment transcripts
            print("Segmenting transcripts...")
            segments = segmenter.segment_all_transcripts(
                [str(f) for f in transcript_files]
            )

            if not segments:
                print("No segments generated from transcripts")
                return 1

            # Generate embeddings
            print("Generating embeddings...")
            embeddings = embedder.generate_embeddings(segments)

            # Save embeddings
            embedder.save_embeddings(embeddings, segments)

        # Build FAISS index
        print("Building FAISS index...")
        index = indexer.build_index(embeddings)

        # Save index
        indexer.save_index(index)

        print(f"\nEmbedding and indexing completed:")
        print(f"  - Segments: {len(segments)}")
        print(f"  - Embeddings shape: {embeddings.shape}")
        print(f"  - Index vectors: {index.ntotal}")
        print(f"  - Files saved to: {config.get('paths.index_dir')}")

        # Cleanup
        embedder.cleanup()

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
