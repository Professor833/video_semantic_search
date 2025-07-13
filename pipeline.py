#!/usr/bin/env python3
"""
Main pipeline script for video semantic search system.
Orchestrates the complete process from video transcription to search index building.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from video_search.utils import (
    Config,
    setup_logging,
    get_video_files,
    format_file_size,
    get_file_size,
)
from video_search.transcription.transcriber import WhisperTranscriber
from video_search.embedding.embedder import (
    TranscriptSegmenter,
    EmbeddingGenerator,
    FAISSIndexBuilder,
)
from video_search.query.searcher import VideoSearchEngine
from video_search.query.api import VideoSearchAPI


def run_transcription(config: Config, force: bool = False) -> bool:
    """Run video transcription step."""
    logger = setup_logging(config)
    logger.info("=" * 60)
    logger.info("STEP 1: VIDEO TRANSCRIPTION")
    logger.info("=" * 60)

    try:
        # Get video files
        videos_dir = config.get("paths.videos_dir", "./videos")
        video_files = get_video_files(videos_dir)

        if not video_files:
            logger.error(f"No video files found in {videos_dir}")
            return False

        logger.info(f"Found {len(video_files)} video files:")
        total_size = 0
        for vf in video_files:
            size = get_file_size(vf)
            total_size += size
            logger.info(f"  - {Path(vf).name} ({format_file_size(size)})")

        logger.info(f"Total size: {format_file_size(total_size)}")

        # Initialize transcriber
        transcriber = WhisperTranscriber(config)

        # Estimate processing time
        from video_search.utils import estimate_processing_time

        estimated_minutes = estimate_processing_time(
            video_files, transcriber.model_size
        )
        logger.info(f"Estimated processing time: {estimated_minutes:.1f} minutes")

        # Run transcription
        results = transcriber.transcribe_batch(video_files, force)

        successful = len([r for r in results if r is not None])
        logger.info(
            f"Transcription completed: {successful}/{len(video_files)} successful"
        )

        # Cleanup
        transcriber.cleanup()

        return successful > 0

    except Exception as e:
        logger.error(f"Transcription step failed: {e}")
        return False


def run_embedding(config: Config, force: bool = False) -> bool:
    """Run embedding generation and indexing step."""
    logger = setup_logging(config)
    logger.info("=" * 60)
    logger.info("STEP 2: EMBEDDING GENERATION & INDEXING")
    logger.info("=" * 60)

    try:
        # Get transcript files
        transcripts_dir = Path(
            config.get("paths.transcripts_dir", "./data/transcripts")
        )
        transcript_files = list(transcripts_dir.glob("*.json"))

        if not transcript_files:
            logger.error(f"No transcript files found in {transcripts_dir}")
            return False

        logger.info(f"Found {len(transcript_files)} transcript files")

        # Initialize components
        segmenter = TranscriptSegmenter(config)
        embedder = EmbeddingGenerator(config)
        indexer = FAISSIndexBuilder(config)

        # Check if embeddings already exist
        embeddings_file = embedder.embeddings_dir / "embeddings.npy"
        if embeddings_file.exists() and not force:
            logger.info("Embeddings already exist. Loading existing embeddings...")
            embeddings, segments = embedder.load_embeddings()
        else:
            # Segment transcripts
            logger.info("Segmenting transcripts...")
            segments = segmenter.segment_all_transcripts(
                [str(f) for f in transcript_files]
            )

            if not segments:
                logger.error("No segments generated from transcripts")
                return False

            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = embedder.generate_embeddings(segments)

            # Save embeddings
            embedder.save_embeddings(embeddings, segments)

        # Build FAISS index
        logger.info("Building FAISS index...")
        index = indexer.build_index(embeddings)

        # Save index
        indexer.save_index(index)

        logger.info(f"Embedding and indexing completed:")
        logger.info(f"  - Segments: {len(segments)}")
        logger.info(f"  - Embeddings shape: {embeddings.shape}")
        logger.info(f"  - Index vectors: {index.ntotal}")

        # Cleanup
        embedder.cleanup()

        return True

    except Exception as e:
        logger.error(f"Embedding step failed: {e}")
        return False


def test_search(config: Config, test_queries: Optional[List[str]] = None) -> bool:
    """Test the search functionality."""
    logger = setup_logging(config)
    logger.info("=" * 60)
    logger.info("STEP 3: TESTING SEARCH FUNCTIONALITY")
    logger.info("=" * 60)

    try:
        # Initialize search engine
        search_engine = VideoSearchEngine(config)

        # Test queries
        if test_queries is None:
            test_queries = [
                "how to install Python",
                "working with lists",
                "for loops and iterations",
                "functions and parameters",
                "string formatting",
            ]

        logger.info(f"Testing {len(test_queries)} queries...")

        for i, query in enumerate(test_queries, 1):
            logger.info(f"\nTest {i}: '{query}'")

            try:
                results = search_engine.search(query, top_k=3)

                if results:
                    logger.info(f"  Found {len(results)} results")
                    for j, result in enumerate(results, 1):
                        logger.info(
                            f"    {j}. {result['video_filename']} "
                            f"({result['start_timestamp']}-{result['end_timestamp']}) "
                            f"- Score: {result['similarity_score']:.3f}"
                        )
                else:
                    logger.warning(f"  No results found")

            except Exception as e:
                logger.error(f"  Search failed: {e}")

        # Get statistics
        stats = search_engine.get_stats()
        logger.info(f"\nSearch Engine Statistics:")
        logger.info(f"  - Total segments: {stats['total_segments']}")
        logger.info(f"  - Total videos: {stats['total_videos']}")
        logger.info(f"  - Embedding dimension: {stats['embedding_dimension']}")
        logger.info(f"  - Index type: {stats['index_type']}")
        logger.info(f"  - Model: {stats['model_name']}")

        # Cleanup
        search_engine.cleanup()

        return True

    except Exception as e:
        logger.error(f"Search test failed: {e}")
        return False


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(
        description="Video Semantic Search Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python pipeline.py --all

  # Run only transcription
  python pipeline.py --transcribe

  # Run only embedding generation
  python pipeline.py --embed

  # Run with custom config
  python pipeline.py --all --config my_config.yaml

  # Force regeneration
  python pipeline.py --all --force

  # Start API server after building
  python pipeline.py --all --serve
        """,
    )

    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run complete pipeline (transcribe + embed + test)",
    )
    parser.add_argument(
        "--transcribe", action="store_true", help="Run transcription step only"
    )
    parser.add_argument(
        "--embed", action="store_true", help="Run embedding generation step only"
    )
    parser.add_argument("--test", action="store_true", help="Run search tests only")
    parser.add_argument(
        "--serve", action="store_true", help="Start Flask API server after pipeline"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force regeneration of existing files"
    )
    parser.add_argument(
        "--test-queries",
        nargs="+",
        default=None,
        help="Custom test queries (space-separated)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not any([args.all, args.transcribe, args.embed, args.test, args.serve]):
        parser.error(
            "Must specify at least one action: --all, --transcribe, --embed, --test, or --serve"
        )

    try:
        # Load configuration
        config = Config(args.config)
        config.ensure_directories()

        logger = setup_logging(config)
        logger.info("🎥 Video Semantic Search Pipeline")
        logger.info("=" * 60)

        success = True

        # Run pipeline steps
        if args.all or args.transcribe:
            success &= run_transcription(config, args.force)

        if success and (args.all or args.embed):
            success &= run_embedding(config, args.force)

        if success and (args.all or args.test):
            success &= test_search(config, args.test_queries)

        if success:
            logger.info("=" * 60)
            logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)

            # Show usage instructions
            logger.info("\nUsage Instructions:")
            logger.info("1. CLI Search:")
            logger.info(
                "   python -m src.video_search.query.searcher 'your query here'"
            )
            logger.info("\n2. Flask API:")
            logger.info("   python -m src.video_search.query.api")
            logger.info("   Then visit: http://localhost:5000")
            logger.info("\n3. Direct Python usage:")
            logger.info(
                "   from src.video_search.query.searcher import VideoSearchEngine"
            )
            logger.info("   engine = VideoSearchEngine(config)")
            logger.info("   results = engine.search('your query')")

            # Start API server if requested
            if args.serve:
                logger.info("\n🚀 Starting Flask API server...")
                api = VideoSearchAPI(config)
                api.run()
        else:
            logger.error("=" * 60)
            logger.error("❌ PIPELINE FAILED")
            logger.error("=" * 60)
            return 1

        return 0

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        return 1

    except Exception as e:
        print(f"Pipeline error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
