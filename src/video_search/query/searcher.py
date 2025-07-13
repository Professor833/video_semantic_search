"""
Video semantic search engine with advanced relevance optimization.
Performs fast similarity search using FAISS index and returns relevant video segments.
"""

import gc
import re
from typing import List, Dict, Any, Optional
import torch
from collections import Counter

from src.video_search.utils import Config, setup_logging, format_timestamp, clean_text
from src.video_search.embedding.embedder import EmbeddingGenerator, FAISSIndexBuilder


class VideoSearchEngine:
    """Main search engine for video semantic search with advanced relevance optimization."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)

        # Search configuration
        self.top_k = config.get("query.top_k", 5)  # 5 results
        self.similarity_threshold = config.get(
            "query.similarity_threshold", 0.3
        )  # 0.3 is the default threshold
        self.context_window = config.get("query.context_window", 10)  # 10 seconds

        # Advanced search parameters
        self.rerank_enabled = config.get("query.rerank_enabled", True)
        self.query_expansion_enabled = config.get("query.query_expansion_enabled", True)
        self.semantic_weight = config.get(
            "query.semantic_weight", 0.7
        )  # Weight for semantic similarity
        self.lexical_weight = config.get(
            "query.lexical_weight", 0.3
        )  # Weight for lexical similarity

        # Components
        self.embedder = EmbeddingGenerator(config)
        self.indexer = FAISSIndexBuilder(config)

        # Data
        self.segments = []
        self.embeddings = None
        self.index = None

        # State
        self.is_loaded = False

        # Common programming terms for query expansion
        self.programming_synonyms = {
            "function": ["function", "method", "def", "procedure", "routine"],
            "variable": ["variable", "var", "identifier", "name"],
            "loop": ["loop", "iteration", "for", "while", "iterate"],
            "list": ["list", "array", "sequence", "collection"],
            "dictionary": ["dictionary", "dict", "map", "hash", "key-value"],
            "string": ["string", "text", "str", "character"],
            "number": ["number", "integer", "int", "float", "numeric"],
            "condition": ["condition", "if", "conditional", "boolean"],
            "import": ["import", "module", "library", "package"],
            "class": ["class", "object", "instance", "type"],
        }

    def load_index(self, force_reload: bool = False) -> None:
        """Load FAISS index, embeddings, and segment metadata."""
        if self.is_loaded and not force_reload:
            return

        try:
            self.logger.info("Loading search index and embeddings...")

            # Load embeddings and segments
            self.embeddings, self.segments = self.embedder.load_embeddings()

            # Load FAISS index
            self.index = self.indexer.load_index()

            # Load sentence transformer model for query encoding
            self.embedder.load_model()

            self.is_loaded = True
            self.logger.info(
                f"Successfully loaded search index with {len(self.segments)} segments"
            )

        except Exception as e:
            self.logger.error(f"Failed to load search index: {e}")
            raise

    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and related terms for better semantic coverage.
        Note: This is a simple implementation of query expansion.

        This step ensures that different phrasings of the same intent can still retrieve relevant video segments.

        Args:
            query: Original query string

        Returns:
            List of expanded query variations
        """
        if not self.query_expansion_enabled:
            return [query]

        expanded_queries = [query]
        query_lower = query.lower()

        # Add synonym-based expansions
        for term, synonyms in self.programming_synonyms.items():
            if term in query_lower:
                for synonym in synonyms:
                    if synonym != term:
                        expanded_query = re.sub(
                            r"\b" + re.escape(term) + r"\b",
                            synonym,
                            query_lower,
                            flags=re.IGNORECASE,
                        )
                        if expanded_query != query_lower:
                            expanded_queries.append(expanded_query)

        # Add conceptual expansions
        conceptual_expansions = {
            "how to": ["tutorial", "guide", "example", "demonstration"],
            "what is": ["definition", "explanation", "meaning", "concept"],
            "create": ["make", "build", "define", "implement"],
            "use": ["utilize", "apply", "work with", "employ"],
        }

        for phrase, expansions in conceptual_expansions.items():
            if phrase in query_lower:
                for expansion in expansions:
                    expanded_query = query_lower.replace(phrase, expansion)
                    if expanded_query != query_lower:
                        expanded_queries.append(expanded_query)

        return list(set(expanded_queries))  # Remove duplicates

    def _calculate_lexical_similarity(self, query: str, segment_text: str) -> float:
        """
        Calculate lexical similarity using word overlap and TF-IDF-like scoring.
        Note: This is a simple implementation of lexical similarity.
        This step ensures that the search results are relevant to the query.

        Args:
            query: Query string
            segment_text: Segment text

        Returns:
            Lexical similarity score (0.0 to 1.0)
        """
        query_words = set(clean_text(query).lower().split())
        segment_words = set(clean_text(segment_text).lower().split())

        if not query_words or not segment_words:
            return 0.0

        # Basic Jaccard similarity
        intersection = query_words & segment_words
        union = query_words | segment_words
        jaccard = len(intersection) / len(union) if union else 0.0

        # Boost for exact phrase matches
        phrase_boost = 0.0
        query_clean = clean_text(query).lower()
        segment_clean = clean_text(segment_text).lower()

        if len(query_clean) > 3 and query_clean in segment_clean:
            phrase_boost = 0.3

        # Word frequency boost (important words appearing multiple times)
        query_counter = Counter(clean_text(query).lower().split())
        segment_counter = Counter(clean_text(segment_text).lower().split())

        frequency_score = 0.0
        for word in intersection:
            # Boost words that appear multiple times in segment
            freq_factor = min(segment_counter[word], 3) / 3.0  # Cap at 3 occurrences
            frequency_score += freq_factor

        frequency_score = frequency_score / len(query_words) if query_words else 0.0

        return min(1.0, jaccard + phrase_boost + frequency_score * 0.2)

    def _rerank_results(
        self, results: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results using hybrid semantic + lexical scoring.

        Args:
            results: Initial search results
            query: Original query

        Returns:
            Re-ranked results
        """
        if not self.rerank_enabled or not results:
            return results

        reranked_results = []

        for result in results:
            # Get semantic similarity (already calculated)
            semantic_score = result["similarity_score"]

            # Calculate lexical similarity
            lexical_score = self._calculate_lexical_similarity(query, result["text"])

            # Hybrid scoring
            hybrid_score = (
                self.semantic_weight * semantic_score
                + self.lexical_weight * lexical_score
            )

            # Update result with new scores
            result["lexical_score"] = lexical_score
            result["hybrid_score"] = hybrid_score
            result["original_semantic_score"] = semantic_score
            result["similarity_score"] = hybrid_score  # Use hybrid as main score

            reranked_results.append(result)

        # Sort by hybrid score
        reranked_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # Update ranks
        for i, result in enumerate(reranked_results):
            result["rank"] = i + 1

        return reranked_results

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for videos matching the query with advanced relevance optimization.

        Args:
            query: Search query string
            top_k: Number of results to return (overrides config)
            similarity_threshold: Minimum similarity score (overrides config)

        Returns:
            List of search results with video info and timestamps
        """
        if not self.is_loaded:
            self.load_index()

        # Use provided parameters or defaults
        k = top_k if top_k is not None else self.top_k
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.similarity_threshold
        )

        try:
            # Clean and expand query
            clean_query = clean_text(query)
            if not clean_query:
                return []

            self.logger.info(f"Searching for: '{clean_query}'")

            # Expand query for better semantic coverage
            expanded_queries = self._expand_query(clean_query)
            self.logger.debug(f"Expanded queries: {expanded_queries}")

            # Search with multiple query variations
            all_results = {}  # Use dict to avoid duplicates

            for expanded_query in expanded_queries:
                # Generate query embedding
                query_embedding = self.embedder.model.encode(
                    [expanded_query],
                    normalize_embeddings=self.embedder.normalize_embeddings,
                    convert_to_numpy=True,
                )

                # Search in FAISS index with more candidates for better filtering
                search_k = min(k * 5, len(self.segments))  # Increased candidate pool
                distances, indices = self.indexer.search(query_embedding, search_k)

                # Process results for this query variation
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    # Skip invalid indices
                    if idx >= len(self.segments) or idx < 0:
                        continue

                    # Skip if already found with better score
                    if (
                        idx in all_results
                        and all_results[idx]["similarity_score"] >= distance
                    ):
                        continue

                    # Convert FAISS distance to similarity score
                    similarity_score = float(max(-1.0, min(1.0, distance)))

                    # Apply similarity threshold - use configured threshold
                    if similarity_score < threshold:
                        continue

                    segment = self.segments[idx]

                    # Enhanced relevance check with original query
                    query_words = set(clean_query.lower().split())
                    segment_words = set(clean_text(segment["text"]).lower().split())

                    # More nuanced relevance filtering
                    word_overlap = len(query_words & segment_words)
                    overlap_ratio = (
                        word_overlap / len(query_words) if query_words else 0.0
                    )

                    # Balanced filtering: require either good semantic similarity OR decent word overlap
                    # Allow high semantic similarity results even with low word overlap
                    if similarity_score < 0.65 and overlap_ratio < 0.2:
                        continue

                    # Store result (will be overwritten if better score found)
                    all_results[idx] = {
                        "similarity_score": similarity_score,
                        "video_id": segment["video_id"],
                        "video_path": segment["video_path"],
                        "video_filename": segment["video_filename"],
                        "start_time": segment["start_time"],
                        "end_time": segment["end_time"],
                        "text": segment["text"],
                        "formatted_time": format_timestamp(segment["start_time"]),
                        "duration": segment["end_time"] - segment["start_time"],
                        "word_overlap_ratio": overlap_ratio,
                        "expanded_query_used": expanded_query,
                    }

            # Convert to list and add context
            results = []
            for idx, result in all_results.items():
                # Add context information
                context = self._get_context(idx, self.segments[idx])
                result.update(context)
                results.append(result)

            # Re-rank results using hybrid scoring
            results = self._rerank_results(results, clean_query)

            # Apply final filtering and limit results
            final_results = []
            for result in results:
                if len(final_results) >= k:
                    break

                # Final threshold check on hybrid score
                if result["similarity_score"] >= threshold:
                    final_results.append(result)

            self.logger.info(f"Found {len(final_results)} relevant results")
            return final_results

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def _get_context(self, segment_idx: int, segment: Dict[str, Any]) -> Dict[str, Any]:
        """Get context around a segment for better understanding."""
        context = {"before": "", "after": "", "video_segments_count": 0}

        try:
            # Find segments from the same video
            video_id = segment["video_id"]
            video_segments = [
                (i, s) for i, s in enumerate(self.segments) if s["video_id"] == video_id
            ]

            context["video_segments_count"] = len(video_segments)

            # Sort by start time
            video_segments.sort(key=lambda x: x[1]["start_time"])

            # Find current segment position
            current_pos = None
            for pos, (idx, seg) in enumerate(video_segments):
                if idx == segment_idx:
                    current_pos = pos
                    break

            if current_pos is not None:
                # Get context before
                if current_pos > 0:
                    before_seg = video_segments[current_pos - 1][1]
                    context["before"] = before_seg["text"]

                # Get context after
                if current_pos < len(video_segments) - 1:
                    after_seg = video_segments[current_pos + 1][1]
                    context["after"] = after_seg["text"]

        except Exception as e:
            self.logger.warning(f"Failed to get context for segment: {e}")

        return context

    def search_by_video(
        self, video_filename: str, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search within a specific video."""
        if not self.is_loaded:
            self.load_index()

        # Filter segments by video
        video_segments = [
            (i, s)
            for i, s in enumerate(self.segments)
            if s["video_filename"] == video_filename
        ]

        if not video_segments:
            self.logger.warning(f"No segments found for video: {video_filename}")
            return []

        # Perform search on all segments first
        all_results = self.search(query, top_k=len(self.segments))

        # Filter results to only include the specified video
        video_results = [
            result
            for result in all_results
            if result["video_filename"] == video_filename
        ]

        # Limit results
        k = top_k if top_k is not None else self.top_k
        return video_results[:k]

    def get_video_list(self) -> List[Dict[str, Any]]:
        """Get list of all videos in the index."""
        if not self.is_loaded:
            self.load_index()

        videos = {}
        for segment in self.segments:
            video_id = segment["video_id"]
            if video_id not in videos:
                videos[video_id] = {
                    "video_id": video_id,
                    "video_path": segment["video_path"],
                    "video_filename": segment["video_filename"],
                    "segments_count": 0,
                    "total_duration": 0,
                }

            videos[video_id]["segments_count"] += 1
            videos[video_id]["total_duration"] = max(
                videos[video_id]["total_duration"], segment["end_time"]
            )

        # Convert to list and add formatted duration
        video_list = list(videos.values())
        for video in video_list:
            video["duration_formatted"] = format_timestamp(video["total_duration"])

        return sorted(video_list, key=lambda x: x["video_filename"])

    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        if not self.is_loaded:
            self.load_index()

        stats = {
            "total_segments": len(self.segments),
            "total_videos": len(set(s["video_id"] for s in self.segments)),
            "embedding_dimension": (
                self.embeddings.shape[1] if self.embeddings is not None else 0
            ),
            "index_type": self.config.get("faiss.index_type", "unknown"),
            "model_name": self.embedder.model_name,
            "is_loaded": self.is_loaded,
        }

        return stats

    def cleanup(self) -> None:
        """Clean up resources."""
        self.embedder.cleanup()

        if self.index is not None:
            del self.index
            self.index = None

        if self.embeddings is not None:
            del self.embeddings
            self.embeddings = None

        self.segments = []
        self.is_loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("Search engine resources cleaned up")


def main():
    """Main function for running search as a standalone CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Search videos using semantic similarity"
    )
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--top-k", type=int, default=None, help="Number of results to return"
    )
    parser.add_argument(
        "--video", type=str, default=None, help="Search within specific video file"
    )
    parser.add_argument(
        "--threshold", type=float, default=None, help="Minimum similarity threshold"
    )
    parser.add_argument(
        "--list-videos", action="store_true", help="List all videos in the index"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show search engine statistics"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = Config(args.config)

        # Initialize search engine
        search_engine = VideoSearchEngine(config)

        if args.list_videos:
            # List all videos
            videos = search_engine.get_video_list()
            print(f"\nFound {len(videos)} videos:")
            for video in videos:
                print(f"  - {video['video_filename']}")
                print(f"    Duration: {video['duration_formatted']}")
                print(f"    Segments: {video['segments_count']}")
                print()
            return 0

        if args.stats:
            # Show statistics
            stats = search_engine.get_stats()
            print(f"\nSearch Engine Statistics:")
            print(f"  - Total segments: {stats['total_segments']}")
            print(f"  - Total videos: {stats['total_videos']}")
            print(f"  - Embedding dimension: {stats['embedding_dimension']}")
            print(f"  - Index type: {stats['index_type']}")
            print(f"  - Model: {stats['model_name']}")
            print(f"  - Loaded: {stats['is_loaded']}")
            return 0

        # Perform search
        if args.video:
            results = search_engine.search_by_video(args.video, args.query, args.top_k)
        else:
            results = search_engine.search(args.query, args.top_k, args.threshold)

        # Display results
        if not results:
            print(f"No results found for query: '{args.query}'")
            return 0

        print(f"\nFound {len(results)} results for query: '{args.query}'")
        print("=" * 80)

        for result in results:
            print(
                f"\nRank {result['rank']} - Similarity: {result['similarity_score']:.3f}"
            )
            print(f"Video: {result['video_filename']}")
            print(f"Time: {result['formatted_time']}")
            print(f"Duration: {result['duration']:.1f}s")
            print(f"Text: {result['text']}")

            # Show context if available
            if result["before"]:
                print(f"Before: ...{result['before'][-100:]}")
            if result["after"]:
                print(f"After: {result['after'][:100]}...")

            print("-" * 80)

        # Cleanup
        search_engine.cleanup()

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
