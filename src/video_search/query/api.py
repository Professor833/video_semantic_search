"""
Flask API for video semantic search.
Provides web endpoints for searching videos and managing the search index.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import traceback

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.video_search.utils import Config, setup_logging
from src.video_search.query.searcher import VideoSearchEngine


# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Semantic Search</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .search-box { margin-bottom: 20px; }
        .search-box input { width: 70%; padding: 10px; font-size: 16px; border: 1px solid #ddd; border-radius: 4px; }
        .search-box button { padding: 10px 20px; font-size: 16px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin-left: 10px; }
        .search-box button:hover { background-color: #0056b3; }
        .filters { margin-bottom: 20px; }
        .filters select, .filters input { margin-right: 10px; padding: 5px; }
        .result { border: 1px solid #ddd; border-radius: 4px; padding: 15px; margin-bottom: 15px; background-color: #f9f9f9; }
        .result-header { font-weight: bold; color: #333; margin-bottom: 10px; }
        .result-meta { color: #666; font-size: 14px; margin-bottom: 10px; }
        .result-text { line-height: 1.5; }
        .context { margin-top: 10px; padding: 10px; background-color: #e9ecef; border-radius: 4px; font-size: 14px; }
        .loading { text-align: center; color: #666; }
        .error { color: #dc3545; padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px; }
        .stats { background-color: #e7f3ff; padding: 15px; border-radius: 4px; margin-bottom: 20px; }
        .video-list { margin-top: 20px; }
        .video-item { padding: 10px; border-bottom: 1px solid #eee; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎥 Video Semantic Search</h1>
            <p>Search through video transcripts using natural language queries</p>
        </div>

        <div class="search-box">
            <input type="text" id="queryInput" placeholder="Enter your search query..." />
            <button onclick="performSearch()">Search</button>
        </div>

        <div class="filters">
            <label>Results: </label>
            <select id="topK">
                <option value="5">5</option>
                <option value="10">10</option>
                <option value="20">20</option>
            </select>

            <label>Min Similarity: </label>
            <input type="number" id="threshold" min="0" max="1" step="0.1" value="0.3" />

            <button onclick="loadStats()">Show Stats</button>
            <button onclick="loadVideos()">List Videos</button>
        </div>

        <div id="results"></div>
    </div>

    <script>
        async function performSearch() {
            const query = document.getElementById('queryInput').value;
            const topK = document.getElementById('topK').value;
            const threshold = document.getElementById('threshold').value;

            if (!query.trim()) {
                alert('Please enter a search query');
                return;
            }

            document.getElementById('results').innerHTML = '<div class="loading">Searching...</div>';

            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, top_k: parseInt(topK), similarity_threshold: parseFloat(threshold) })
                });

                const data = await response.json();

                if (data.success) {
                    displayResults(data.results, data.query);
                } else {
                    document.getElementById('results').innerHTML = `<div class="error">Error: ${data.error}</div>`;
                }
            } catch (error) {
                document.getElementById('results').innerHTML = `<div class="error">Error: ${error.message}</div>`;
            }
        }

        function displayResults(results, query) {
            const resultsDiv = document.getElementById('results');

            if (results.length === 0) {
                resultsDiv.innerHTML = `<div class="error">No results found for query: "${query}"</div>`;
                return;
            }

            let html = `<h3>Found ${results.length} results for: "${query}"</h3>`;

            results.forEach(result => {
                html += `
                    <div class="result">
                        <div class="result-header">
                            Rank ${result.rank} - Similarity: ${result.similarity_score.toFixed(3)}
                        </div>
                        <div class="result-meta">
                            📹 ${result.video_filename}<br>
                            ⏰ ${result.start_timestamp} - ${result.end_timestamp} (${result.duration.toFixed(1)}s)
                        </div>
                        <div class="result-text">
                            ${result.text}
                        </div>
                        ${result.context.before || result.context.after ? `
                            <div class="context">
                                ${result.context.before ? `<strong>Before:</strong> ...${result.context.before.slice(-100)}<br>` : ''}
                                ${result.context.after ? `<strong>After:</strong> ${result.context.after.slice(0, 100)}...` : ''}
                            </div>
                        ` : ''}
                    </div>
                `;
            });

            resultsDiv.innerHTML = html;
        }

        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();

                if (data.success) {
                    const stats = data.stats;
                    document.getElementById('results').innerHTML = `
                        <div class="stats">
                            <h3>📊 Search Engine Statistics</h3>
                            <p><strong>Total Segments:</strong> ${stats.total_segments}</p>
                            <p><strong>Total Videos:</strong> ${stats.total_videos}</p>
                            <p><strong>Embedding Dimension:</strong> ${stats.embedding_dimension}</p>
                            <p><strong>Index Type:</strong> ${stats.index_type}</p>
                            <p><strong>Model:</strong> ${stats.model_name}</p>
                            <p><strong>Status:</strong> ${stats.is_loaded ? 'Loaded' : 'Not Loaded'}</p>
                        </div>
                    `;
                } else {
                    document.getElementById('results').innerHTML = `<div class="error">Error: ${data.error}</div>`;
                }
            } catch (error) {
                document.getElementById('results').innerHTML = `<div class="error">Error: ${error.message}</div>`;
            }
        }

        async function loadVideos() {
            try {
                const response = await fetch('/api/videos');
                const data = await response.json();

                if (data.success) {
                    const videos = data.videos;
                    let html = `<div class="video-list"><h3>📹 Available Videos (${videos.length})</h3>`;

                    videos.forEach(video => {
                        html += `
                            <div class="video-item">
                                <strong>${video.video_filename}</strong><br>
                                Duration: ${video.duration_formatted} | Segments: ${video.segments_count}
                            </div>
                        `;
                    });

                    html += '</div>';
                    document.getElementById('results').innerHTML = html;
                } else {
                    document.getElementById('results').innerHTML = `<div class="error">Error: ${data.error}</div>`;
                }
            } catch (error) {
                document.getElementById('results').innerHTML = `<div class="error">Error: ${error.message}</div>`;
            }
        }

        // Allow Enter key to trigger search
        document.getElementById('queryInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                performSearch();
            }
        });
    </script>
</body>
</html>
"""


class VideoSearchAPI:
    """Flask API wrapper for video search functionality."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)
        self.search_engine = VideoSearchEngine(config)

        # Flask configuration
        self.app = Flask(__name__)
        self.app.config["JSON_SORT_KEYS"] = False

        # Enable CORS
        CORS(self.app)

        # Setup routes
        self._setup_routes()

        # Load search index on startup
        try:
            self.search_engine.load_index()
            self.logger.info("Search index loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load search index: {e}")

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route("/")
        def index():
            """Serve the main web interface."""
            return render_template_string(HTML_TEMPLATE)

        @self.app.route("/api/search", methods=["POST"])
        def search():
            """Search for videos based on query."""
            try:
                data = request.get_json()

                if not data or "query" not in data:
                    return (
                        jsonify(
                            {"success": False, "error": "Query parameter is required"}
                        ),
                        400,
                    )

                query = data["query"]
                top_k = data.get("top_k", None)
                similarity_threshold = data.get("similarity_threshold", None)
                video_filename = data.get("video_filename", None)

                # Perform search
                if video_filename:
                    results = self.search_engine.search_by_video(
                        video_filename, query, top_k
                    )
                else:
                    results = self.search_engine.search(
                        query, top_k, similarity_threshold
                    )

                return jsonify(
                    {
                        "success": True,
                        "query": query,
                        "results": results,
                        "total_results": len(results),
                    }
                )

            except Exception as e:
                self.logger.error(f"Search API error: {e}")
                self.logger.error(traceback.format_exc())
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route("/api/videos", methods=["GET"])
        def get_videos():
            """Get list of all videos."""
            try:
                videos = self.search_engine.get_video_list()

                return jsonify(
                    {"success": True, "videos": videos, "total_videos": len(videos)}
                )

            except Exception as e:
                self.logger.error(f"Videos API error: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route("/api/stats", methods=["GET"])
        def get_stats():
            """Get search engine statistics."""
            try:
                stats = self.search_engine.get_stats()

                return jsonify({"success": True, "stats": stats})

            except Exception as e:
                self.logger.error(f"Stats API error: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route("/api/health", methods=["GET"])
        def health_check():
            """Health check endpoint."""
            return jsonify(
                {
                    "success": True,
                    "status": "healthy",
                    "index_loaded": self.search_engine.is_loaded,
                }
            )

        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({"success": False, "error": "Endpoint not found"}), 404

        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({"success": False, "error": "Internal server error"}), 500

    def run(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        debug: Optional[bool] = None,
    ):
        """Run the Flask application."""
        # Use config values or defaults
        host = host or self.config.get("flask.host", "0.0.0.0")
        port = port or self.config.get("flask.port", 5000)
        debug = debug if debug is not None else self.config.get("flask.debug", False)

        self.logger.info(f"Starting Flask API server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

    def cleanup(self):
        """Clean up resources."""
        self.search_engine.cleanup()


def main():
    """Main function for running the Flask API."""
    import argparse

    parser = argparse.ArgumentParser(description="Video search Flask API")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--host", type=str, default=None, help="Host to bind to (overrides config)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Port to bind to (overrides config)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    try:
        # Load configuration
        config = Config(args.config)

        # Create and run API
        api = VideoSearchAPI(config)
        api.run(host=args.host, port=args.port, debug=args.debug)

    except KeyboardInterrupt:
        print("\nShutting down...")
        api.cleanup()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
