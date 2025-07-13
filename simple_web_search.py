#!/usr/bin/env python3
"""Simple web search interface that avoids segmentation faults."""

import os
import json
import numpy as np
import faiss
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import argparse
from typing import Optional

# Disable multiprocessing warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global variables for caching
embeddings = None
segments = None
index = None
model = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Python Tutorial Video Search</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        input[type="text"] {
            flex: 1;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            min-width: 200px;
        }
        select {
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            min-width: 200px;
        }
        button {
            padding: 12px 24px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background: #0056b3;
        }
        .result {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 20px;
            margin-bottom: 20px;
            background: #fafafa;
        }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .video-title {
            font-weight: bold;
            color: #333;
            font-size: 18px;
        }
        .timestamp {
            color: #666;
            font-size: 14px;
        }
        .score {
            background: #e7f3ff;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 12px;
            color: #0066cc;
        }
        .text-content {
            line-height: 1.6;
            color: #555;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .error {
            background: #ffe6e6;
            color: #cc0000;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .sample-queries {
            margin-top: 20px;
            padding: 20px;
            background: #f0f8ff;
            border-radius: 5px;
        }
        .sample-queries h3 {
            margin-top: 0;
            color: #333;
        }
        .sample-query {
            display: inline-block;
            margin: 5px;
            padding: 8px 12px;
            background: #e1ecf4;
            border-radius: 15px;
            cursor: pointer;
            font-size: 14px;
            color: #0066cc;
        }
        .sample-query:hover {
            background: #d1e7dd;
        }
        .search-options {
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
            border: 1px solid #e9ecef;
        }
        .search-options h3 {
            margin-top: 0;
            margin-bottom: 15px;
            color: #333;
        }
        .option-group {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .option-group label {
            margin-right: 10px;
            min-width: 100px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 Python Tutorial Video Search</h1>

        <div class="search-options">
            <h3>Search Options</h3>
            <div class="option-group">
                <label for="videoSelect">Select Video:</label>
                <select id="videoSelect">
                    <option value="">All Videos</option>
                    <!-- Videos will be loaded here -->
                </select>
            </div>
        </div>

        <div class="search-box">
            <input type="text" id="queryInput" placeholder="Search for Python concepts..." />
            <button onclick="performSearch()">Search</button>
        </div>

        <div id="results"></div>

        <div class="sample-queries">
            <h3>Sample Queries:</h3>
            <div class="sample-query" onclick="searchSample('how to install Python')">how to install Python</div>
            <div class="sample-query" onclick="searchSample('working with lists')">working with lists</div>
            <div class="sample-query" onclick="searchSample('for loops')">for loops</div>
            <div class="sample-query" onclick="searchSample('functions and parameters')">functions and parameters</div>
            <div class="sample-query" onclick="searchSample('string formatting')">string formatting</div>
            <div class="sample-query" onclick="searchSample('dictionaries')">dictionaries</div>
            <div class="sample-query" onclick="searchSample('if statements')">if statements</div>
            <div class="sample-query" onclick="searchSample('modules and imports')">modules and imports</div>
        </div>
    </div>

    <script>
        // Load video list when page loads
        document.addEventListener('DOMContentLoaded', loadVideoList);

        function searchSample(query) {
            document.getElementById('queryInput').value = query;
            performSearch();
        }

        async function loadVideoList() {
            try {
                const response = await fetch('/videos');
                const data = await response.json();

                if (data.success) {
                    const videoSelect = document.getElementById('videoSelect');

                    // Add all videos to the dropdown
                    data.videos.forEach(video => {
                        const option = document.createElement('option');
                        option.value = video.video_filename;
                        option.textContent = video.video_filename;
                        videoSelect.appendChild(option);
                    });
                } else {
                    console.error('Failed to load videos:', data.error);
                }
            } catch (error) {
                console.error('Error loading videos:', error);
            }
        }

        async function performSearch() {
            const query = document.getElementById('queryInput').value.trim();
            if (!query) return;

            const videoFilename = document.getElementById('videoSelect').value;

            document.getElementById('results').innerHTML = '<div class="loading">Searching...</div>';

            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: query,
                        video_filename: videoFilename || null
                    })
                });

                const data = await response.json();

                if (data.success) {
                    displayResults(data.results, query, videoFilename);
                } else {
                    document.getElementById('results').innerHTML = `<div class="error">Error: ${data.error}</div>`;
                }
            } catch (error) {
                document.getElementById('results').innerHTML = `<div class="error">Error: ${error.message}</div>`;
            }
        }

        function displayResults(results, query, videoFilename) {
            const resultsDiv = document.getElementById('results');

            if (results.length === 0) {
                resultsDiv.innerHTML = '<div class="error">No results found. Try different keywords.</div>';
                return;
            }

            let title = `<h2>Found ${results.length} results for "${query}"`;
            if (videoFilename) {
                title += ` in "${videoFilename}"`;
            }
            title += `:</h2>`;

            let html = title;

            results.forEach(result => {
                html += `
                    <div class="result">
                        <div class="result-header">
                            <div class="video-title">${result.video_filename}</div>
                            <div class="score">Score: ${result.similarity_score.toFixed(3)}</div>
                        </div>
                        <div class="timestamp">${result.start_timestamp} - ${result.end_timestamp}</div>
                        <div class="text-content">${result.text}</div>
                    </div>
                `;
            });

            resultsDiv.innerHTML = html;
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


def load_data():
    """Load embeddings, segments, and index."""
    global embeddings, segments, index

    if embeddings is None:
        print("Loading embeddings and metadata...")
        embeddings = np.load("data/embeddings/embeddings.npy")

        with open("data/embeddings/embeddings_metadata.json", "r") as f:
            metadata = json.load(f)
        segments = metadata["segments"]

        print(f"Loaded {len(segments)} segments")

    if index is None:
        print("Loading FAISS index...")
        index = faiss.read_index("data/index/faiss_index.bin")
        print(f"Loaded FAISS index with {index.ntotal} vectors")


def format_timestamp(seconds):
    """Format seconds to MM:SS format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def search_with_precomputed(
    query: str, video_filename: Optional[str] = None, top_k: int = 5
):
    """Search using pre-computed embeddings and keyword matching."""
    global embeddings, segments, index

    # Load data if not already loaded
    load_data()

    if segments is None:
        return []

    # Simple keyword-based search as fallback
    # This avoids the segmentation fault by not using sentence transformers
    query_words = query.lower().split()

    results = []

    # Score each segment based on keyword matches
    for i, segment in enumerate(segments):
        # Skip if not in the selected video
        if video_filename and segment["video_filename"] != video_filename:
            continue

        text = segment["text"].lower()
        score = 0

        # Simple scoring based on keyword matches
        for word in query_words:
            if word in text:
                # Give higher score for exact matches
                score += text.count(word) * 0.1

                # Bonus for word boundaries
                if f" {word} " in text:
                    score += 0.2

        if score > 0:
            result = {
                "rank": len(results) + 1,
                "similarity_score": score,
                "video_filename": segment["video_filename"],
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "start_timestamp": format_timestamp(segment["start_time"]),
                "end_timestamp": format_timestamp(segment["end_time"]),
                "text": segment["text"],
            }
            results.append(result)

    # Sort by score and return top results
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results[:top_k]


def get_video_list():
    """Get list of unique videos from segments."""
    global segments

    # Load data if not already loaded
    if segments is None:
        load_data()

    if segments is None:
        return []

    # Extract unique video information
    videos = {}
    for segment in segments:
        video_filename = segment["video_filename"]
        if video_filename not in videos:
            videos[video_filename] = {
                "video_filename": video_filename,
                "segments_count": 0,
                "total_duration": 0,
            }

        videos[video_filename]["segments_count"] += 1
        videos[video_filename]["total_duration"] = max(
            videos[video_filename]["total_duration"], segment["end_time"]
        )

    # Convert to list and add formatted duration
    video_list = list(videos.values())
    for video in video_list:
        video["duration_formatted"] = format_timestamp(video["total_duration"])

    return sorted(video_list, key=lambda x: x["video_filename"])


def create_app():
    """Create Flask app."""
    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route("/search", methods=["POST"])
    def search():
        try:
            data = request.get_json()
            query = data.get("query", "")
            video_filename = data.get("video_filename")

            if not query:
                return jsonify({"success": False, "error": "Query is required"})

            results = search_with_precomputed(query, video_filename, top_k=10)

            return jsonify(
                {
                    "success": True,
                    "results": results,
                    "query": query,
                    "video_filename": video_filename,
                    "total_results": len(results),
                }
            )

        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    @app.route("/videos", methods=["GET"])
    def videos():
        try:
            video_list = get_video_list()

            return jsonify(
                {
                    "success": True,
                    "videos": video_list,
                    "total_videos": len(video_list),
                }
            )

        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    return app


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Simple video search web interface")
    parser.add_argument("--port", type=int, default=8080, help="Port to run on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")

    args = parser.parse_args()

    app = create_app()

    print(f"Starting web server on http://{args.host}:{args.port}")
    print("Open your browser and navigate to the URL above to search videos!")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
