# 🎥 Video Semantic Search Tool

A powerful video semantic search system that allows you to search through video content using natural language queries. The system transcribes videos using OpenAI Whisper, generates embeddings with sentence-transformers, and enables fast similarity search using FAISS.

## 🚀 Features

- **Automatic Transcription**: Uses OpenAI Whisper for high-quality speech-to-text conversion
- **Semantic Search**: Powered by sentence-transformers for meaningful similarity matching
- **Fast Retrieval**: FAISS index for sub-second search performance
- **Multiple Interfaces**: CLI, Flask API, and Python library
- **Flexible Configuration**: YAML-based configuration system
- **Memory Management**: Efficient handling of large video datasets
- **Context Awareness**: Returns surrounding context for better understanding
- **Web Interface**: Beautiful, responsive web UI for easy searching

## 📋 Requirements

- Python 3.8+
- FFmpeg (for video processing)
- CUDA (optional, for GPU acceleration)

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd SuperBrynTask
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install FFmpeg** (if not already installed):
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## 🏗️ Project Structure

```
SuperBrynTask/
├── videos/                          # Video files directory
├── src/video_search/               # Main package
│   ├── transcription/              # Whisper transcription
│   ├── embedding/                  # Embedding generation & FAISS
│   ├── query/                      # Search interface
│   └── utils.py                    # Utilities and configuration
├── data/                           # Generated data
│   ├── transcripts/               # JSON transcripts
│   ├── embeddings/               # Numpy embeddings
│   └── index/                    # FAISS index files
├── logs/                          # Log files
├── config.yaml                    # Configuration file
├── requirements.txt               # Python dependencies
├── pipeline.py                    # Main pipeline script
└── README.md                      # This file
```

## ⚙️ Configuration

The system uses `config.yaml` for configuration. Key settings include:

```yaml
# Whisper settings
whisper:
  model_size: "base"  # tiny, base, small, medium, large
  language: "en"
  device: "auto"      # auto, cpu, cuda

# Embedding settings
sentence_transformer:
  model_name: "all-MiniLM-L6-v2"
  normalize_embeddings: true
  batch_size: 32

# Search settings
query:
  top_k: 5
  similarity_threshold: 0.3
  context_window: 10

# API settings
flask:
  host: "0.0.0.0"
  port: 5000
  debug: false
```

## 🚀 Quick Start

### 1. Run the Complete Pipeline

```bash
# Process all videos and build search index
python pipeline.py --all

# With custom configuration
python pipeline.py --all --config my_config.yaml

# Force regeneration of existing files
python pipeline.py --all --force

# Start web server after processing
python pipeline.py --all --serve
```

### 2. Step-by-Step Processing

```bash
# Step 1: Transcribe videos
python pipeline.py --transcribe

# Step 2: Generate embeddings and build index
python pipeline.py --embed

# Step 3: Test search functionality
python pipeline.py --test
```

### 3. Search Your Videos

**CLI Search:**
```bash
python -m src.video_search.query.searcher "how to install Python"
```

**Web Interface:**
```bash
python -m src.video_search.query.api
# Visit http://localhost:5000
```

**Python API:**
```python
from src.video_search.utils import Config
from src.video_search.query.searcher import VideoSearchEngine

config = Config('config.yaml')
engine = VideoSearchEngine(config)
results = engine.search("your query here")

for result in results:
    print(f"Video: {result['video_filename']}")
    print(f"Time: {result['start_timestamp']} - {result['end_timestamp']}")
    print(f"Text: {result['text']}")
    print(f"Score: {result['similarity_score']:.3f}")
    print("-" * 50)
```

## 🔍 Usage Examples

### CLI Examples

```bash
# Basic search
python -m src.video_search.query.searcher "functions and parameters"

# Search with custom parameters
python -m src.video_search.query.searcher "loops" --top-k 10 --threshold 0.2

# Search within specific video
python -m src.video_search.query.searcher "variables" --video "Python Tutorial for Beginners 3.mp4"

# List all videos
python -m src.video_search.query.searcher "" --list-videos

# Show statistics
python -m src.video_search.query.searcher "" --stats
```

### API Examples

**Search Request:**
```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how to use loops", "top_k": 5, "similarity_threshold": 0.3}'
```

**Response:**
```json
{
  "success": true,
  "query": "how to use loops",
  "results": [
    {
      "rank": 1,
      "similarity_score": 0.842,
      "video_filename": "Python Tutorial for Beginners 7- Loops and Iterations.mp4",
      "start_timestamp": "00:02:15",
      "end_timestamp": "00:02:45",
      "text": "So let's talk about loops. Loops allow us to iterate over sequences...",
      "context": {
        "before": "In this video we'll learn about...",
        "after": "There are two main types of loops..."
      }
    }
  ]
}
```

## 📊 Performance

The system is optimized for fast search performance:

- **Transcription**: ~1x real-time with base model (varies by hardware)
- **Embedding Generation**: ~1000 segments/minute on CPU
- **Search**: <100ms for typical queries
- **Memory**: ~2GB RAM for 10 hours of video content

### Model Performance Comparison

| Model | Speed | Accuracy | Memory |
|-------|-------|----------|---------|
| tiny  | 5x    | Good     | 1GB     |
| base  | 1x    | Better   | 2GB     |
| small | 0.5x  | Great    | 3GB     |
| medium| 0.25x | Excellent| 5GB     |
| large | 0.125x| Best     | 10GB    |

## 🛡️ Error Handling

The system includes comprehensive error handling:

- **Memory Management**: Automatic cleanup and garbage collection
- **Graceful Failures**: Continue processing other videos if one fails
- **Retry Logic**: Automatic retries for transient failures
- **Detailed Logging**: Comprehensive logs for debugging
- **Validation**: Input validation and sanity checks

## 🔧 Advanced Usage

### Custom Segmentation

```python
from src.video_search.embedding.embedder import TranscriptSegmenter

# Custom segmentation parameters
config.config['transcript']['segment_length'] = 45  # 45 seconds
config.config['transcript']['overlap'] = 10        # 10 seconds overlap

segmenter = TranscriptSegmenter(config)
```

### Custom Embedding Models

```python
# Use different sentence transformer model
config.config['sentence_transformer']['model_name'] = 'all-mpnet-base-v2'
```

### FAISS Index Types

```python
# Different index types for different use cases
config.config['faiss']['index_type'] = 'IndexIVFFlat'  # For large datasets
config.config['faiss']['index_type'] = 'IndexFlatL2'   # For L2 distance
```

## 🐛 Troubleshooting

### Common Issues

1. **FFmpeg not found**:
   ```bash
   # Install FFmpeg
   brew install ffmpeg  # macOS
   sudo apt install ffmpeg  # Ubuntu
   ```

2. **CUDA out of memory**:
   ```yaml
   # Use CPU instead
   whisper:
     device: "cpu"
   ```

3. **Slow transcription**:
   ```yaml
   # Use smaller model
   whisper:
     model_size: "tiny"
   ```

4. **No search results**:
   ```yaml
   # Lower similarity threshold
   query:
     similarity_threshold: 0.1
   ```

### Debug Mode

Enable debug logging:
```yaml
logging:
  level: "DEBUG"
```

## 📈 Optimization Tips

1. **GPU Acceleration**: Use CUDA for faster processing
2. **Model Selection**: Balance speed vs accuracy based on needs
3. **Batch Processing**: Process multiple videos simultaneously
4. **Index Optimization**: Use IVF index for large datasets
5. **Memory Management**: Monitor memory usage with large datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) for embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for similarity search
- [Flask](https://flask.palletsprojects.com/) for the web API

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in the `logs/` directory
3. Open an issue on GitHub
4. Check configuration settings

---

**Happy Searching! 🎉**