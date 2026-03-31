# Hand Gesture Recognition

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-yellow)

Real-time hand gesture recognition system that detects hand signs via webcam and converts them into grammatically correct sentences using deep learning and LLM.

## Features

- Real-time hand detection and gesture classification
- Deep learning-based model (Keras/TensorFlow)
- Automatic sentence generation from gesture sequences
- Web-based UI with Streamlit
- Optimized for smooth, lag-free performance

## Quick Start

### Installation

```bash
git clone https://github.com/AtharvaM25/Hand_Recognition.git
cd Hand_Recognition
pip install -r requirements.txt
```

### Setup

Create a `.env` file:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
```

Ensure Ollama is running:

```bash
ollama serve
```

### Usage

```bash
streamlit run ui.py
```

Then:

1. Click **Toggle Camera** to start
2. Show hand gestures to the camera
3. Click again to stop and generate a sentence

## Project Structure

```
Hand_Recognition/
├── test.py              # Core gesture recognition logic
├── ui.py                # Streamlit web interface
├── data.py              # Data collection script
├── keras_model.h5       # Pre-trained model
├── labels.txt           # Gesture classes
└── requirements.txt     # Dependencies
```

## How It Works

```
Camera → Hand Detection → Gesture Classification → LLM → Output Sentence
```

## Requirements

- Python 3.8+
- Webcam
- 2GB+ RAM

Key dependencies:

- OpenCV
- cvzone
- Keras/TensorFlow
- Streamlit
- Ollama (for LLM)

## Troubleshooting

**Camera Lag**: Reduce resolution or skip frames in `test.py`

```python
cap_ob.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_ob.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

**Model Not Found**: Ensure `keras_model.h5` and `labels.txt` are in the root directory

**Camera Permission Denied**: Grant camera access in system settings

## Configuration

Edit settings in `test.py`:

```python
offset = 25              # Padding around hand
imgSize = 300           # Hand image size
confidence_threshold = 0.7  # Minimum confidence
```

## License

MIT License - See LICENSE file for details

## Author

**Atharva M**

- GitHub: [@AtharvaM25](https://github.com/AtharvaM25)
- Email: atharva25dev@gmail.com

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

**Made with ❤️ using OpenCV, Keras, and Streamlit**
