# Hand Gesture Recognition System

A real-time hand gesture recognition system that uses deep learning to detect hand gestures from a webcam feed and converts recognized gestures into grammatically correct sentences using an LLM.

## 🌟 Features

- **Real-time Hand Detection**: Uses MediaPipe for robust hand tracking
- **Gesture Classification**: Deep learning model for recognizing hand gestures
- **AI-Powered Text Generation**: Integrates with Ollama LLM to convert gesture sequences into proper English sentences
- **Data Collection Tool**: Script to easily collect training data for custom gestures
- **Visual Feedback**: Real-time visualization with confidence scores and detection highlights
- **Web Interface**: Streamlit-based web application for easy interaction

## 📋 Prerequisites

- Python 3.9 or higher
- Webcam/Camera connected to your system
- Ollama installed (for LLM integration) - [Download](https://ollama.ai)
- At least 4GB RAM recommended

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AtharvaM25/Hand_Recognition.git
cd Hand_Recognition
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

Using pip:

```bash
pip install -r requirements.txt
```

Or using poetry:

```bash
pip install poetry
poetry install
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root (if needed for Ollama configuration):

```
OLLAMA_BASE_URL=http://localhost:11434
```

## 🚀 Usage

### Running the Hand Gesture Recognition System

```bash
python test.py
```

**Controls:**

- Press `q` to quit the application
- Press `s` (in data collection mode) to capture an image of the current gesture

### Key Features in test.py:

- **Real-time Detection**: Displays hand bounding boxes with confidence scores
- **Gesture Recognition**: Classifies detected gestures using the trained model
- **Sentence Generation**: Collects recognized gestures and converts them into proper sentences using Ollama LLM
- **Flash Animation**: Visual feedback when a gesture is confidently recognized
- **Debug Output**: Comprehensive console logging for troubleshooting

### Collecting Training Data

Use `data.py` to collect images for training custom gesture models:

```bash
python data.py
```

**How to use:**

1. Position your hand in front of the camera
2. Press `s` to capture the current hand gesture image
3. Images are saved with timestamps for later training
4. Update the save path in the script to your preferred location

## 📁 Project Structure

```
Hand_Recognition/
├── data.py                 # Data collection script
├── test.py                 # Main gesture recognition application
├── keras_model.h5          # Pre-trained gesture classification model
├── labels.txt              # Gesture class labels
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project configuration
└── README.md              # This file
```

## 🔧 How It Works

### 1. **Hand Detection**

- Uses MediaPipe library to detect hands in video frames
- Extracts hand bounding box and region of interest

### 2. **Image Preprocessing**

- Crops detected hand region
- Resizes to 300x300 pixels
- Centers on white background to maintain aspect ratio

### 3. **Gesture Classification**

- Passes preprocessed image to Keras model
- Returns predicted gesture label with confidence score
- Filters results based on confidence threshold (0.7)

### 4. **Sentence Generation**

- Collects recognized gestures over a short time period
- Sends sequence to Ollama LLM (using Gemma 3.4B model)
- LLM generates grammatically correct sentences from gesture sequence

### 5. **Visual Feedback**

- Green box for confident predictions (>0.7 confidence)
- Red box for low confidence predictions
- Confidence score displayed in real-time
- Flash animation when gesture is confirmed

## ⚙️ Configuration

### Key Parameters (in test.py)

- **confidence_threshold**: Minimum confidence level for recognition (default: 0.7)
- **count_alphabet**: Frame count threshold for gesture confirmation (default: 15 frames)
- **offset**: Pixel offset for hand region extraction (default: 25)
- **imgSize**: Output image size for model (default: 300x300)
- **model**: LLM model used for sentence generation (default: "gemma3:4b")

### Customize Labels

Edit `labels.txt` to add or modify gesture class labels:

```
A
B
C
...
```

## 📊 Model Information

- **Framework**: TensorFlow/Keras
- **Output**: Class probability distribution for gestures
- **Training Data**: Hand gesture images collected with data.py

## 🔌 Dependencies

Key libraries used:

- **OpenCV**: Video capture and image processing
- **MediaPipe**: Hand detection and tracking
- **TensorFlow/Keras**: Gesture classification model
- **cvzone**: High-level computer vision tools
- **LangChain**: LLM integration and prompt management
- **Ollama**: Local LLM inference
- **NumPy**: Numerical computations

See `requirements.txt` for complete dependency list.

## 🐛 Troubleshooting

### Camera Not Opening

```
ERROR: Could not open camera!
```

- Ensure your webcam is connected and not in use by another application
- Try a different camera index (modify `cv2.VideoCapture(0)`)
- Check camera permissions on your system

### Model Not Found

```
ERROR: Make sure keras_model.h5 and labels.txt are in: [path]
```

- Verify that `keras_model.h5` and `labels.txt` are in the same directory as `test.py`
- Check the file paths and ensure they're correct

### Ollama Connection Issues

- Ensure Ollama is installed and running
- Verify Ollama service is accessible at `http://localhost:11434`
- Pull the Gemma model: `ollama pull gemma3:4b`

### Low Detection Confidence

- Ensure adequate lighting
- Position hand clearly within the frame
- Retrain the model with images from your environment
- Adjust `confidence_threshold` parameter

## 📈 Performance Tips

1. **Optimize Frame Rate**: Reduce video resolution for faster processing
2. **Model Quantization**: Use TensorFlow Lite for mobile deployment
3. **Batch Processing**: Process multiple frames to reduce latency
4. **GPU Acceleration**: Enable CUDA for faster inference (if available)
