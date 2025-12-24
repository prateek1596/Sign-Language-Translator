# Sign Language Translator (MediaPipe + Machine Learning)

A real-time sign language recognition system that converts hand gestures into text and speech using MediaPipe and a trained ML model.

## Features
- Real-time hand tracking
- A–Z alphabet recognition
- Prediction stabilization
- Word formation
- Text-to-speech output
- Custom dataset training

## Tech Stack
- Python
- OpenCV
- MediaPipe
- Scikit-Learn
- Pandas
- PyTTSx3

## How it works
1. MediaPipe extracts 21 hand landmarks
2. Landmarks are normalized
3. ML model predicts the letter
4. Stabilization logic prevents noise
5. Letters form words
6. Press Enter to speak

## Setup

```bash
git clone https://github.com/yourusername/sign-language-translator
cd sign-language-translator
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
