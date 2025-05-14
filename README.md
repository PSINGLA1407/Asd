# Emotion Detection Web Application

A web-based application that uses deep learning to detect emotions from facial images. Built with Python Flask and modern web technologies.

## Features

- Real-time emotion detection from uploaded images
- Modern, responsive UI using Tailwind CSS
- RESTful API endpoint for emotion prediction
- Support for multiple emotion classes

## Tech Stack

- Backend: Python Flask
- Frontend: HTML, Tailwind CSS, JavaScript
- Deep Learning: PyTorch
- API: RESTful endpoints

## Project Structure

```
project1/
├── templates/
│   └── index.html      # Main web interface
├── app.py             # Flask application
├── Train/             # Training dataset
├── Test/              # Test dataset
└── requirements.txt   # Python dependencies
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/PSINGLA1407/ASD.git
cd ASD
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Open the web interface
2. Upload an image containing a face
3. Click "Detect Emotion"
4. View the detected emotion result

## Model Information

The application uses a deep learning model trained on facial emotion datasets to classify emotions. The model is optimized for real-time inference and high accuracy.

## Contributing

Feel free to submit issues and enhancement requests!
