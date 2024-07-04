# Autism Emotion Recognition Model using CNN and DiCNN in PyTorch

## Overview

This repository contains a deep learning model for recognizing emotions in individuals with autism, developed using Convolutional Neural Networks (CNN) and Dilated Convolutional Neural Networks (DiCNN) in PyTorch. The model aims to help in understanding and recognizing the emotional expressions of individuals with autism, aiding in better communication and support.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features

- Emotion recognition using images
- Developed with CNN and DiCNN architectures
- Implemented in PyTorch
- Supports custom datasets
- High accuracy and robust performance

## Requirements

- Python 3.7 or higher
- PyTorch 1.7.0 or higher
- NumPy
- OpenCV
- scikit-learn
- Matplotlib

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/autism-emotion-recognition.git
   cd autism-emotion-recognition
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The model can be trained on any dataset containing labeled images of emotions. For demonstration purposes, you can use publicly available datasets like FER-2013 or any other emotion recognition dataset.

1. Download and extract your dataset.
2. Organize the dataset into the following structure:

   ```plaintext
   dataset/
       train/
           happy/
               img1.jpg
               img2.jpg
               ...
           sad/
               img1.jpg
               img2.jpg
               ...
           ...
       test/
           happy/
               img1.jpg
               img2.jpg
               ...
           sad/
               img1.jpg
               img2.jpg
               ...
           ...
   ```

## Usage

### Training

1. Configure the training parameters in `config.py`.
2. Start training:

   ```bash
   python train.py
   ```

### Evaluation

1. Evaluate the model on the test dataset:

   ```bash
   python evaluate.py
   ```

### Inference

1. Use the trained model for inference on new images:

   ```bash
   python inference.py --image path/to/image.jpg
   ```

## Training

The training script `train.py` allows you to train the model from scratch or fine-tune a pre-trained model. You can configure the training parameters such as learning rate, batch size, and number of epochs in the `config.py` file.

## Evaluation

The evaluation script `evaluate.py` computes metrics such as accuracy, precision, recall, and F1-score on the test dataset. The results are printed to the console and saved to a log file.

## Results

The model achieves high accuracy in recognizing emotions in individuals with autism. Detailed results and performance metrics are available in the `results` directory.

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING](CONTRIBUTING.md) guidelines before submitting a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to reach out with any questions or suggestions. Your feedback is highly appreciated!
