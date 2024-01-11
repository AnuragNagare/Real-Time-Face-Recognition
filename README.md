# Real-Time Face Recognition with Python



## Overview

This project implements a real-time face recognition system using Python, OpenCV, and the face_recognition library. It leverages computer vision techniques to detect and recognize faces in a live video feed from the default camera.

## Features

- **Face Detection:** Utilizes OpenCV to locate faces in each video frame.
- **Face Recognition:** Employs the face_recognition library to recognize faces based on precomputed encodings.
- **Model Selection:** Supports both "hog" (CPU) and "cnn" (GPU) models for recognition.
- **Command-Line Interface:** Allows users to specify recognition model preferences using command-line arguments.
- **Efficient Data Storage:** Serializes and deserializes face encodings using Python's pickle for improved performance.

## Getting Started

### Prerequisites

- Python 3.x
- OpenCV
- face_recognition library

### Installation

1. Clone the repository:
   git clone [repository_link]
   cd face-recognition-python

Install dependencies:
pip install -r requirements.txt

How to run:

1] python detector.py --train -m="hog"
2] python detector.py --validate
3] python detector.py "path to the image"


Or you can run 
1] python dynamic.py 
for real time face recognition




https://github.com/AnuragNagare/Real-Time-Face-Recognition/assets/85473989/c833df8b-6a88-4864-88fc-f144097a00ac


