# Real-time-Attendance-Management-System-using-Deep-Learning-Techniques

This project is a facial recognition-based attendance management system developed using deep learning models (ResNet and VGG) and Flask for web interface. The system can recognize faces in real-time, log attendance, and generate attendance reports.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Installation](#installation)
- [Contributing](#contributing)
  
## Introduction

The Face Attendance System aims to automate the process of attendance management using facial recognition technology. It employs deep learning models, specifically ResNet and VGG, to accurately recognize faces in real-time. The system provides a user-friendly web interface developed using Flask, allowing users to easily interact with the system.

## Features

- Real-time face recognition for attendance management.
- Web interface for easy interaction.
- Automatic logging of attendance records.
- Generation of attendance reports in Excel format.

## Technologies Used

- **Deep Learning Models**:
  - [ResNet](https://arxiv.org/abs/1512.03385): A deep convolutional neural network architecture used for feature extraction in face recognition.
  - [VGG](https://arxiv.org/abs/1409.1556): Another convolutional neural network architecture utilized for face recognition tasks.

- **Web Development**:
  - [Flask](https://flask.palletsprojects.com/): A micro web framework for Python used to develop the web interface of the system.
  - HTML: HyperText Markup Language used for creating the structure of web pages.
  - CSS: Cascading Style Sheets used for styling the web pages.

- **Python Libraries**:
  - [NumPy](https://numpy.org/): A library for numerical computing used for array manipulation and mathematical operations.
  - [PIL](https://python-pillow.org/): Python Imaging Library used for image processing tasks.
  - [OpenCV](https://opencv.org/): Open Source Computer Vision Library used for computer vision tasks such as face detection.
  - [TensorFlow](https://www.tensorflow.org/): An open-source machine learning framework used for building and training deep learning models.
  - [scikit-learn](https://scikit-learn.org/): A library for machine learning tasks such as data preprocessing and model evaluation.
  - [matplotlib](https://matplotlib.org/): A plotting library used for data visualization.
  
- **Other Technologies**:
  - [MTCNN](https://github.com/ipazc/mtcnn): Multi-task Cascaded Convolutional Networks used for face detection.

## Project Structure

The project directory is organized as follows:

- `Face_Attendance.ipynb`: Jupyter Notebook containing the training code for the face recognition models.
- `app.py`: Server file implementing the Flask application for the web interface.
- `index.html`: HTML file providing the web interface for the attendance system.
- `requirements.txt`: File listing the required Python dependencies.
- `resnet_model.h5`: Trained ResNet model for face recognition.
- `vgg_model.h5`: Trained VGG model for face recognition.

## Usage

To use the Face Attendance System, follow these steps:

1. Install the required dependencies listed in `requirements.txt`.
2. Run the Flask application by executing `app.py`.
3. Access the web interface through a web browser.
4. Upload images or stream video to recognize faces and log attendance.

## Installation

To install and run the Face Attendance System locally, follow these steps:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/your_username/face-attendance-system.git
   ```

2. Navigate to the project directory:

   ```bash
   cd face-attendance-system
   ```

3. Install the required dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask application:

   ```bash
   python app.py
   ```

5. Access the web interface by opening a web browser and navigating to `http://localhost:5000`.

## Contributing

Contributions to the Face Attendance System are welcome! Feel free to open issues for feature requests, bug fixes, or general improvements. Pull requests are also appreciated.
