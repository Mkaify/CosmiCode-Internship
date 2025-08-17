# Face Mask Detection Using Deep Learning

## Project Description

This project implements a **real-time face mask detection system** using **Python, OpenCV, and TensorFlow/Keras**. It classifies images of people into **two classes**: `with_mask` and `without_mask`. The system can also be integrated with a webcam for **real-time detection**, making it useful for public safety and automated monitoring applications.

**Key Features:**

- Detects faces and predicts mask usage.
- Works with images as well as webcam feed.
- Uses a Convolutional Neural Network (CNN) for binary classification.
- Data augmentation for better generalization.

---

## Dataset

- Dataset used: [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- Classes: `with_mask`, `without_mask`
- Total images: \~7,500
- All images are resized to 128x128 pixels.

---

## Project Structure

```
FaceMaskDetection/
│
├── dataset/               # Dataset folder with 'with_mask' and 'without_mask'
├── mask_detector_model.h5 # Trained CNN model
├── train_model.py         # Script to preprocess data, train & save model
├── predict_images.py      # Script to predict on test images
├── real_time_detection.py # Script for webcam real-time detection
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Setup Instructions

1. **Clone the repository**

```bash
git clone <your-repo-link>
cd FaceMaskDetection
```

2. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

Key libraries: `tensorflow`, `keras`, `opencv-python`, `numpy`, `matplotlib`, `scikit-learn`

3. **Place the dataset**

```
dataset/
│── with_mask/
│── without_mask/
```

4. **Train the model (optional)**

```bash
python train_model.py
```

This will preprocess the images, train the CNN, and save `mask_detector_model.h5`.

---

## Usage Guide

### Predict on Test Images

```bash
python predict_images.py
```

- Loads `mask_detector_model.h5` and predicts masks on test images.
- Displays sample predictions with true and predicted labels.

### Real-Time Webcam Detection

```bash
python real_time_detection.py
```

- Detects faces using OpenCV’s Haar Cascade or DNN face detector.
- Predicts mask usage on each detected face in real-time.
- Shows bounding boxes and labels on the webcam feed.

---

## Screenshots

### Sample Predictions:



### Real-Time Detection:



*(Replace with your actual screenshots)*

---

## Model Performance

- **Test Accuracy:** \~95% (update with your final results)
- Confusion Matrix and Classification Report available in `predict_images.py`.

---

## References

- [Kaggle – Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- TensorFlow & Keras Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)

---

## License

This project is licensed under the MIT License.

