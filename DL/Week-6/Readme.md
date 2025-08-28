# Week 6 — Image Captioning (Deep Learning)

This week focuses on building and running an image captioning system using deep learning with Keras/TensorFlow. The directory contains a trained model, tokenizer, pre-extracted image features, and a Jupyter notebook that walks through the workflow end-to-end.

## Contents

- ImageCaptioning.ipynb
  - A complete notebook covering dataset preparation, image feature extraction, tokenizer fitting, model architecture, training, and inference.
- model.keras
  - The trained image captioning model saved in Keras format.
- tokenizer.pkl
  - The fitted text tokenizer for mapping between words and integer tokens used by the model.
- features.pkl
  - Pre-extracted CNN image features (e.g., from a backbone like InceptionV3/VGG) for the training/validation images.

Note: Large binary files are included for convenience and may be heavy to download. If you plan to fork/extend, consider using Git LFS.

## What this project does

- Encodes images into feature vectors via a CNN backbone.
- Decodes feature vectors into natural language captions using an RNN/sequence model.
- Provides a trained model and tokenizer for quick inference, plus a notebook to retrain or explore the approach.

## Environment and requirements

- Python 3.8+
- Recommended packages:
  - tensorflow (or tensorflow-gpu if you have a compatible GPU)
  - keras
  - numpy
  - pillow
  - matplotlib
  - tqdm
  - scikit-learn (optional, for utilities)
  - jupyter (to run the notebook)
- Install basics:
  ```bash
  pip install --upgrade pip
  pip install tensorflow keras numpy pillow matplotlib tqdm scikit-learn jupyter
  ```

## Quick start: Run the notebook

1. Open ImageCaptioning.ipynb in Jupyter or upload it to Google Colab.
2. Ensure model.keras and tokenizer.pkl are available in the same directory (DL/Week-6/). If using Colab, upload them or mount your drive.
3. Follow the inference cells in the notebook to generate captions for sample images.

## Quick start: Minimal inference script

The snippet below shows how to:
- Load the trained model and tokenizer.
- Extract features from a new image using InceptionV3.
- Generate a caption with greedy decoding.

Important:
- MAX_LEN must match the sequence length used during training. If unsure, check the notebook or model definition. For many Flickr-like setups this is often around 34, but confirm in your notebook.

```python
import pickle
import numpy as np

from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.utils import load_img, img_to_array
from keras import Model
from keras.preprocessing.sequence import pad_sequences

# Paths (adjust as needed)
MODEL_PATH = "DL/Week-6/model.keras"
TOKENIZER_PATH = "DL/Week-6/tokenizer.pkl"
# Set this to the max sequence length used for training (verify in the notebook)
MAX_LEN = 34

# 1) Load model and tokenizer
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Build reverse index for token -> word
index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

# 2) Build CNN feature extractor (InceptionV3 without final classification layer)
base_cnn = InceptionV3(weights="imagenet")
feature_extractor = Model(inputs=base_cnn.input, outputs=base_cnn.layers[-2].output)

def extract_features(image_path):
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = feature_extractor.predict(image, verbose=0)
    return features  # shape (1, 2048) for InceptionV3 penultimate layer

def word_for_id(integer_id):
    return index_to_word.get(integer_id)

def generate_caption_greedy(photo_features):
    in_text = "startseq"
    for _ in range(MAX_LEN):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=MAX_LEN)
        yhat = model.predict([photo_features, seq], verbose=0)
        yhat = np.argmax(yhat)  # greedy
        word = word_for_id(yhat)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    # Clean up start/end tokens
    words = in_text.split()
    words = [w for w in words if w not in ("startseq", "endseq")]
    return " ".join(words)

# 3) Caption a new image
if __name__ == "__main__":
    test_image_path = "path/to/your/image.jpg"
    feat = extract_features(test_image_path)
    caption = generate_caption_greedy(feat)
    print("Caption:", caption)
```

Tips:
- If you used a different CNN backbone in training (e.g., VGG16), switch the feature extractor accordingly, and ensure feature dimensions match what the captioning model expects.
- If your model expects normalized feature shapes or a specific preprocessing pipeline, replicate those steps exactly.

## Using pre-extracted features (features.pkl)

features.pkl typically stores a mapping:
- key: an image identifier or filename
- value: a feature vector (e.g., 2048-d) extracted from the CNN

Example of loading:
```python
import pickle
with open("DL/Week-6/features.pkl", "rb") as f:
    features_dict = pickle.load(f)

# Example access (adjust key to your data)
some_image_id = list(features_dict.keys())[0]
photo_features = features_dict[some_image_id]  # shape should align with model input
```

Use these features for faster training/evaluation without recomputing CNN features every run. For captioning a brand-new image not in the training set, extract features on-the-fly as shown in the minimal inference script.

## Training workflow (as covered in the notebook)

High-level steps:
1. Collect dataset of images and captions (e.g., Flickr8k/30k/COCO).
2. Clean captions (lowercasing, tokenization, handling punctuation).
3. Fit tokenizer on the training captions; persist tokenizer.pkl.
4. Extract image features via CNN; persist features.pkl.
5. Build the captioning model (image feature input + text sequence input).
6. Train the model with teacher forcing; monitor validation loss/metrics.
7. Save the trained model as model.keras; evaluate with your chosen metrics (e.g., BLEU) and perform inference.

Refer to ImageCaptioning.ipynb for exact code and hyperparameters.

## Troubleshooting

- Model load issues:
  - Ensure your Keras/TensorFlow versions support loading the .keras model format. Upgrading keras and tensorflow usually resolves this:
    ```bash
    pip install --upgrade keras tensorflow
    ```
- Shape mismatch:
  - Confirm the CNN feature extractor architecture and preprocessing match those used during training.
  - Verify MAX_LEN and tokenizer consistency.
- Tokenizer inconsistencies:
  - Always pair the model with the exact tokenizer.pkl used in training.

## Acknowledgments

- Keras Applications for pretrained CNNs.
- Standard datasets (e.g., Flickr/COCO) commonly used for image captioning research and tutorials.
