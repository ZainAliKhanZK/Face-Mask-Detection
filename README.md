# 😷 Face Mask Detection

A deep learning web app that checks whether a person in an image is wearing a face mask. Upload a photo or take a picture with your webcam or phone camera, and the model returns **Masked** or **No Mask** with a confidence score.

**🔗 Live Demo:** [seemask.streamlit.app](https://seeface.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![License](https://img.shields.io/badge/License-MIT-green)


## Features

- Upload an image (JPG, JPEG, PNG)
- Take a picture with a laptop webcam or phone camera
- Clear result card with a confidence bar
- Clean, responsive interface built with Streamlit

## Tech Stack

- Python
- TensorFlow / Keras (CNN model)
- Streamlit
- Pillow and NumPy

## Project Structure

```
.
├── app.py              # Streamlit app
├── mask_model.keras    # Trained model
├── requirements.txt    # Dependencies
└── README.md
```

## How to Run Locally

1. Clone the repository and open the folder:
   ```
   git clone <your-repo-link>
   cd <your-repo-folder>
   ```
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Start the app:
   ```
   streamlit run app.py
   ```
4. Open the link shown in the terminal (usually `http://localhost:8501`).

## How It Works

1. The image is converted to RGB and resized to 128 x 128 pixels.
2. The model outputs a number between 0 and 1.
3. A value below 0.5 means **Masked**, otherwise **No Mask**.

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `mask_model.keras`, `requirements.txt` and `README.md` to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose your repository, and set `app.py` as the main file.
4. Click **Deploy**. You get a public `https` link, which also allows the phone camera to work.

## Dataset

[Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) by Omkar Gurav on Kaggle.

## Author

**Zain Ali Khan**
GitHub: [ZainAliKhanZK](https://github.com/ZainAliKhanZK)
