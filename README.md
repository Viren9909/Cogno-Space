# 😄 CognoSpace - Real-Time Facial Expression Recognition

CognoSpace is a deep learning-based system that detects human emotions from real-time webcam input using facial expression analysis. Built using Convolutional Neural Networks (CNN) and trained on the FER-2013 dataset, the system classifies faces into seven emotional states: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

---

## 📌 Project Overview

- 🎯 **Goal:** Real-time recognition of facial expressions using CNN and live camera feed.
- 🤖 **Core Tech:** TensorFlow, Keras, OpenCV, Python
- 📊 **Dataset:** FER-2013 (35,887 grayscale images of facial expressions)
- 💡 **Key Outcome:** Real-time emotion recognition with ~80%+ accuracy

---

## 🧠 Features

- ✅ Live webcam-based face detection and emotion classification
- 🎯 Seven emotion categories supported
- 🧪 CNN model trained with data augmentation and dropout
- 📉 Accuracy metrics and confusion matrix for performance evaluation
- ⚡ Real-time results with low-latency predictions

---

## 🛠️ Tech Stack

| Category       | Technology            |
|----------------|------------------------|
| Language       | Python 3.11.8          |
| Deep Learning  | TensorFlow 2.18, Keras 3.8 |
| Image Processing | OpenCV 4.11.0        |
| Data Handling  | NumPy, Pandas          |
| Visualization  | Matplotlib, Seaborn    |
| IDE            | VS Code                |

---

## 🧪 Methodology

1. **Dataset Preprocessing**
   - Normalize grayscale 48x48 pixel images
   - Data augmentation (rotation, flipping, zoom)

2. **Model Architecture**
   - CNN with Batch Normalization and Dropout
   - Softmax output layer for 7-class emotion prediction

3. **Training & Optimization**
   - Categorical cross-entropy loss
   - Adam optimizer + ReduceLROnPlateau
   - Early stopping and checkpointing

4. **Deployment**
   - Real-time video capture with OpenCV
   - Face detection → ROI → Emotion prediction

---

## 💻 System Requirements

### Software
- Python 3.11+
- TensorFlow, Keras
- OpenCV
- Jupyter/VS Code

### Hardware
- Intel i5/Ryzen 7 or higher
- 8GB RAM or more
- NVIDIA GPU (optional but recommended)
- Webcam (480p or above)

---

## 🎯 Applications

- 💬 Emotion-aware AI Chatbots
- 🧠 Mental Health Monitoring
- 🛡️ Security & Surveillance
- 🧑‍🏫 Student Engagement Analysis
- 📈 Customer Experience in Marketing

---

## 📈 Performance Metrics

- ✔️ **Accuracy:** 75–85% (FER-2013)
- ✔️ **Precision, Recall, F1-Score**: Evaluated per class
- ✔️ **Inference Time:** < 50 ms/frame for smooth live feedback

---

## 🚀 Future Scope

- Deploy on mobile/web platforms
- Multi-face detection for group emotion analytics
- Integrate with AR/VR for immersive environments
- Use Vision Transformers (ViTs) for advanced accuracy
- Chatbot integration for adaptive emotional response

---

## 👨‍💻 Authors

- Virendra Rathva  
- Ved Thakar  
- Bhavik Chaudhary  
- Dhwanish Parmar  
*(Parul Institute of Technology, CSE Department)*

---

## 📚 References

- [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [OpenCV](https://opencv.org/)
- Goodfellow et al. (NIPS 2013), Simonyan & Zisserman (VGGNet), LeCun et al. (Deep Learning)

---

