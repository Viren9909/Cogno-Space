import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model("facial_recognition_model.h5")  


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


class_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]  


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w] 
        face = cv2.resize(face, (48, 48)) 

        
        face = np.stack((face,) * 3, axis=-1)  

        
        face = face.astype("float32") / 255.0  
        face = np.expand_dims(face, axis=0) 

        
        predictions = model.predict(face)[0]
        predicted_class = np.argmax(predictions)
        emotion_label = class_labels[predicted_class]

        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    
    cv2.imshow("Real-Time Emotion Detection", frame)

    
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
