### Taking a video as an input, detecting faces,
##  analysing facial expressions and giving emotions and emotion Matrix  along with the most frequently occured emotion


import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from collections import Counter


# Load pre-trained model
model = load_model("C:/Users/vidya Peddinti/Desktop/FLAME/FSP/CSIT431 Adv ML/Emotion-detection-main/Emotion-detection-main/best_model.h5")

# Load the Haar Cascade for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Specify the path to the input video file
video_path = "C:/Users/vidya Peddinti/Desktop/FLAME/FSP/CSIT431 Adv ML/Emotion-detection-main/Emotion-detection-main/V1.mp4"

# Open the video file for reading
cap = cv2.VideoCapture(video_path)

predicted_emotions_list = []

while True:
    ret, test_img = cap.read()  # Read the next frame from the video
    if not ret:
        break  # If no frame is read, exit the loop

    # Convert the frame to grayscale for face detection
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    frame_emotions = []  # Create a list to store emotions detected in this frame

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # Crop region of interest (face)
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)
        rounded_predictions = [round(val, 3) for val in predictions[0]]
        print(rounded_predictions)
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        print(predicted_emotion, '\n')

        frame_emotions.append(predicted_emotion)  # Add detected emotion to the frame's list

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    predicted_emotions_list.append(frame_emotions)  # Add the frame's emotions list to the main list

    cv2.imshow('Facial emotion analysis', test_img)

    if cv2.waitKey(10) == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()

# Now, predicted_emotions_list contains the detected emotions for each frame of the video.

#print(predicted_emotions_list)

# Analyze the predicted emotions list after processing the video
all_emotions = [emotion for frame_emotions in predicted_emotions_list for emotion in frame_emotions]

# Use Counter to count the occurrences of each emotion
emotion_counts = Counter(all_emotions)

# Find the most common emotion
most_common_emotion, most_common_count = emotion_counts.most_common(1)[0]

# Print the summary
print("Summary Emotion for the Entire Video:", most_common_emotion)
print("Occurrences:", most_common_count)
