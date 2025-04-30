import cv2
#from tensorflow.keras.models import model_from_json
import numpy as np

# Load the model from JSON file
#json_file = open("facialemotionmodel.json", "r")
#model_json = json_file.read()
#json_file.close()

#model = model_from_json(model_json)
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("facialemotionmodel.h5")


print("Model loaded successfully!")

# Load the Haar Cascade file for face detection
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess the image for prediction
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Adjust dimensions for the model
    return feature / 255.0  # Normalize pixel values

# Labels for emotions
labels = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}

# Initialize webcam
webcam = cv2.VideoCapture(0)
print("Press 'q' to quit the webcam feed.")

while True:
    ret, im = webcam.read()
    if not ret:
        print("Error accessing webcam. Exiting...")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (p, q, r, s) in faces:
        image = gray[q:q + s, p:p + r]  # Crop face region
        cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)  # Draw rectangle on face
        image = cv2.resize(image, (48, 48))  # Resize to match model input
        img = extract_features(image)  # Preprocess image
        pred = model.predict(img)  # Make prediction
        prediction_label = labels[pred.argmax()]  # Get predicted label

        # Display prediction on the video feed
        cv2.putText(im, prediction_label, (p, q - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Emotion Detection", im)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
