import cv2
import streamlit as st
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#The detect_faces() function captures frames from the webcam and detects faces in the frames.
def detect_faces():
    # Initialize the webcam ,It first initializes the webcam using cv2.VideoCapture().
    cap = cv2.VideoCapture(0)
    while True:
        # Read the frames from the webcam,It then reads frames from the webcam using cap.read(),
        ret, frame = cap.read()
        # Convert the frames to grayscale
        # converts them to grayscale using cv2.cvtColor(), and detects faces using the face_cascade.detectMultiScale() method.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces using the face cascade classifier
        #The scaleFactor and minNeighbors parameters of the detectMultiScale() method control
        # the sensitivity and accuracy of the face detection.
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        # Draw rectangles around the detected faces
        #Once faces are detected, the function draws rectangles around them using cv2.rectangle().
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Display the frames
        #The function then displays the frames with the detected faces using cv2.imshow().
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
        # Exit the loop when 'q' is pressed
        #The function exits the loop and releases the webcam and all windows when the user presses the 'q' key.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Press the button below to start detecting faces from your webcam")
    # Add a button to start detecting faces
    if st.button("Detect Faces"):
        # Call the detect_faces function
        detect_faces()
if __name__ == "__main__":
    app()