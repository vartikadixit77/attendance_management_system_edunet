import cv2

# Initialize the webcam
video = cv2.VideoCapture(0)

# Load the Haar Cascade for face detection
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default (1).xml")

# Input ID for the dataset
id = input("Enter your id")
# id = int(id) # Uncomment if you need ID as an integer

count = 0

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture frame. Exiting.")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        count += 1
        # Save the captured face as a grayscale image
        cv2.imwrite(f'datasets/User.{str(id)}.{str(count)}.jpg', gray[y:y+h, x:x+w])
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)

    # Display the frame with rectangles
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' is pressed or 500 images are collected
    if cv2.waitKey(1) & 0xFF == ord('q') or count > 500:
        break

# Release the video capture and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
print("Dataset Collection Done.")
