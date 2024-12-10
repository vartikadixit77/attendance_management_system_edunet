import cv2
import numpy as np
from PIL import Image  # Ensure Pillow is installed
import os

# Initialize the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Define the path to the dataset
path = "datasets"

def getImageID(path):
    # Get all file paths in the dataset folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    
    faces = []
    ids = []

    for imagePath in imagePaths:
        try:
            # Load the image and convert it to grayscale
            faceImage = Image.open(imagePath).convert('L')  # Ensure this image exists and is accessible
            faceNP = np.array(faceImage, dtype=np.uint8)  # Ensure array is in the correct type
            Id = int(os.path.split(imagePath)[-1].split(".")[1])  # Extract ID from the filename
            
            faces.append(faceNP)
            ids.append(Id)
            
            # Show the image during training
            cv2.imshow("Training", faceNP)
            cv2.waitKey(1)
        except Exception as e:
            print(f"Error processing image {imagePath}: {e}")

    return ids, faces

# Collect IDs and face data
IDs, facedata = getImageID(path)

# Train the recognizer
recognizer.train(facedata, np.array(IDs))
recognizer.write("Trainer.yml")
cv2.destroyAllWindows()
print("Training Completed.")
