import os
from PIL import Image
import numpy as np
import cv2
import pickle as plk

print("[INFO] Initializing a CascadeClassifier")
classifier_path = "../../../data/haarcascades/haarcascade_frontalface_alt2.xml"
# init a CascadeClassifier instance
classifier = cv2.CascadeClassifier(classifier_path)
print("[INFO] Initialized a CascadeClassifier")
print("[INFO] Initializing an LBPH Face Recognizer")
# init a  LBPHFaceRecognizer
recognizer = cv2.face.LBPHFaceRecognizer_create() # FOR WINDOWS USERS
#recognizer = cv2.face.createLBPHFaceRecognizer()
print("[INFO] Initialized an LBPH Face Recognizer")

# base directory of the dataset(images)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))+"/../../../data/imgs"
print(BASE_DIR)
# faces folder with in the base dir
img_dir = os.path.join(BASE_DIR , "faces")

# an array to store images
imgs_arrays = []
# an array to store associated labels
imgs_labels = []
# a python set to store label ids
label_ids = {}
# a base id 
current_id = 0

print("[INFO] Entering Main Loop to scan for images")
# get all dirs/files in img_dir
for root, dirs, files in os.walk(img_dir):
    # foreach file found in img_dir
    for file in files:
        # check if the file is of type png or jpg
        if file.endswith("png") or file.endswith("jpg"):
            # get path of current image
            path = os.path.join(root, file)
            print("[INFO] Processing Image:", path)
            # get parent directory name of the current image 
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            # check if this label was found previously 
            if not label in label_ids:
                # add new label to labels list 
                label_ids[label] = current_id
                # increment current id to ensure a new id for next label
                current_id += 1
            # store current label in a temp var
            _id = label_ids[label]
            # load file using pillow lib then convert to gray scale
            pil_img = Image.open(path).convert("L")
            # cast file to a numpy array (for opencv)
            img_values2D = np.array(pil_img , "uint8")
            # detect faces in the current image 
            faces = classifier.detectMultiScale(img_values2D)
            # for every detected face add it to imgs_arrays and add the associated label to imgs_labels
            print("[INFO] Processing faces:", str(len(faces)))
            for(x,y,w,h) in faces:
                # define the region of interset
                roi = img_values2D[y:y+h, x:x+w]
                # add to images arrays
                imgs_arrays.append(roi)
                # add associated id 
                imgs_labels.append(_id)
print("[INFO] Writing labels data")
# save labels to face-labels.plk using the pickle lib
with open("../../../output/face-labels.plk", 'wb')as f:
    plk.dump(label_ids, f)
# train the recognizer using imgs_arrays and imgs_labels as numpy arrays
print("[INFO] Starting model training process")
recognizer.train(imgs_arrays, np.array(imgs_labels))
print("[INFO] Finished model training process")
# save the trained recognizer to be used later
print("[INFO] Saving final model data")
recognizer.save("../../../output/faces.yml")
print("[INFO] Successfully saved final model data")

