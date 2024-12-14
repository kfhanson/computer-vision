import os
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Dataset Preparation : Load & Split dataset

# Nama orang yang mau kita pake
name_labels = os.listdir("dataset/train")

# Preprocessing & Face detection
def trainAndSave():
    face_list = []
    class_list = []
    
    train_path = "dataset/train"
    persons = os.listdir(train_path)
    
    for idx, name in enumerate(persons):
        full_path = train_path + '/' + name
        # full path to person's folder

        for image_name in os.listdir(full_path):
            img_full_path = full_path + '/' + image_name
            img = cv2.imread(img_full_path, 0)

            # Detect face using haarcascade
            detected_face = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
            if len(detected_face) < 1:
                continue
            
            for face_rect in detected_face:
                x, y, w, h = face_rect #exact coordinates of the face region
                face_img = img[y:y+h, x:x+w] #crop face region

                face_list.append(face_img)
                class_list.append(idx)
    
    # Training Model
    face_recognizer = cv2.face.LBPHFaceRecognizer_create() #bikin object
    face_recognizer.train(face_list, np.array(class_list))

    # Save model biar bisapake lagi tanpa training ulang
    face_recognizer.save('recognizer.yml')

def loadAndTest():
    # Load trained model
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        face_recognizer.read("recognizer.yml")
    except:
        print("Model not found, please train the model first")
        return

    test_path = "./dataset/test"

    for image_name in os.listdir(test_path):
        full_img_path = test_path + "/" + image_name

        img_gray = cv2.imread(full_img_path, 0) 
        img_bgr = cv2.imread(full_img_path)
        
        # Detect face using haarcascade
        detected_face = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
        if len(detected_face) < 1:
            continue
        
        for face_rect in detected_face:
            x, y, w, h = face_rect #exact coordinates of the face region
            face_img = img_gray[y:y+h, x:x+w] #crop face region

            # Predict face using trained recognizer
            res, confidence = face_recognizer.predict(face_img)
            # res : result detection || confidence : seberapa yakin face recognizer kita

            # Show result and accuracy
            # draw rectangle to detected face (color image)
            cv2.rectangle(img_bgr, (x,y), (x+w, y+h), (0, 0, 255), 1)
            # par1 : image || par2 : start coordinate || par3 : end coordinate || par4 : color || par5 : line thickness

            text = name_labels[res] + " : " + str(confidence) + "%"
            cv2.putText(img_bgr, text, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

            # Display result in a window
            cv2.imshow('Result', img_bgr)
            cv2.waitKey(0)

# Menu & Validation
if __name__ == "__main__":
    while True:
        print("1. Load Model & Test")
        print("2. Train & Save Model")
        print("3. Exit")

        choice = input(">> ")

        if choice == '1':
            loadAndTest()
        elif choice == "2":
            trainAndSave()
        elif choice == "3":
            break
        else:
            print("Invalid choice")

# trainAndSave()
# loadAndTest()