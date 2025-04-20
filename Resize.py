#Automating the resizing process

import cv2
import os

input = '/Users/adhishreeviti/Darpa/Real-Raw'
output ='/Users/adhishreeviti/Darpa/TrainingData/Real'
size = (512,512)


for fname in os.listdir(input):
    if fname.lower().endswith('.jpg'):
        img_path = os.path.join(input,fname)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to load image: {fname}")
            continue

        resized = cv2.resize(img,size)
        output_path = os.path.join(output,fname)
        cv2.imwrite(output_path, resized)

        
print("Done")