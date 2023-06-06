import cv2
import os
import numpy as np

path = "datasets/base_datasets/dataset_aft/"
destiny_path = "datasets/base_datasets/dataset_aft_sift/"
fonts = ["Antiqua","Fraktur","Textura"]

descr = []
label = []

for i, font in enumerate(fonts):
    image_path = os.path.join(path,font)
    image_destiny_path = os.path.join(destiny_path,font)
    arr = np.empty([20, 128,1])
    
    
    for f in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, f))
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray,None)
        
        if(des is not None):
            if(len(des) > 19):
                des = des[:20]
                descr.append(des)
                
                if(fonts[i] == 'Antiqua'):
                    label.append(1.0)
                if(fonts[i] == 'Fraktur'):
                    label.append(2.0)
                if(fonts[i] == 'Textura'):
                    label.append(3.0)


                
    

np.savez("dataset.npz", x=descr, y=label)
            
            
            
            
        