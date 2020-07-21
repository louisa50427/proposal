import os
import sys
import numpy as np
from cv2 import cv2 as cv
from darkmodel import predictbox
from PIL import Image
 

def read_path(path_name):
    #讀取路經中的檔案，若資料夾有資料夾則遞迴讀取    
    dir_items = os.listdir(path_name)
    dir_items.sort(key=lambda x: (x.split('-')[1]))
    dir_items.sort(key=lambda x: (x.split('-')[0]))
    images = []

        
    for dir_item in dir_items:
        full_path = os.path.abspath(os.path.join(path_name, dir_item))        
        if os.path.isdir(full_path):   
            read_path(full_path)
        else:   
            if dir_item.endswith('.png'):
                #讀取圖片資料
                image = cv.imread(full_path)                 
                images.append(image)                                              
                     
    return images
def load_dataset(path_name):
    print(path_name)
    label = []
    images = read_path(path_name)#read_path return 所有圖片的list及圖片full path names
    
    boxlist =  predictbox(images) 
    images = np.array(images)
    
    #標註數據
    labels = os.listdir(path_name)
    labels.sort(key=lambda x: (x.split('-')[1]))
    labels.sort(key=lambda x: (x.split('-')[0]))
    for a in labels: 
        temp = os.path.splitext(a)[0]
        label.append(temp.split("-")[3])
    print(len(label))
    
    return images, label, boxlist
    