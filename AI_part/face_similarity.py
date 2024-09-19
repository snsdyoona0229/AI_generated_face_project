# -*- coding: utf-8 -*-
import os
import face_recognition
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.image as img

def face_similarity_compare():
   # 已知人臉位置的臉部編碼
   known_image_path =  "temp/content_path.jpg"
   known_image      = face_recognition.load_image_file(str(known_image_path))
   known_image_encoding = face_recognition.face_encodings(known_image)[0]
   # 未知人臉位置的臉部編碼
   new_image_path = "temp/style_path.jpg"
   new_image      = face_recognition.load_image_file(str(new_image_path))
   new_image_encoding = face_recognition.face_encodings(new_image)[0]
   # 進行計算並顯示結果
   distance_img = face_recognition.face_distance([known_image_encoding], new_image_encoding)
   
   image01 = img.imread(known_image_path)  
   image02 = img.imread(new_image_path)  
   ######################## 
   plt.figure(figsize=(12,12))
   similarity ="Compare image similarity:"+ str(distance_img[0])
   plt.title(similarity)
   title = ['Self Image', 'Compare image similarity']
   plt.subplot(1, 2, 1)
   plt.title(title[0])
   plt.imshow(image01)
   plt.subplot(1, 2, 2)
   plt.title(title[1])
   plt.imshow(image02)
   plt.savefig('temp/face_similarity_compare.jpg')
