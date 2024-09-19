#capture_facial_contours(2)
from __future__ import division
import cv2
import dlib
import numpy as np
import os
#capture_face(1)
import matplotlib.pyplot as plt
import cv2
import os
import dlib
from IPython.display import clear_output
from mtcnn import MTCNN
import cv2
import numpy as np
import sys

sys.path.append('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\')

#from Data_pre_processing import extract_faces,save_extract_face,reshape_for_polyline, prepare_training_data,Combine_pictures
import Data_pre_processing
import output_test

#capture_face(1)capture_face_and_contours
def extract_faces(source,destination,detector):
    counter = 0
    for dirname, _, filenames in os.walk(source):
        for filename in filenames:
            try:
                image = cv2.imread(os.path.join(dirname, filename))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                detections = detector.detect_faces(image)
                x, y, width, height = detections[0]['box']
                x1,y1,x2,y2 = x-10,y+10,x-10 +width + 20,y+10+height
                face = image[y1:y2, x1:x2]
                face = cv2.resize(face, (256, 256), interpolation=cv2.INTER_LINEAR)
                plt.imsave(os.path.join(destination,filename),face)
                clear_output(wait=True)
                print("Extraction progress: "+str(counter)+"\\"+str(len(filenames)-1))
            except:
                pass
            counter += 1
def save_extract_face():
    detector = MTCNN()
    extract_faces('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\','C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\extract_faces_MTCNN\\',detector)


#capture_facial_contours(2)--------------------------------------------------------------
DOWNSAMPLE_RATIO = 4 
photo_number = 400 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\shape_predictor_68_face_landmarks.dat')

def reshape_for_polyline(array):
    return np.array(array, np.int32).reshape((-1, 1, 2))

def prepare_training_data():
    DOWNSAMPLE_RATIO = 4 
    photo_number = 400 
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\shape_predictor_68_face_landmarks.dat')
    path2= 'C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\extract_faces_MTCNN\\'
    result =[f for f in os.listdir(path2) if os.path.isfile(os.path.join(path2,f))]
    count = 0
    for i in range(0,len(result),1) :
        frame = cv2.imread(path2 + result[i])
        frame_resize = cv2.resize(frame, (0,0), fx=1 / DOWNSAMPLE_RATIO, fy=1 / DOWNSAMPLE_RATIO)
        gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1) # 辨識人臉位置
        black_image = np.zeros(frame.shape, np.uint8) # 一張黑色圖片用於描繪人臉特徵
        if len(faces) == 1:
            for face in faces:
                detected_landmarks = predictor(gray, face).parts() # 提取人臉特徵
                landmarks = [[p.x * DOWNSAMPLE_RATIO, p.y * DOWNSAMPLE_RATIO] for p in detected_landmarks]

                jaw = reshape_for_polyline(landmarks[0:17])
                left_eyebrow = reshape_for_polyline(landmarks[22:27])
                right_eyebrow = reshape_for_polyline(landmarks[17:22])
                nose_bridge = reshape_for_polyline(landmarks[27:31])
                lower_nose = reshape_for_polyline(landmarks[30:35])
                left_eye = reshape_for_polyline(landmarks[42:48])
                right_eye = reshape_for_polyline(landmarks[36:42])
                outer_lip = reshape_for_polyline(landmarks[48:60])
                inner_lip = reshape_for_polyline(landmarks[60:68])

                color = (255, 255, 255) # 人脸特徵用於白色描繪
                thickness = 3 # 描繪線條粗細

                cv2.polylines(img=black_image, 
                              pts=[jaw,left_eyebrow, right_eyebrow, nose_bridge],
                              isClosed=False,
                              color=color,
                              thickness=thickness)
                cv2.polylines(img=black_image, 
                              pts=[lower_nose, left_eye, right_eye, outer_lip,inner_lip],
                              isClosed=True,
                              color=color,
                              thickness=thickness)
                              
                          # 保存圖片
            path_address = "C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\extract_contours_MTCNN\\"+result[i]
            cv2.imwrite(path_address,black_image)
            count += 1

#Combine_pictures(3)--------------------------------------------------------------

def Combine_pictures():
  #path1 ='C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\media\\'
  path2 = 'C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\extract_faces_MTCNN\\'
  path3= 'C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\extract_contours_MTCNN\\'
  #media =os.listdir(path1)
  result =[f for f in os.listdir(path2) if os.path.isfile(os.path.join(path2,f))]
  result2 =[d for d in os.listdir(path3) if os.path.isfile(os.path.join(path3,d))]

  intersection_set = list(set(result2).intersection(set(result)))

  for i in range(0,len(intersection_set),1) :
    img1 = cv2.imread(path2 + intersection_set[i])
    img2 = cv2.imread(path3 + intersection_set[i])
    vis = np.concatenate((img1, img2), axis=1)
    cv2.imwrite('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\Combine_pictures\\image_temp.jpg',vis)