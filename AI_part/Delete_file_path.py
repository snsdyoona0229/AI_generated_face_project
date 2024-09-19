import sys
import os
import shutil
import pathlib

def apple():
    photo = models.PhotoModel.objects.values_list('purl')
    x = models.PhotoModel.objects.all()
    count = 0
    for i in photo[:5]:
        print(i)
        #刪除相片檔
        x[count].delete()
        count += 1
        os.remove(os.path.join('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\media\\'+i[0] ))
        
def ball(file,class_ai,style_transfer):
#    f = open("C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp.txt", "r")
#    R = f.read()

    if class_ai =="face_similarity":
       temp_face_x = open("C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp_x.txt", "r")
       m01 = temp_face_x.read()
       
       temp_face_y = open("C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp_y.txt", "r")
       m02 = temp_face_y.read()
                    
       shutil.copy('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\media\\'+m01,os.path.join('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\content_path.jpg'))
       
       shutil.copy('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\media\\'+m02,os.path.join('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\style_path.jpg'))
              
    if class_ai =="style_transfer":
       shutil.copy('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\media\\'+m02,os.path.join('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\'+style_transfer))

    if class_ai !="style_transfer":
       shutil.copy('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\media\\'+file,os.path.join('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\temp.jpg'))
    
def remove_file(select_model_item):
    if select_model_item =='pix2pix':
      file_num = len(os.listdir('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\media\\'))
      shutil.move('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\pix2pix.jpg','C:\\Users\\user\\Desktop\django\\AI_generated_face_project_02\\album\\media\\'+str(file_num)+'.jpg')
      Combine_pictures = os.listdir(
      'C:\\Users\\user\\Desktop\django\\AI_generated_face_project_02\\album\\AI_part\\Combine_pictures\\')
      extract_contours_MTCNN = os.listdir('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\extract_contours_MTCNN\\')
      extract_faces_MTCNN = os.listdir('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\extract_faces_MTCNN')
      temp = os.listdir('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\')
      
      #==Upload files, file processing==#
      f = open("C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp02.txt", "w")
      f.write(str(file_num)+'.jpg')
      f.close()
     #==Upload files, file processing==#
    
      for a in Combine_pictures:
         os.remove("C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\Combine_pictures\\"+a)
      for b in extract_contours_MTCNN:
         os.remove('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\extract_contours_MTCNN\\'+b)
      for c in extract_faces_MTCNN:
         os.remove('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\extract_faces_MTCNN\\'+c)
      for d in temp:
         os.remove('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\'+d)
    ###############
    if select_model_item =='CycleGAN':
      file_num = len(os.listdir('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\media\\'))
      shutil.move('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\CycleGAN.jpg','C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\media\\'+str(file_num)+'.jpg')
      extract_faces_MTCNN = os.listdir('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\extract_faces_MTCNN')
      cycle_anime = os.listdir('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\Cycle_GAN\\cycle_anime\\')
      cycle_humen = os.listdir('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\Cycle_GAN\\cycle_humen\\')
      temp = os.listdir('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\')
          
      #==Upload files, file processing==#
      f = open("C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp02.txt", "w")
      f.write(str(file_num)+'.jpg')
      f.close()
     #==Upload files, file processing==#

      for h in extract_faces_MTCNN:
         os.remove('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\extract_faces_MTCNN\\'+h)
      for k in temp:
         os.remove('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\'+k)
      for l in cycle_anime:
         os.remove('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\Cycle_GAN\\cycle_anime\\'+l)
      for m in cycle_humen:
         os.remove('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\Cycle_GAN\\cycle_humen\\'+m)

    if select_model_item =='style_transfer':
       file_num = len(os.listdir('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\media\\'))
       shutil.move('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\style_transfer.jpg','C:\\Users\\user\\Desktop\django\\AI_generated_face_project_02\\album\\media\\'+str(file_num)+'.jpg')
       f = open("C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp02.txt", "w")
       f.write(str(file_num)+'.jpg')
       f.close()
       temp = os.listdir('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\')
       for f in temp:
         os.remove('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\'+f)
    
    if select_model_item =='BiSeNet':
       temp = os.listdir('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\')
       for g in temp:
         os.remove('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\'+g)
       
    if select_model_item =='face_swape':
       file_num = len(os.listdir('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\media\\'))
       shutil.move('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\output_Camera.jpg','C:\\Users\\user\\Desktop\django\\AI_generated_face_project_02\\album\\media\\'+str(file_num)+'.jpg')
       f = open("C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp02.txt", "w")
       f.write(str(file_num)+'.jpg')
       f.close()
       temp = os.listdir('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\')
       for k in temp:
         os.remove('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\'+k)
         
    if select_model_item =='face_similarity':
    
       file_num = len(os.listdir('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\media\\'))
       shutil.move('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\face_similarity_compare.jpg','C:\\Users\\user\\Desktop\django\\AI_generated_face_project_02\\album\\media\\'+str(file_num)+'.jpg')
       f = open("C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp02.txt", "w")
       f.write(str(file_num)+'.jpg')
       f.close()
       temp = os.listdir('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\')
       for ll in temp:
         os.remove('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\'+ll)
    
    
    if select_model_item =='Dlib':
       file_num = len(os.listdir('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\media\\'))
       shutil.move('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\face_makeup_Dlib\\Dlib.jpg','C:\\Users\\user\\Desktop\django\\AI_generated_face_project_02\\album\\media\\'+str(file_num)+'.jpg')
       face_makeup_Dlib = os.listdir('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\face_makeup_Dlib')
       temp = os.listdir('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\')

      #==Upload files, file processing==#
       f = open("C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp02.txt", "w")
       f.write(str(file_num)+'.jpg')
       f.close()
     #==Upload files, file processing==#
       for e in temp:
         os.remove('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\'+e)
       for f in face_makeup_Dlib:
         os.remove('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\AI_part\\face_makeup_Dlib\\'+f)
         