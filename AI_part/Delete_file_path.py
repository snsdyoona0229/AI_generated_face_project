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
        os.remove(os.path.join('media/'+i[0] ))
        
def ball(file,class_ai,style_transfer):
#    f = open("temp_files_info/temp.txt", "r")
#    R = f.read()

    if class_ai =="face_similarity":
       temp_face_x = open("temp_files_info/temp_x.txt", "r")
       m01 = temp_face_x.read()
       
       temp_face_y = open("temp_files_info/temp_y.txt", "r")
       m02 = temp_face_y.read()
                    
       shutil.copy('media/'+m01,os.path.join('temp/content_path.jpg'))
       
       shutil.copy('media/'+m02,os.path.join('temp/style_path.jpg'))
              
    if class_ai =="style_transfer":
       shutil.copy('media/'+m02,os.path.join('temp/'+style_transfer))

    if class_ai !="style_transfer":
       shutil.copy('media/'+file,os.path.join('temp/temp.jpg'))
    
def remove_file(select_model_item):
    if select_model_item =='pix2pix':
      file_num = len(os.listdir('media'))
      shutil.move('temp/pix2pix.jpg','media/'+str(file_num)+'.jpg')
      Combine_pictures = os.listdir(
      AI_part/Combine_pictures')
      extract_contours_MTCNN = os.listdir('AI_partext/act_contours_MTCNN')
      extract_faces_MTCNN = os.listdir('AI_part/extract_faces_MTCNN')
      temp = os.listdir('temp')
      
      #==Upload files, file processing==#
      f = open("temp02.txt", "w")
      f.write(str(file_num)+'.jpg')
      f.close()
     #==Upload files, file processing==#
    
      for a in Combine_pictures:
         os.remove("AI_part/Combine_pictures/"+a)
      for b in extract_contours_MTCNN:
         os.remove('AI_part/extract_contours_MTCNN/'+b)
      for c in extract_faces_MTCNN:
         os.remove('AI_part/extract_faces_MTCNN/'+c)
      for d in temp:
         os.remove('temp/'+d)
    ###############
    if select_model_item =='CycleGAN':
      file_num = len(os.listdir('media'))
      shutil.move('temp/CycleGAN.jpg','media/'+str(file_num)+'.jpg')
      extract_faces_MTCNN = os.listdir('AI_part/extract_faces_MTCNN')
      cycle_anime = os.listdir('AI_part/Cycle_GAN\\cycle_anime')
      cycle_humen = os.listdir('AI_part/Cycle_GAN/cycle_humen')
      temp = os.listdir('temp')
          
      #==Upload files, file processing==#
      f = open("temp02.txt", "w")
      f.write(str(file_num)+'.jpg')
      f.close()
     #==Upload files, file processing==#

      for h in extract_faces_MTCNN:
         os.remove('AI_part/extract_faces_MTCNN/'+h)
      for k in temp:
         os.remove('temp/'+k)
      for l in cycle_anime:
         os.remove('AI_part/Cycle_GAN/cycle_anime/'+l)
      for m in cycle_humen:
         os.remove('AI_part/Cycle_GAN/cycle_humen/'+m)

    if select_model_item =='style_transfer':
       file_num = len(os.listdir('media'))
       shutil.move('temp/style_transfer.jpg','media/'+str(file_num)+'.jpg')
       f = open("temp_files_info/temp02.txt", "w")
       f.write(str(file_num)+'.jpg')
       f.close()
       temp = os.listdir('temp')
       for f in temp:
         os.remove('temp'+f)
    
    if select_model_item =='BiSeNet':
       temp = os.listdir('temp')
       for g in temp:
         os.remove('temp/'+g)
       
    if select_model_item =='face_swape':
       file_num = len(os.listdir('media'))
       shutil.move('temp/output_Camera.jpg','media/'+str(file_num)+'.jpg')
       f = open("temp_files_info/temp02.txt", "w")
       f.write(str(file_num)+'.jpg')
       f.close()
       temp = os.listdir('temp')
       for k in temp:
         os.remove('temp/'+k)
         
    if select_model_item =='face_similarity':
    
       file_num = len(os.listdir('media'))
       shutil.move('temp/face_similarity_compare.jpg','media/'+str(file_num)+'.jpg')
       f = open("temp_files_info/temp02.txt", "w")
       f.write(str(file_num)+'.jpg')
       f.close()
       temp = os.listdir('album/temp')
       for ll in temp:
         os.remove('album/temp/'+ll)
    
    
    if select_model_item =='Dlib':
       file_num = len(os.listdir('media'))
       shutil.move('AI_part/face_makeup_Dlib/Dlib.jpg','media/'+str(file_num)+'.jpg')
       face_makeup_Dlib = os.listdir('album/AI_part/face_makeup_Dlib')
       temp = os.listdir('temp')

      #==Upload files, file processing==#
       f = open("temp_files_info/temp02.txt", "w")
       f.write(str(file_num)+'.jpg')
       f.close()
     #==Upload files, file processing==#
       for e in temp:
         os.remove('temp'+e)
       for f in face_makeup_Dlib:
         os.remove('AI_part/face_makeup_Dlib/'+f)
         
