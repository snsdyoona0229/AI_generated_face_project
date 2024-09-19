from django.shortcuts import render, redirect
from albumapp import models
from django.contrib.auth import authenticate
from django.contrib import auth
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import sys
import shutil
from django.contrib import messages
import pathlib
sys.path.append('AI_part')
import webbrowser
import Data_pre_processing
import output_test
import face_makeup_Dlib
import Cycle_GAN
from subprocess import call
import Delete_file_path
import face_similarity



def index(request):
	#apple()
	albums = models.AlbumModel.objects.all().order_by('-id')  #讀取所有相簿 
	totalalbum = len(albums)  #相簿總數
	photos = []  #每一相簿第1張相片串列
	lengths = []  #每一相簿的相片總數串列
	for album in albums:
		photo = models.PhotoModel.objects.filter(palbum__atitle=album.atitle).order_by('-id')  #讀取相片
		lengths.append(len(photo))  #加入相片總數
		if len(photo) == 0:  #若無相片加入空字串
			photos.append('')
		else:
			photos.append(photo[0].purl)  #加入第1張相片
	return render(request, "index.html", locals())
	
def albumshow(request, albumid=None):  #顯示相簿
	album = albumid  #以區域變數傳送給html
	photos = models.PhotoModel.objects.filter(palbum__id=album).order_by('-id')  #讀取所有相片
	#apple = models.PhotoModel.objects.get(id='1')  #取得相簿
	monophoto = photos[0]  #第1張相片
	totalphoto = len(photos)  #相片總數
	#apple.delete()
	return render(request, "albumshow.html", locals())
	
def albumphoto(request, photoid=None, albumid=None):  #顯示單張相片
	album = albumid  #以區域變數傳送給html
	photo = models.PhotoModel.objects.get(id=photoid)  #取得點選的相片
	photo.phit += 1  #點擊數加1
	photo.save()  #儲存資料
	return render(request, "albumphoto.html", locals())

def login(request):  #登入
	messages = ''  #初始時清除訊息
	if request.method == 'POST':  #如果是以POST方式才處理
		name = request.POST['username'].strip()  #取得輸入帳號
		password = request.POST['passwd']  #取得輸入密碼
		user1 = authenticate(username=name, password=password)  #驗證
		if user1 is not None:  #驗證通過
			if user1.is_active:  #帳號有效
				auth.login(request, user1)  #登入
				return redirect('/adminmain/')  #開啟管理頁面
			else:  #帳號無效
				message = '帳號尚未啟用！'
		else:  #驗證未通過
			message = '登入失敗！'
	return render(request, "login.html", locals())

def logout(request):  #登出
	auth.logout(request)
	return redirect('/index/')

def adminmain(request, albumid=None):  #管理頁面
	if albumid == None:  #按相簿管理鈕進管理頁面
		albums = models.AlbumModel.objects.all().order_by('-id')
		totalalbum = len(albums)
		photos = []
		lengths = []
		for album in albums:
			photo = models.PhotoModel.objects.filter(palbum__atitle=album.atitle).order_by('-id')
			lengths.append(len(photo))
			if len(photo) == 0:
				photos.append('')
			else:
				photos.append(photo[0].purl)
	else:  #按刪除相簿鈕
		album = models.AlbumModel.objects.get(id=albumid)  #取得相簿
		photo = models.PhotoModel.objects.filter(palbum__atitle=album.atitle).order_by('-id')  #取得所有相片
		for photounit in photo:  #刪除所有相片檔案
			os.remove(os.path.join(settings.MEDIA_ROOT, photounit.purl ))
		album.delete()  #移除相簿
		return redirect('/adminmain/')
	return render(request, "adminmain.html", locals())

def adminadd(request):  #新增相簿
	message = ''
	title = request.POST.get('album_title', '')  #取得輸入資料
	location = request.POST.get('album_location', '')
	desc = request.POST.get('album_desc', '')
    
	if title=='':  #按新增相簿鈕進入此頁
		message = '相簿名稱一定要填寫...'
	else:  #按確定新增鈕
		unit = models.AlbumModel.objects.create(atitle=title, alocation=location, adesc=desc)
		unit.save()
		return redirect('/adminmain/')
	return render(request, "adminadd.html", locals())

def adminfix(request, albumid=None, photoid=None, deletetype=None):  #相簿維護
    model_AI = request.POST.get('model', '')  #取得輸入資料
    album = models.AlbumModel.objects.get(id=albumid)  #取得指定相簿
    photos = models.PhotoModel.objects.filter(palbum__id=albumid).order_by('-id')
    totalphoto = len(photos)
    
    up_file = ''
    
    if photoid != None:  #不是由管理頁面進入本頁面
        if photoid == 999999:  #按更新及上傳資料鈕
            album.atitle = request.POST.get('album_title', '')  #更新相簿資料
            album.alocation = request.POST.get('album_location', '')
            album.adesc = request.POST.get('album_desc', '')
            album.save()
            files = []  #上傳相片串列
            descs = []  #相片說明串列
            picurl = ["ap_picurl1"]
            subject = ["ap_subject1"]
            
            if str(model_AI) == 'style_transfer' or str(model_AI) =='face_similarity':

               picurl = ["ap_picurl2","ap_picurl3"]
               subject = ["ap_subject2","ap_subject3"]
            
            
            for count in range(0,len(picurl),1):
                files.append(request.FILES.get(picurl[count], ''))
                descs.append(request.POST.get(subject[count], ''))
            i = 0
            for upfile in files:  
                if upfile != '' and descs[i] != '':	
                    up_file =  upfile                
                    fs = FileSystemStorage()  #上傳檔案
                    filename = fs.save(upfile.name, upfile)
                    unit = None
                    if str(album) == '原始圖像':
                       unit = models.PhotoModel.objects.create(palbum=album, psubject=descs[i], purl=upfile)  #寫入資料庫
                    else:
                       unit = models.PhotoModel.objects.create(palbum=models.AlbumModel.objects.all()[0], psubject=descs[i], purl=upfile)  #寫入資料庫
                                       
                    #==Upload files, file processing==#
                    f = open("C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp.txt", "w")
                    f.write(str(upfile))
                    f.close()
                    #==Upload files, file processing==#:
                    unit.save()
                    if str(model_AI) == 'face_similarity':
                       if i == 0:
                          f = open("C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp_x.txt", "w")
                          f.write(str(upfile))
                          f.close()
                       if i == 1:
                          f = open("C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp_y.txt", "w")
                          f.write(str(upfile))
                          f.close()
                    

                    if str(model_AI) == 'style_transfer':
                       style = ['content_path.jpg','style_path.jpg']
                       Delete_file_path.ball(str(upfile),str(model_AI),style[i])                   
                    i += 1
            #==RUN_AI==#
            if str(model_AI) == 'face_similarity':
                     
               Delete_file_path.ball(str(up_file),str(model_AI),str(model_AI))
               
               face_similarity.face_similarity_compare()
               
               Delete_file_path.remove_file(str(model_AI))
               temp02 = open("C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp02.txt", "r")
               m = temp02.read()
               face_similarity_s = models.AlbumModel.objects.all()[7]
               face_similarity_g = models.PhotoModel.objects.create(palbum=face_similarity_s, psubject='face_similarity', purl=str(m))  #寫入資料庫 
               face_similarity_g.save()
                   
            if str(model_AI) =="face_swape":
            
               style = []            
               Delete_file_path.ball(str(up_file),str(model_AI),style)
               call(["python", "C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\face_swape.py"])
               Delete_file_path.remove_file(str(model_AI))
               temp02 = open("C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp02.txt", "r")
               m = temp02.read()
               face_swape_s = models.AlbumModel.objects.all()[6]
               face_swape_g = models.PhotoModel.objects.create(palbum=face_swape_s, psubject='face_swape', purl=str(m))  #寫入資料庫 
               face_swape_g.save()
            
            if str(model_AI) == 'pix2pix': 
               style = []            
               Delete_file_path.ball(str(up_file),str(model_AI),style)
               Data_pre_processing.save_extract_face()
               Data_pre_processing.prepare_training_data()
               Data_pre_processing.Combine_pictures()
               output_test.count_img()#AI_detection
               Delete_file_path.remove_file(str(model_AI))
               temp02 = open("C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp02.txt", "r")
               m = temp02.read()
                  

               Gan = models.AlbumModel.objects.all()[1]
               Pix2Pix = models.PhotoModel.objects.create(palbum=Gan, psubject='pix2pix', purl=str(m))  #寫入資料庫 
               Pix2Pix.save()
            if str(model_AI) == 'Dlib':
            
               textInput01 = request.POST.get('textInput01', '')  #取得輸入資料
               textInput02 = request.POST.get('textInput02', '')  #取得輸入資料
               textInput03 = request.POST.get('textInput03', '')  #取得輸入資料
               textInput04 = request.POST.get('textInput04', '')  #取得輸入資料
               textInput05 = request.POST.get('textInput05', '')  #取得輸入資料
               textInput06 = request.POST.get('textInput06', '')  #取得輸入資料
               textInput07 = request.POST.get('textInput07', '')  #取得輸入資料
               textInput08 = request.POST.get('textInput08', '')  #取得輸入資料
               textInput09 = request.POST.get('textInput09', '')  #取得輸入資料
               style = []            
               Delete_file_path.ball(str(up_file),str(model_AI),style)
               face_makeup_Dlib.Dlib_face_makeup(int(textInput01),int(textInput02),int(textInput03),int(textInput04),int(textInput05),int(textInput06),int(textInput07),int(textInput08),int(textInput09))
               Delete_file_path.remove_file(str(model_AI))
               
               temp02 = open("C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp02.txt", "r")
               m = temp02.read()

               D = models.AlbumModel.objects.all()[2]
               Dlib_n = models.PhotoModel.objects.create(palbum=D, psubject='Dlib', purl=str(m))  #寫入資料庫 
               Dlib_n.save()
               
            if str(model_AI) == 'CycleGAN':
               CycleGAN_condition = request.POST.get('CycleGAN_condition', '')  #取得輸入資料
               style = []            
               Delete_file_path.ball(str(up_file),str(model_AI),style)
               Data_pre_processing.save_extract_face()
               Cycle_GAN.CycleGAN_humen_anime(str(CycleGAN_condition))
               Cycle_GAN.Combine_pictures()
               Delete_file_path.remove_file(str(model_AI))
              
               temp02 = open("C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp02.txt", "r")
               m = temp02.read()
               
               
               Circle = models.AlbumModel.objects.all()[3]
               Circle_g = models.PhotoModel.objects.create(palbum=Circle, psubject='Cycle_GAN', purl=str(m))  #寫入資料庫 
               Circle_g.save()
               
            if str(model_AI) == 'style_transfer':
               #style_transfer.train_style_transfer()
               call(["python", "C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\style_transfer.py"])
               Delete_file_path.remove_file(str(model_AI))
               temp02 = open("C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp02.txt", "r")
               m = temp02.read()
               style_transfer_s = models.AlbumModel.objects.all()[4]
               style_transfer_g = models.PhotoModel.objects.create(palbum=style_transfer_s, psubject='style_transfer', purl=str(m))  #寫入資料庫 
               style_transfer_g.save()
            if str(model_AI) == 'BiSeNet':
            
               textInput01 = request.POST.get('textInput01', '')  #取得輸入資料
               textInput02 = request.POST.get('textInput02', '')  #取得輸入資料
               textInput03 = request.POST.get('textInput03', '')  #取得輸入資料
               textInput04 = request.POST.get('textInput04', '')  #取得輸入資料
               textInput05 = request.POST.get('textInput05', '')  #取得輸入資料
               textInput06 = request.POST.get('textInput06', '')  #取得輸入資料
               textInput07 = request.POST.get('textInput07', '')  #取得輸入資料
               textInput08 = request.POST.get('textInput08', '')  #取得輸入資料
               textInput09 = request.POST.get('textInput09', '')  #取得輸入資料
               
               str_color = str(textInput01)+","+str(textInput02)+","+str(textInput03)+","+str(textInput04)+","+str(textInput05)+","+str(textInput06)+","+str(textInput07)+","+str(textInput08)+","+str(textInput09)
               
               cc = open("C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\face-makeup.PyTorch-master\\color_temp.txt", "w")
               cc.write(str_color)
               cc.close()
            
               style = []            
               Delete_file_path.ball(str(up_file),str(model_AI),style)
               call(["python", "C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\face-makeup.PyTorch-master\\makeup.py"])
               Delete_file_path.remove_file(str(model_AI))
               temp02 = open("C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp02.txt", "r")
               m = temp02.read()
               BiSeNet_s = models.AlbumModel.objects.all()[5]
               BiSeNet_g = models.PhotoModel.objects.create(palbum=BiSeNet_s, psubject='BiSeNet', purl=str(m))  #寫入資料庫 
               BiSeNet_g.save()
               
            if str(model_AI) == 'StyleGAN':
               
               url = "https://colab.research.google.com/github/danielroich/PTI/blob/main/notebooks/inference_playground.ipynb" # 注意:"http://"不可省略
               webbrowser.open(url)
  
              
            #==RUN_AI==#            
            return redirect('/adminfix/' + str(album.id) + '/')
        elif deletetype == 'update':  #更新相片說明
            photo = models.PhotoModel.objects.get(id=photoid)
            photo.psubject = request.POST.get('ap_subject', '')  #取得相片說明
            photo.save()  #存寫入資料庫

            return redirect('/adminfix/' + str(album.id) + '/')
        elif deletetype=='delete':  #刪除相片
            photo = models.PhotoModel.objects.get(id=photoid)
            #photo = models.PhotoModel.objects.all()
            #photo = photo.id
            os.remove(os.path.join(settings.MEDIA_ROOT, photo.purl ))  #刪除相片檔
            photo.delete()  #從資料庫移除
            return redirect('/adminfix/' + str(album.id) + '/')
    return render(request, "adminfix.html", locals())
    
    
    
    
    
############################################################




