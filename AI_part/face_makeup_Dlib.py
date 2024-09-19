from PIL import Image, ImageDraw
import face_recognition
import dlib


def Dlib_face_makeup(color_01,color_02,color_03,color_04,color_05,color_06,color_07,color_08,color_09):

  image = face_recognition.load_image_file('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\temp\\temp.jpg')

  # Find all facial features in all the faces in the image
  face_landmarks_list = face_recognition.face_landmarks(image)
  pil_image = Image.fromarray(image)
  for face_landmarks in face_landmarks_list:
    d = ImageDraw.Draw(pil_image, 'RGBA')

    # Make the eyebrows into a nightmare
    d.polygon(face_landmarks['left_eyebrow'], fill=(color_01,color_02,color_03, 128))
    d.polygon(face_landmarks['right_eyebrow'], fill=(color_01,color_02,color_03, 128))
    d.line(face_landmarks['left_eyebrow'], fill=(color_01,color_02,color_03, 150), width=5)
    d.line(face_landmarks['right_eyebrow'], fill=(color_01,color_02,color_03, 150), width=5)

      # Gloss the lips
    d.polygon(face_landmarks['top_lip'], fill=(color_04,color_05, color_06, 128))
    d.polygon(face_landmarks['bottom_lip'], fill=(color_04,color_05, color_06, 128))
    d.line(face_landmarks['top_lip'], fill=(color_04, color_05, color_06, 64), width=8)
    d.line(face_landmarks['bottom_lip'], fill=(color_04, color_05, color_06, 64), width=8)

    # Sparkle the eyes
    d.polygon(face_landmarks['left_eye'], fill=(color_07, color_08, color_09, 30))
    d.polygon(face_landmarks['right_eye'], fill=(color_07, color_08, color_09, 30))

    # Apply some eyeliner
    d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
    d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

    #pil_image.show()
    pil_image.save('C:\\Users\\user\\Desktop\\django\\AI_generated_face_project_02\\album\\AI_part\\face_makeup_Dlib\\Dlib.jpg')