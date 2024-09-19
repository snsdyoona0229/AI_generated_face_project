import tensorflow as tf
import os
import time
from matplotlib import pyplot as plt
from IPython import display
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
import sys
sys.path.append('AI_part')
from pix2pix_model import load_image_test,load,resize,normalize,random_crop,generate_images
#from Data_pre_processing import extract_faces,save_extract_face,reshape_for_polyline, prepare_training_data,Combine_pictures
import Data_pre_processing
import output_test
import cv2


BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_png(image)

  w = tf.shape(image)[1]

  w = w // 2
  real_image = image[:, :w, :]
  input_image = image[:, w:, :]

  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image


def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image
  
def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]
  
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

generator = load_model("h5_file/pix2pix_h5/generatorw.h5")
discriminator = load_model("AI_part/h5_file/pix2pix_h5/discriminatorw.h5")

def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']
  
  
  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  files_save = os.listdir('AI_part/Combine_pictures')
  for j in files_save:
    plt.savefig('temp/pix2pix.jpg')

  def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image
  
def count_img():
  test_dataset01 = tf.data.Dataset.list_files('AI_part/Combine_pictures/*.jpg')
  #test_dataset01 = tf.data.Dataset.list_files('media/*.jpg')
  test_dataset01 = test_dataset01.map(load_image_test)
  test_dataset01 = test_dataset01.batch(BATCH_SIZE)
  # Run the trained model on a few examples from the test dataset
  for inp, tar in test_dataset01.take(len(test_dataset01)):
    generate_images(generator, inp, tar)
    
