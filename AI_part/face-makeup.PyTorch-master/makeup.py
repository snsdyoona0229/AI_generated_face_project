import cv2
import os
import numpy as np
from skimage.filters import gaussian
import os
import sys
sys.path.append('AI_part/face-makeup.PyTorch-master')
from test import evaluate
import argparse




def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def hair(image, parsing, part=17, color=[230, 50, 20]):
    b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part == 12 or part == 13:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if part == 17:
        changed = sharpen(changed)

    changed[parsing != part] = image[parsing != part]
    return changed


#if __name__ == '__main__':
def make_up_run():
    # 1  face
    # 11 teeth
    # 12 upper lip
    # 13 lower lip
    # 17 hair
    # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
            #         10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

    table = {
        'hair': 17,
        'upper_lip': 12,
        'lower_lip': 13,
        'l_brow':2,
        'r_brow':3,
        'l_eye':5,
        'r_eye':6,
        'eye_g':7,
    }

    image_path = 'temp/temp.jpg'
    cp = 'AI_part/face-makeup.PyTorch-master/cp/79999_iter.pth'

    image = cv2.imread(image_path)
    ori = image.copy()
    parsing = evaluate(image_path, cp)
    parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

    parts = [table['upper_lip'], table['lower_lip'], table['l_brow'], table['r_brow'],table['l_eye'], table['r_eye']]
    #[table['upper_lip'], table['lower_lip'], table['l_brow'], table['r_brow'],table['l_eye'], table['r_eye'], table['eye_g']]
    #colors = [[230, 50, 20], [20, 70, 180], [20, 70, 180]]
    cc = open("AI_part/face-makeup.PyTorch-master/color_temp.txt", "r")
    color_read = cc.read()
    color_num = color_read.split(",")
    
    colors = [[int(color_num[5]),int(color_num[4]),int(color_num[3])], [int(color_num[5]),int(color_num[4]),int(color_num[3])],[int(color_num[2]),int(color_num[1]),int(color_num[0])],[int(color_num[2]),int(color_num[1]),int(color_num[0])],[int(color_num[8]),int(color_num[7]),int(color_num[6])],[int(color_num[8]),int(color_num[7]),int(color_num[6])]]
    
    for part, color in zip(parts, colors):
        image = hair(image, parsing, part, color)

    #cv2.imshow('image', cv2.resize(ori, (512, 512)))
    #cv2.imshow('color', cv2.resize(image, (512, 512)))
    
    file_num = len(os.listdir('media'))
    cv2.imwrite('media/'+str(file_num)+'.jpg', image)
    #==Upload files, file processing==#
    f = open("temp02.txt", "w")
    f.write(str(file_num)+'.jpg')
    f.close()
    #==Upload files, file processing==#


make_up_run()


