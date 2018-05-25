import tensorflow as tf
import numpy as np
import math
import time
import os
import tensornets as nets
import cv2
import dlib
import sys
from age_gender_model import run_test


tf.app.flags.DEFINE_string('img_path', "/images/trump.jpg", 'the path of imgaine')
tf.app.flags.DEFINE_string('model_path', "train_models/age_gender_tensornets_wiki_cropface_model_20180519_123211", 'how often to print training status')
FLAGS = tf.app.flags.FLAGS


def main(_):

    tf.reset_default_graph()
    
    detector = dlib.get_frontal_face_detector() #获取人脸分类器
    
    with tf.Session() as sess:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        
        dets = detector(img, 1) #使用detector进行人脸检测 dets为返回的结果
        HH, WW, _ = img.shape
        faces_tensor = np.array([])
        for index, face in enumerate(dets):

            # 在图片中标注人脸，并显示
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            w = right - left
            h = bottom - top
            left = int(max(0, left - 0.2 * w))
            top = int(max(0, top - 0.4 * h))
            right = int(min(WW, right + 0.2 * w))
            bottom = int(min(HH, bottom + 0.2 * h))
            face_img = img[top:bottom, left:right].copy()
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
            face_img = cv2.resize(face_img, (224, 224))
            np.append(faces_tensor, face_img)
        print("faces_tensor.shape", faces_tensor.shape)
        age, gender = run_test(sess, input_tensor, model_path)
        
        for i in range(len(dets)):
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if top - 10 < 0:
                top = top + 20
            else:
                top = top - 10
            gender_text = "male" if gender[i] == 1 else "female"
            cv2.putText(frame,str(age[i]) + ", " + gender_text,(left,top), font, 1,(0,255,0),2,cv2.LINE_AA)
            #print("age: " + str(age[i]) + "gender: " + str(gender[i]))
        
        pass
    
    
if __name__ == '__main__':
    tf.app.run()