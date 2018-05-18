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


tf.app.flags.DEFINE_string('img_path', "/images/trump.jpg", 'how often to print training status')
FLAGS = tf.app.flags.FLAGS

# 传入的命令行参数
for f in sys.argv[1:]:
    # opencv 读取图片，并显示
    

    # 摘自官方文档：
    # image is a numpy ndarray containing either an 8bit grayscale or RGB image.
    # opencv读入的图片默认是bgr格式，我们需要将其转换为rgb格式；都是numpy的ndarray类。
    b, g, r = cv2.split(img)    # 分离三个颜色通道
    img2 = cv2.merge([r, g, b])   # 融合三个颜色通道生成新图片

    dets = detector(img, 1) #使用detector进行人脸检测 dets为返回的结果
    print("Number of faces detected: {}".format(len(dets)))  # 打印识别到的人脸个数
    # enumerate是一个Python的内置方法，用于遍历索引
    # index是序号；face是dets中取出的dlib.rectangle类的对象，包含了人脸的区域等信息
    # left()、top()、right()、bottom()都是dlib.rectangle类的方法，对应矩形四条边的位置
    for index, face in enumerate(dets):
        print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(), face.bottom()))

        # 在图片中标注人脸，并显示
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.namedWindow(f, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(f, img)

# 等待按键，随后退出，销毁窗口
k = cv2.waitKey(0)
cv2.destroyAllWindows()


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
        print(faces_tensor.shape)
        age, gender = run_test(sess, input_tensor)
        
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