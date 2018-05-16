import numpy as np
import math
import cv2
import sys
import scipy.io as sio
import os
import h5py

def add_margin(img, face_loc):
    crop_h = int(0.4 * (face_loc[3] - face_loc[1]))
    crop_w = int(0.4 * (face_loc[2] - face_loc[0]))
    img_h = img.shape[0]
    img_w = img.shape[1]
    #new_crop_low = crop_h + face_loc[1] - crop_h
    new_crop_high = crop_h + face_loc[3] + crop_h
    #new_crop_left = crop_w + face_loc[0] - crop_w
    new_crop_right = crop_w + face_loc[2] + crop_w
    replicate = cv2.copyMakeBorder(img, crop_h, crop_h, crop_w, crop_w, cv2.BORDER_REPLICATE)
    crop_face_img = replicate[face_loc[1] : new_crop_high, face_loc[0] : new_crop_right]
    return crop_face_img

#产生新图像
def add_augment_img(img):
    translateX = np.random.uniform(-0.1, 0.1)
    translateY = np.random.uniform(-0.1, 0.1)
    scale = np.random.uniform(0.9, 1.1)
    rotate_angle = np.random.uniform(-10, 10)
    
    def translate(image, x, y):
        # 定义平移矩阵
        M = np.float32([[1, 0, x], [0, 1, y]])
        shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # 返回转换后的图像
        return shifted
    
    # 定义旋转rotate函数
    def rotate(image, angle, center=None, scale=1.0):
        # 获取图像尺寸
        (h, w) = image.shape[:2]

        # 若未指定旋转中心，则将图像中心设为旋转中心
        if center is None:
            center = (w / 2, h / 2)

        # 执行旋转
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        # 返回旋转后的图像
        return rotated
    
    img = translate(img, translateX, translateY)
    
    rotated = rotate(img, rotate_angle, scale = scale)
    
    return rotated

    
def load_wiki(wiki_path, num_data = None, target_size = (224,224)):
    mat_path = wiki_path + 'wiki_with_age.mat'
    
    data = sio.loadmat(mat_path)
    wiki_data = data['wiki'][0][0]
        
    limit_age_low = 10    #年龄下限
    limit_age_high = 90   #年龄上限
    num_age_class = limit_age_high - limit_age_low   #年龄范围
    
    num_full_data = len(wiki_data[6][0])
    if num_data is None:
        num_data = num_age_class * 1000  #总数据量
        
    y_data_age = np.zeros([num_data],dtype = "uint8")
    y_data_gender = np.zeros([num_data],dtype = "uint8")
    
    full_X_data = np.zeros([num_data, 224, 224, 3], dtype = "uint8")
    
    #用于存储真实数据数量
    data_count = 0
    
    #每个年龄和性别应该获取的数据量
    assert (num_data / num_age_class / 2).is_integer() == True, "num_every_age_gender = " + str(num_data / num_age_class / 2) +"不是整数"
    num_every_age_gender = int(num_data / num_age_class / 2)       
    print("num_every_age_gender: ", num_every_age_gender)
    
    #按年龄及计算相应年龄有几个图（从wiki数据库中读取的图片数量），0-79为女，80-159为男
    counter_every_age_gender = np.zeros(2 * num_age_class, np.int32)     
    #按年龄和性别存储相应年龄及性别在full_X_data中的索引位置
    index_every_age_gender = np.zeros([2 * num_age_class, num_every_age_gender], np.int32)  
        
    i = 0
    
    #先读取wiki样本库中本来就有的样本
    while i < num_full_data and data_count < num_data:
        face_score =wiki_data[6][0][i]
        if face_score != float("-inf"):         #如果face_score == -inf说明不存在脸,这个比例比较大，因此先排除
            full_path = wiki_path + wiki_data[2][0][i][0]
            img = cv2.imread(full_path)
            age = int(wiki_data[8][0][i])            #有些age在正常值之外要排除
            date_of_birth = wiki_data[0][0][i]  #下面的657438是出生于1800年的Matlab serial date number
            gender = wiki_data[3][0][i]        #有一些gender==None会引起一场，需要排除
            if img is not None and gender == gender and date_of_birth > 657438 and age >= limit_age_low and age < limit_age_high:
                gender = int(gender)  
                face_loc = wiki_data[5][0][i][0]
                face_loc = face_loc.astype("int32")
                roi_img = add_margin(img, face_loc)    
                face_img = cv2.resize(roi_img, target_size)
                index_in_age_gender = (age - limit_age_low) + gender * num_age_class #10-90岁映射为0-80 后面加上那项是为了区分性别
                
                if counter_every_age_gender[index_in_age_gender] < num_every_age_gender: 
                    index_every_age_gender[index_in_age_gender, counter_every_age_gender[index_in_age_gender]] = data_count
                    counter_every_age_gender[index_in_age_gender] = counter_every_age_gender[index_in_age_gender] + 1

                    full_X_data[data_count] = face_img                
                    y_data_age[data_count] = age
                    y_data_gender[data_count] = gender
                    data_count += 1
        i += 1
        
    print("read from file finished")
    
    
    print("before: data_count: ", data_count)
    
    #new_counter_every_age_gender = counter_every_age_gender.copy() #new_counter_every_age_gender是原来数值加上add_margin的数据，是全部的量
    #给不满足样本数的年龄添加样本，通过add_margin对原有的图片进行加工操作
    for cur_gender in range(2): 
        for cur_age in range(limit_age_low, limit_age_high):      
            index_in_age_gender = (cur_age - limit_age_low) + cur_gender * num_age_class
            cur_age_gender_start_index = index_in_age_gender * num_every_age_gender
            #full_X_data[cur_age_gender_start_index : cur_age_gender_start_index + index_in_age_gender] \
            #                 = full_X_data[index_every_age[index_in_age]]
            
            #防止哪个年龄段没有基础样本，这样也产生不了新样本
            assert counter_every_age_gender[index_in_age_gender] > 0, "年龄"+ str(cur_age) + "性别"+ str(cur_gender) + "没有数据" 
            
            if counter_every_age_gender[index_in_age_gender] < num_every_age_gender:          #说明样本数不够，需要添加新样本
                for cur_augment in range(num_every_age_gender - counter_every_age_gender[index_in_age_gender]):
                    #从样本数中随机选1个索引值，去索引index_every中的值index_every_age_g[age,cur_choice]，注意这是老样本，不会索引新增加的样本
                    cur_choice = np.random.choice(counter_every_age_gender[index_in_age_gender], replace = False)  
                    full_X_data[data_count] = add_augment_img(full_X_data[index_every_age_gender[index_in_age_gender,cur_choice]])
                    y_data_age[data_count] = cur_age
                    y_data_gender[data_count] = cur_gender
                    data_count += 1
            #y_data_age[cur_age_start_index : cur_age_start_index + num_every_age] = cur_age
    
    #assert data_count == num_data, "获取的数据数量对不上：data_count = " + str(data_count) + ", num_data = " + str(num_data)
    print("add_augment_img finished")

    return full_X_data[:data_count], y_data_age, y_data_gender

def get_data_for_train_val_test(X_data, y_data_age, y_data_gender, num_training=49000, num_validation=1000, num_test=1000, subtract_mean = False):
    
    data_index = np.arange(X_data.shape[0])
    np.random.shuffle(data_index)
    X_data = X_data[data_index]
    y_data_age = y_data_age[data_index]
    y_data_gender = y_data_gender[data_index]
    
    #for i in data_index:
    #    temp = cv2.cvtColor(X_data[i], cv2.COLOR_BGR2GRAY)
    #    X_data[i] = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
        #(R*299 + G*587 + B*114 + 500) / 1000    
    
    X_train = X_data[:-(num_validation + num_test)].copy()
    y_age_train = y_data_age[:-(num_validation + num_test)].copy()
    y_gender_train = y_data_gender[:-(num_validation + num_test)].copy()
    
    X_val = X_data[X_train.shape[0] : X_data.shape[0]-num_test].copy()
    y_age_val = y_data_age[X_train.shape[0] : X_data.shape[0]-num_test].copy()
    y_gender_val = y_data_gender[X_train.shape[0] : X_data.shape[0]-num_test].copy()
    
    if num_test != 0:
        X_test = X_data[-num_test:].copy()
        y_age_test = y_data_age[-num_test:].copy()
        y_gender_test = y_data_gender[-num_test:].copy()
    else:
        X_test = np.array([])
        y_age_test = np.array([])
        y_gender_test = np.array([])
        
    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_data, axis=0).astype("uint8")
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image


    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_age_train': y_age_train, 'y_gender_train': y_gender_train,
      'X_val': X_val, 'y_age_val': y_age_val,'y_gender_val': y_gender_val,
      'X_test': X_test, 'y_age_test': y_age_test, 'y_gender_test': y_gender_test
    }
    
    
def get_wiki_crop_data(wiki_path = None, num_training=58000, num_validation=2000, num_test=1000):
    if wiki_path is None:
        if sys.platform == "linux" :
            wiki_path = "/devdata/wiki/"
        else:
            wiki_path = "G:\\MachineLearning\\wiki\\wiki\\"

    mat_path = wiki_path + 'wiki_with_age.mat'
    
    # Create a new file
    f = None
    X_data_age_gender = None
    y_data_age = None
    y_data_gender = None
    if (num_training is not None) and (num_validation is not None) and (num_test is not None):
        num_all_data =  num_training + num_validation + num_test
    else:
        num_all_data = None
    #wiki_crop_dataset = f.create_dataset('wiki_cropface_data', dtype = "int32")
    if os.path.exists('/devdata/wiki_cropface_data.h5'):
        f = h5py.File('/devdata/wiki_cropface_data.h5', "r")
        wiki_cropface_group = f["wiki_cropface_group"]
        X_data_age_gender = np.array(wiki_cropface_group["X_data_age_gender"][:num_all_data])
        y_data_age = np.array(wiki_cropface_group["y_data_age"][:num_all_data])
        y_data_gender = np.array(wiki_cropface_group["y_data_gender"][:num_all_data])
    else:
        f = h5py.File('/devdata/wiki_cropface_data.h5',"w")
        X_data_age_gender, y_data_age, y_data_gender = load_wiki(wiki_path, num_data = num_all_data)
        wiki_cropface_group = f.create_group("wiki_cropface_group")
        wiki_cropface_group.create_dataset('X_data_age_gender', dtype = "uint8", data = X_data_age_gender)
        wiki_cropface_group.create_dataset('y_data_age', dtype = "uint8", data = y_data_age)
        wiki_cropface_group.create_dataset('y_data_gender', dtype = "uint8", data = y_data_gender)
    print("read from wiki finished")
    data_dict = get_data_for_train_val_test(X_data_age_gender, y_data_age, y_data_gender,  num_training, num_validation, num_test)
    
    f.close()
    return data_dict