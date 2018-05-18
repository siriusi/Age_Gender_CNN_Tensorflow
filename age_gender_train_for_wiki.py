import tensorflow as tf
import numpy as np
import math
import time
import os
import tensornets as nets
import cv2
import sys
from load_wiki_cropface import get_wiki_crop_data
from age_gender_model import run_train


tf.app.flags.DEFINE_integer('batch_size', 500, 'batch size')
tf.app.flags.DEFINE_integer('num_train_data', 58000, 'number of train data')
tf.app.flags.DEFINE_integer('num_val_data', 2000, 'number of val data')
tf.app.flags.DEFINE_integer('num_epochs', 4, 'number of epochs')
tf.app.flags.DEFINE_float('learning_rate', 1e-5, 'init learning rate')

tf.app.flags.DEFINE_integer('print_every', 5, 'how often to print training status')
tf.app.flags.DEFINE_boolean('is_save_summary', True, 'is save summary data')

FLAGS = tf.app.flags.FLAGS

def main(_):
    
    print("========================" + time.strftime("%Y%m%d_%H:%M:%S", time.localtime()) + "=========================")
    # Invoke the above function to get our data.  
    age_gender_dict = get_wiki_crop_data(num_training=FLAGS.num_train_data, num_validation=FLAGS.num_val_data, num_test=0)

    print('Train data shape: ', age_gender_dict["X_train"].shape)
    print('Train labels shape for age: ', age_gender_dict["y_age_train"].shape)
    print('Train labels shape for gender: ', age_gender_dict["y_gender_train"].shape)

    print('Validation data shape: ', age_gender_dict["X_val"].shape)
    print('Validation labels shape for age: ', age_gender_dict["y_age_val"].shape)
    print('Validation labels shape for gender: ', age_gender_dict["y_gender_val"].shape)

    print('Test data shape: ', age_gender_dict["X_test"].shape)
    print('Test labels shape for age: ', age_gender_dict["y_age_test"].shape)
    print('Test labels shape for gender: ', age_gender_dict["y_gender_test"].shape)

    tf.reset_default_graph()
  
    with tf.Session() as sess:
        #with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0" 
        #sess.run(tf.global_variables_initializer())
        #print('Training')
        run_train(sess,age_gender_dict,num_class = 80, epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size,
                  print_every=FLAGS.print_every, learning_rate = FLAGS.learning_rate, 
                  is_save_summary = FLAGS.is_save_summary)
        pass
    print("==================================================================")
    print("\n")
    
if __name__ == '__main__':
    tf.app.run()