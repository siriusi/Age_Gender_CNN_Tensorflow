import tensorflow as tf
import numpy as np
import math
#import matplotlib.pyplot as plt
#%matplotlib inline
import time
import os
import tensornets as nets
import cv2
import sys
from load_wiki_cropface import get_wiki_crop_data

from tensornets.utils import var_scope
from tensornets.layers import dropout
from tensornets.layers import fc
from tensornets.ops import softmax
from tensornets.ops import reduce_mean


@var_scope('mobilenet100')
def age_gender_model(inputs, outputs_age = None, outputs_gender = None, num_class = 80, batch_size = 100, learning_rate = 1e-5):
    
    is_training = True
    if outputs_age == None:
        is_training = False
        
    cnn_net = nets.MobileNet100(inputs, is_training = is_training, classes = num_class, stem=True)
    
    cnn_net = reduce_mean(cnn_net, [1, 2], name='avgpool')
    cnn_net = dropout(cnn_net, keep_prob=0.999, is_training=is_training, scope='dropout')
    
    cnn_net_age = fc(cnn_net, num_class, scope='logits_age')
    cnn_net_gender = fc(cnn_net, 2, scope='logits_gender')
    
    cnn_net_age = softmax(cnn_net_age, name='probs_age')
    cnn_net_gender = softmax(cnn_net_gender, name='probs_age')

    cnn_predictions_age = tf.reduce_sum(tf.multiply(tf.range(0, 80, dtype = np.float32), cnn_net_age), axis = 1, name = "cnn_predictions_age")
    cnn_predictions_gender = tf.argmax(cnn_net_gender, axis = 1, name = "cnn_predictions_gender")
    
    if is_training == False:
        return cnn_predictions_age, cnn_predictions_gender
    
    age_loss = tf.divide(tf.sqrt(tf.reduce_sum((cnn_predictions_age - tf.cast(outputs_age, tf.float32)) ** 2), name = "sqrt_age_loss"), \
             float(batch_size), name = "age_loss")
        
    cnn_MAE_age = tf.reduce_mean(tf.abs(cnn_predictions_age - tf.cast(outputs_age, tf.float32)))
    
    
    cnn_predictions_gender_correct = tf.equal(tf.cast(cnn_predictions_gender, dtype=tf.int32), outputs_gender)
    cnn_accuracy_gender = tf.reduce_mean(tf.cast(cnn_predictions_gender_correct, tf.float32))
    gender_loss = tf.losses.softmax_cross_entropy(tf.one_hot(outputs_gender,2, dtype=tf.float32), cnn_net_gender)
    
    total_loss = tf.add_n([gender_loss, age_loss])
    
    #cnn_predictions = tf.argmax(cnn_net)
    #cnn_loss = tf.losses.softmax_cross_entropy(tf.one_hot(outputs,num_class, dtype=tf.int32), cnn_net)
    #cnn_loss = tf.reduce_mean(cnn_loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        cnn_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

    return cnn_train, cnn_net_age, cnn_net_gender, total_loss, age_loss, gender_loss, cnn_MAE_age, cnn_accuracy_gender, cnn_predictions_age

def run_test(session, input_data, model_path = "train_models/age_gender_tensornets_wiki_cropface_model_20180519_123211"):
    cnn_predictions_age = 0
    cnn_predictions_gender = 0
    with tf.Session() as sess:
        #inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])        
        loader = tf.train.import_meta_graph(model_path + ".meta")
        
        #saver.restore(sess, tf.train.latest_checkpoint("train_models/"))
        #cnn_predictions_age, cnn_predictions_gender = age_gender_model(inputs)
        #graph = tf.get_default_graph()
        #saver_def = saver.as_saver_def()
        #print (saver_def.filename_tensor_name)
        #print (saver_def.restore_op_name)
        #sess.run(tf.global_variables_initializer())
        loader.restore(sess, tf.train.latest_checkpoint("train_models/"))
        inference_graph = tf.get_default_graph()
    
        cnn_predictions_age = inference_graph.get_tensor_by_name("mobilenet100/cnn_predictions_age:0")
        cnn_predictions_gender = inference_graph.get_tensor_by_name('mobilenet100/cnn_predictions_gender:0')
        
        feed = {'inputs_tensor:0' : input_data} 
        cnn_predictions_age_test, cnn_predictions_gender_test = sess.run([cnn_predictions_age, cnn_predictions_gender], feed_dict=feed)
        for i in range(len(cnn_predictions_age_test)):
            cnn_predictions_age_test[i] = int(cnn_predictions_age_test[i]) + 10
    return cnn_predictions_age_test, cnn_predictions_gender_test


def run_train(session, input_dict, num_class, epochs=3, batch_size=100,print_every=10, 
              learning_rate = 1e-5, dropout = 0.5, is_save_summary = True):
    
    is_training = True
    
    Xd = input_dict["X_train"]
    yd_age = input_dict["y_age_train"]
    yd_gender = input_dict["y_gender_train"]
    Xv = input_dict["X_val"]
    yv_age = input_dict["y_age_val"]
    yv_gender = input_dict["y_gender_val"]
    
    print("Batch dataset initialized.\n# of training data: {}\n# of val data: {}\n# of class: {}"
          .format(Xd.shape[0], Xv.shape[0], num_class))
    
    print("learning_rate: ", learning_rate)
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)
    
    with tf.Session() as sess:

        inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], name = "inputs_tensor")
        outputs_age = tf.placeholder(tf.int32, [None])
        outputs_gender = tf.placeholder(tf.int32, [None])
        
        cnn_train, cnn_net_age, cnn_net_gender, total_loss, age_loss, gender_loss, cnn_MAE_age, cnn_accuracy_gender, cnn_predictions_age = \
                            age_gender_model(inputs, outputs_age, outputs_gender, num_class, batch_size, learning_rate)
        
        
        train_summary = tf.summary.merge([tf.summary.scalar("total_loss", total_loss),
                          tf.summary.scalar("age_loss", age_loss),
                          tf.summary.scalar("gender_loss", gender_loss),
                          tf.summary.scalar("cnn_MAE_age", cnn_MAE_age),
                          tf.summary.scalar("cnn_accuracy_gender", cnn_accuracy_gender), 
                          tf.summary.scalar("cnn_predictions50", cnn_predictions_age[50]),
                          tf.summary.scalar("outputs_age", outputs_age[50]),
                          tf.summary.scalar("cnn_predictions_min", tf.reduce_min(cnn_predictions_age)),
                          tf.summary.image("inputs", tf.expand_dims(inputs[0], 0))])
                    
        test_summary = tf.summary.merge([tf.summary.scalar("val_total_loss", total_loss),
                          tf.summary.scalar("val_age_loss", age_loss),
                          tf.summary.scalar("val_gender_loss", gender_loss),
                          tf.summary.scalar("val_cnn_MAE_age", cnn_MAE_age),
                          tf.summary.scalar("val_cnn_accuracy_gender", cnn_accuracy_gender)])    
        
        merged = tf.summary.merge_all()
        
        sess.run(tf.global_variables_initializer())
        nets.pretrained(cnn_net_age)        
        
        # tensorboard setting

        time_now = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        if is_save_summary:
            fileName = os.path.normcase("./result/" + time_now)
            summary_writer = tf.summary.FileWriter(fileName, sess.graph)
        
        global_step = 0
                  
        #var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        #bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        #bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        #var_list += bn_moving_vars
        saver = tf.train.Saver(var_list = g_list, max_to_keep=2, keep_checkpoint_every_n_hours=2)

        for current_epoch in range(epochs):
            # training step
            ###for x_batch, y_batch in batch_set.batches():
            print("#############################Epoch Start##############################")
            
            for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
                start = time.time()
                start_idx = (i*batch_size)%Xd.shape[0]
                idx = np.int32(train_indicies[start_idx:start_idx+batch_size])
                                      
                #batch_Xd = load_img_from_tensor(Xd[idx,:, :, :], target_size=256, crop_size=224)
                batch_Xd = Xd[idx,:, :, :]
                
                """
                H = 224
                low = int(0.4/1.8 * H)
                high = int(1.4 / 1.8 * H)
                tempImgData = batch_Xd[:, low:high, low : high, :]
                for j in range(batch_Xd.shape[0]):
                    batch_Xd[j] = cv2.resize(tempImgData[j], (224, 224))
                """
                
                batch_Xd = nets.preprocess('mobilenet100', batch_Xd)
                batch_yd_age = yd_age[idx]
                batch_yd_gender = yd_gender[idx]
                feed = {inputs : batch_Xd, outputs_age : batch_yd_age, outputs_gender : batch_yd_gender}                
                
                global_step = global_step + 1
                
                _, total_loss_running, age_loss_running, gender_loss_running, cnn_MAE_age_running, \
                        cnn_accuracy_gender_running, summary_running = sess.run([cnn_train, total_loss, \
                        age_loss, gender_loss, cnn_MAE_age, cnn_accuracy_gender, train_summary], feed_dict=feed)
                
                #tem_data_fileName = os.path.normcase("./tmp_data/" + time_now + "/")
                #if i < 100 and current_epoch == 0:
                #    np.save("./tmp_data/batch_yd_batch_" + str(i) + ".npy",batch_yd)
                #    np.save("./tmp_data/cnn_predictions_batch_" + str(i) + ".npy",cnn_predictions_now)
                #    np.save("./tmp_data/cnn_softmax_batch_" + str(i) + ".npy",scores)
                
                if is_save_summary:
                    summary_writer.add_summary(summary_running, global_step)

                
                if global_step % print_every == 0:
                    print("{}/{} ({} epochs) step, total_loss:{:.4f}, aloss:{:.3f}, gloss:{:.3f}, gaccuracy:{:.3f}, ageMAE:{:.3f}, time/batch : {:.3f}sec"
                          .format(global_step, int(round(Xd.shape[0]/batch_size)) * epochs, current_epoch,     \
                               total_loss_running, age_loss_running, gender_loss_running,    \
                                  cnn_accuracy_gender_running, cnn_MAE_age_running, time.time() - start))
                    saver.save(sess, 'train_models/age_gender_tensornets_wiki_cropface_model_' + time_now)

            # test step
            start, avg_loss, avg_accuracy = time.time(), 0, 0
            
            #Xv = load_img_from_tensor(Xv, target_size=256, crop_size=224)
            #Xv = cnn_net.preprocess(Xv) 
            Xv = nets.preprocess('mobilenet100', Xv)
            feed = {inputs : Xv, outputs_age : yv_age, outputs_gender : yv_gender} 
            total_loss_running, age_loss_running, gender_loss_running, cnn_MAE_age_running, \
                    cnn_accuracy_gender_running, summary_running = sess.run([total_loss, \
                    age_loss, gender_loss, cnn_MAE_age, cnn_accuracy_gender, test_summary], feed_dict=feed)
            if is_save_summary:
                summary_writer.add_summary(summary_running, current_epoch)
            print("{} epochs test result, total_loss:{:.4f}, aloss:{:.3f}, gloss:{:.3f}, gaccuracy:{:.3f}, ageMAE:{:.3f}, time/batch : {:.3f}sec"
                      .format(current_epoch, total_loss_running, age_loss_running, gender_loss_running, cnn_accuracy_gender_running, \
                           cnn_MAE_age_running, time.time() - start))
            print("\n")
       
    return 