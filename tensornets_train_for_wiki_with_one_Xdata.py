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

def general_model(inputs, outputs, num_class, learning_rate = 1e-5, is_training = False):
    cnn_net = nets.MobileNet100(inputs, is_training = is_training, classes = num_class)
       
    cnn_loss = tf.losses.softmax_cross_entropy(tf.one_hot(outputs,num_class, dtype=tf.int32), cnn_net)
    cnn_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cnn_loss)

    cnn_predictions = tf.argmax(cnn_net, axis = 1)
        
    cnn_correct_prediction = tf.equal(tf.cast(cnn_predictions, dtype=tf.int32), outputs)
    cnn_accuracy = tf.reduce_mean(tf.cast(cnn_correct_prediction, tf.float32))
    return cnn_net, cnn_train, cnn_loss, cnn_accuracy, cnn_predictions
    
def age_model(inputs, outputs, num_class, learning_rate = 1e-5, is_training = False):

    cnn_net = nets.MobileNet100(inputs, is_training = is_training, classes = num_class)
    
    """
    cnn_softmax = tf.nn.softmax(cnn_net)
    #print("0, ", tf.shape(cnn_softmax).as_list())
    #ages = np.tile(np.arange(10, 90),(tf.shape(inputs)[0], 1))
    #print("0s", tf.shape(ages))    
    cnn_predictions = tf.reduce_sum(tf.multiply(tf.range(0, 80, dtype = np.float32), cnn_softmax), axis = 1)
    cnn_loss = tf.sqrt(tf.reduce_sum((cnn_predictions - tf.cast(outputs, tf.float32)) ** 2))
    """
    """
    min_pred = tf.reduce_min(cnn_predictions)
    max_pred = tf.reduce_max(cnn_predictions)
    min_label = tf.reduce_min(tf.cast(outputs, tf.float32))
    max_label = tf.reduce_max(tf.cast(outputs, tf.float32))
    cnn_predictions = max_label * (cnn_predictions - min_pred) / max_pred  + min_label
    """
    
    cnn_predictions = tf.reduce_sum(tf.multiply(tf.range(0, 80, dtype = np.float32), cnn_net), axis = 1)
    cnn_loss = tf.sqrt(tf.reduce_sum((cnn_predictions - tf.cast(outputs, tf.float32)) ** 2))
    
    #cnn_predictions = tf.argmax(cnn_net)
    #cnn_loss = tf.losses.softmax_cross_entropy(tf.one_hot(outputs,num_class, dtype=tf.int32), cnn_net)
    #cnn_loss = tf.reduce_mean(cnn_loss)
    cnn_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cnn_loss)

    cnn_accuracy = tf.reduce_mean(tf.abs(cnn_predictions - tf.cast(outputs, tf.float32)))
    return cnn_net, cnn_train, cnn_loss, cnn_accuracy, cnn_predictions

@var_scope('mobilenet100')
def age_gender_model(inputs, outputs_age, outputs_gender, num_class, batch_size, learning_rate = 1e-5, is_training = False):

    cnn_net = nets.MobileNet100(inputs, is_training = is_training, classes = num_class, stem=True)
    
    cnn_net_reduce = reduce_mean(cnn_net, [1, 2], name='avgpool')
    cnn_net_dropout = dropout(cnn_net_reduce, keep_prob=0.999, is_training=is_training, scope='dropout')
    
    cnn_net_age = fc(cnn_net_dropout, num_class, scope='logits_age')
    cnn_net_gender = fc(cnn_net_dropout, 2, scope='logits_gender')
    
    cnn_net_age = softmax(cnn_net_age, name='probs')
    cnn_net_gender = softmax(cnn_net_gender, name='probs')

    #print(inputs.get_shape().aslist()[0])  本来咯应该用这个代替batch_size的，但是这个=None，
    cnn_predictions_age = tf.reduce_sum(tf.multiply(tf.range(0, 80, dtype = np.float32), cnn_net_age), axis = 1)
    age_loss = tf.divide(tf.sqrt(tf.reduce_sum((cnn_predictions_age - tf.cast(outputs_age, tf.float32)) ** 2), name = "sqrt_age_loss"), \
             float(batch_size), name = "age_loss")
        
    cnn_MAE_age = tf.reduce_mean(tf.abs(cnn_predictions_age - tf.cast(outputs_age, tf.float32)))
    
    cnn_predictions_gender = tf.argmax(cnn_net_gender, axis = 1)
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

    return cnn_train, cnn_net, cnn_net_age, cnn_net_gender, total_loss, age_loss, gender_loss, cnn_MAE_age, cnn_accuracy_gender, cnn_predictions_age


def run_model(session, useing_model, input_dict, num_class, epochs=3, batch_size=100,print_every=10, 
              learning_rate = 1e-5, dropout = 0.5, is_save_summary = True, is_training = False, name = "for_wiki"):
    
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

        inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        outputs_age = tf.placeholder(tf.int32, [None])
        outputs_gender = tf.placeholder(tf.int32, [None])
        
        cnn_train, cnn_net, cnn_net_age, cnn_net_gender, total_loss, age_loss, gender_loss, cnn_MAE_age, cnn_accuracy_gender, cnn_predictions_age = \
                            useing_model(inputs, outputs_age, outputs_gender, num_class, batch_size, learning_rate, is_training)
        
        
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
                  
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        saver = tf.train.Saver(var_list=var_list, max_to_keep=2, keep_checkpoint_every_n_hours=2)

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
                    saver.save(sess, 'train_models/age_gender_tensornets_wiki_cropface_model')

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
                      .format(current_epoch, total_loss_running, age_loss_running, gender_loss_running, cnn_MAE_age_running, \
                           cnn_accuracy_gender_running, time.time() - start))
            print("\n")
       
    return 


tf.app.flags.DEFINE_integer('batch_size', 500, 'batch size')
tf.app.flags.DEFINE_integer('num_train_data', 58000, 'number of train data')
tf.app.flags.DEFINE_integer('num_val_data', 2000, 'number of val data')
tf.app.flags.DEFINE_integer('num_epochs', 4, 'number of epochs')
tf.app.flags.DEFINE_float('learning_rate', 5e-3, 'init learning rate')

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
        run_model(sess, age_gender_model, age_gender_dict,num_class = 80, epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size,
                  print_every=FLAGS.print_every, learning_rate = FLAGS.learning_rate, 
                  is_save_summary = FLAGS.is_save_summary, is_training = True)
        pass
    print("==================================================================")
    print("\n")
    
if __name__ == '__main__':
    tf.app.run()