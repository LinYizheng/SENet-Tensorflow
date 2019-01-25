from __future__ import absolute_import,division,print_function
import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn import metrics
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials,rand
from hyperopt.mongoexp import MongoTrials
import h5py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import sobol
import datetime
import os
import math
import argparse
import sys
import time
from pyDOE import *
from sklearn.decomposition import PCA
import cv2
from sklearn.metrics import f1_score, accuracy_score

def readData(pairpathlabel):
    '''read image to list'''
    imgs = []
    labels = []
    filename1 = []
    for filepath, label in pairpathlabel:
        print(filepath, label)
        for (path,dirnames,filenames) in os.walk(filepath):
            #print(path)
            for filename in filenames:
                #print(filename)
                if filename.endswith('1.jpg') or filename.endswith('1.PNG'):
                    img_path=path+'/'+filename
                    img=cv2.imread(img_path,0)
                    img=cv2.resize(img,dsize=None,dst=None,fx=0.3,fy=0.3)
                    img= cv2.bilateralFilter(img,7,15,15)
                    img= cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(img)
                    #img=2*(img/255.0)-1.0
                    #img = img.astype('float32')
                    #img = (img - np.mean(img)) / np.std(img)
                    imgs.append(img)
                    labels.append(label)
                    filename1.append(filename)
    return imgs, np.array(labels),filename1
def onehot(numlist):
    ''' get one hot return host matrix is len * max+1 demensions'''
    b = np.zeros([len(numlist), max(numlist)+1])
    b[np.arange(len(numlist)), numlist] = 1
    return b.tolist()

def getfileandlabel(filedir):
    ''' get path and host paire and class index to name'''
    dictdir = dict([[name, os.path.join(filedir, name)] \
                    for name in np.sort(os.listdir(filedir)) 
                    if os.path.isdir(os.path.join(filedir, name)) and not name.startswith('.')])
                    #for (path, dirnames, _) in os.walk(filedir) for dirname in dirnames])
    dirnamelist, dirpathlist = dictdir.keys(), dictdir.values()
    indexlist = list(range(len(dirnamelist)))
    return list(zip(dirpathlist, (indexlist))), dict(zip(indexlist, dirnamelist))
    
pathlabelpair, indextoname = getfileandlabel('/home/hadoop/....')
imgsx, imgsy,fname = readData(pathlabelpair)
imgs=[]
labels=[]
size=120
for (img,label) in zip(imgsx, imgsy):
    h,w = img.shape[:2]  
    imgs.append(img[0:size,0:size])
    #imgs.append(cv2.flip(img[0:size,0:size],-1))  
    imgs.append(img[h-size:h,0:size]) 
    #imgs.append(cv2.flip(img[h-size:h,0:size],-1))
    imgs.append(img[h-size:h,w-size:w])
    #imgs.append(cv2.flip(img[h-size:h,w-size:w],-1))
    imgs.append(img[0:size,w-size:w])
    #imgs.append(cv2.flip(img[0:size,w-size:w],-1))
    hs=(h-size)//2
    ws=(w-size)//2
    imgs.append(img[hs:hs+size,ws:ws+size])
    #imgs.append(cv2.flip(img[hs:hs+size,ws:ws+size],-1))
    labels.append(label)
    labels.append(label)
    labels.append(label)
    labels.append(label)
    labels.append(label)

print(img.shape)    
print(np.array(imgs).shape,"+++",np.array(labels).shape)
X_train=np.array(imgs)
y_train=np.array(labels)
X_val=X_train
y_val=y_train

pathlabelpair, indextoname = getfileandlabel('/home/hadoop/....')
X_test,y_test,fname_test = readData(pathlabelpair)         
te_imgs_1=[]
te_labels_1=[]

te_imgs_2=[]
te_labels_2=[]

te_imgs_3=[]
te_labels_3=[]

te_imgs_4=[]
te_labels_4=[]

te_imgs_5=[]
te_labels_5=[]

for (img,label) in zip(X_test,y_test):  
    h,w = img.shape[:2]
    for i in range(1,6):                            
        if i==1:
            te_imgs_1.append(img[0:size,0:size])
            te_labels_1.append(label)

        elif i==2:
            te_imgs_2.append(img[h-size:h,0:size])
            te_labels_2.append(label)

        elif i==3:
            te_imgs_3.append(img[h-size:h,w-size:w])
            te_labels_3.append(label)

        elif i==4:
            te_imgs_4.append(img[0:size,w-size:w])
            te_labels_4.append(label)

        else :
            hs=(h-size)//2
            ws=(w-size)//2
            te_imgs_5.append(img[hs:hs+size,ws:ws+size])
            te_labels_5.append(label)

def getfileandlabel(filedir):
    ''' get path and host paire and class index to name'''
    dictdir = dict([[name, os.path.join(filedir, name)] \
                    for name in np.sort(os.listdir(filedir)) 
                    if os.path.isdir(os.path.join(filedir, name)) and not name.startswith('.')])
                    #for (path, dirnames, _) in os.walk(filedir) for dirname in dirnames])
    dirnamelist, dirpathlist = dictdir.keys(), dictdir.values()
    index=np.array([12])
    indexlist = list(index.repeat(len(dirnamelist)))
    return list(zip(dirpathlist, (indexlist))), dict(zip(indexlist, dirnamelist))
    
pathlabelpair_un, indextoname_un = getfileandlabel('/home/hadoop/...')

image13,labels13, fname13 = readData(pathlabelpair_un)
print(len(labels13))
for (img,label,fn) in zip(image13,labels13,fname13):  
    h,w = img.shape[:2]
    fname_test.append(fn) 
    for i in range(1,6):                            
        if i==1:
            te_imgs_1.append(img[0:size,0:size])
            te_labels_1.append(label)
        elif i==2:
            te_imgs_2.append(img[h-size:h,0:size])
        elif i==3:
            te_imgs_3.append(img[h-size:h,w-size:w])
        elif i==4:
            te_imgs_4.append(img[0:size,w-size:w])
        else :
            hs=(h-size)//2
            ws=(w-size)//2
            te_imgs_5.append(img[hs:hs+size,ws:ws+size])

               
te_imgs_1=np.array(te_imgs_1)
te_labels_1=np.array(te_labels_1)

te_imgs_2=np.array(te_imgs_2)
te_labels_2=np.array(te_labels_2)

te_imgs_3=np.array(te_imgs_3)
te_labels_3=np.array(te_labels_3)

te_imgs_4=np.array(te_imgs_4)
te_labels_4=np.array(te_labels_4)

te_imgs_5=np.array(te_imgs_5)
te_labels_5=np.array(te_labels_5)
print(np.array(te_imgs_1).shape,"+++",np.array(te_labels_1).shape)
print (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))

def trans2tfRecord(imgs,labels,output_dir): 
    filename = output_dir + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for i,[img,label] in enumerate(zip(imgs,labels)):
        img_raw = img.tostring()    
        example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))                        
                }))
        writer.write(example.SerializeToString()) 
    writer.close()    
    return filename   
######################################
def read_tfRecord(file_tfRecord):     
    queue = tf.train.string_input_producer([file_tfRecord])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
          'img_raw':tf.FixedLenFeature([], tf.string),   
          'label':tf.FixedLenFeature([], tf.int64)
                    }
            )
    image = tf.decode_raw(features['img_raw'],tf.uint8)
    image = tf.reshape(image,[120,120,1])
    image = tf.cast(image, tf.float32)
    #image = tf.image.per_image_standardization(image)
    label = tf.cast(features['label'], tf.int32)  
    return image,label


space = {
    'momentum': hp.uniform('momentum', 0.1,0.95),   
    'learning_rate': hp.uniform('learning_rate', 0.0001,0.01),
    'weight_decay': hp.uniform('weight_decay', 0.00025,0.01),
    'drop_out': hp.uniform('drop_out', 0.1,0.3),
    'batch_size' : hp.choice('batch_size', range(64,257))
       }
def SE_Inception_resnet_v2_1(params):
    import tensorflow as tf
    #from tflearn.layers.conv import global_avg_pool
    from tensorflow.contrib.layers import batch_norm, flatten
    from tensorflow.contrib.framework import arg_scope
    #from cifar10 import *
    import numpy as np
    import os
    import sys
    import time
    import pickle
    import random
   
    #os.environ['CUDA_VISIBLE_DEVICES']='0'
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    cpu_num=20
    config = tf.ConfigProto(device_count={"CPU": cpu_num},inter_op_parallelism_threads = cpu_num,
                intra_op_parallelism_threads = cpu_num,log_device_placement=True)
    
    class_num = 12
    image_size = 120
    img_channels = 1
    weight_decay = params['weight_decay']
    momentum =params['momentum']
    init_learning_rate = params['learning_rate']
    reduction_ratio = 16
    batch_size = params['batch_size']
    

    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
    tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
    tf.app.flags.DEFINE_integer("issync", 0, "1: sync,0:async")

    ps_hosts = ["xx.xxx.xxx.xx1:2222"]
    worker_hosts = ["xx.xxx.xxx.xx2:2224","xx.xxx.xxx.xx3:2224","xx.xxx.xxx.xx4:2224"]
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

    issync = FLAGS.issync
    if FLAGS.job_name == "ps":
        server.join()

    with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,ps_device="/job:ps/cpu:0",cluster=cluster)):
        
        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        def conv_layer(input, filter, kernel, stride=1, padding='SAME', layer_name="conv", activation=True):
            with tf.name_scope(layer_name):
                network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
                if activation :
                    network = Relu(network)
                return network

        def Fully_connected(x, units=class_num, layer_name='fully_connected') :
            with tf.name_scope(layer_name) :
                return tf.layers.dense(inputs=x, use_bias=True, units=units)

        def Relu(x):
            return tf.nn.relu(x)

        def Sigmoid(x):
            return tf.nn.sigmoid(x)

        def get_incoming_shape(incoming):
            """ Returns the incoming data shape """
            if isinstance(incoming, tf.Tensor):
                return incoming.get_shape().as_list()
            elif type(incoming) in [np.array, np.ndarray, list, tuple]:
                return np.shape(incoming)
            else:
                raise Exception("Invalid incoming layer.")

        def global_avg_pool(incoming, name="GlobalAvgPool"):
            input_shape = get_incoming_shape(incoming)
            assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"
            with tf.name_scope(name):
                inference = tf.reduce_mean(incoming, [1, 2])
            return inference

        def Global_Average_Pooling(x):
            return global_avg_pool(x, name='Global_avg_pooling')

        def Max_pooling(x, pool_size=[3,3], stride=2, padding='VALID') :
            return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

        def Batch_Normalization(x, training, scope):
            with arg_scope([batch_norm],
                           scope=scope,
                           updates_collections=None,
                           decay=0.9,
                           center=True,
                           scale=True,
                           zero_debias_moving_mean=True) :
                return tf.cond(training,
                               lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                               lambda : batch_norm(inputs=x, is_training=training, reuse=True))

        def Concatenation(layers) :
            return tf.concat(layers, axis=3)

        def Dropout(x, rate, training) :
            return tf.layers.dropout(inputs=x, rate=rate, training=training)

        def Evaluate(sess,X_val,y_val):
            val_acc = 0.0
            val_loss = 0.0

            test_feed_dict = {
                x: X_val,
                label: y_val,
                learning_rate: epoch_learning_rate,
                training_flag: False
            }

            loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

            val_loss += loss_
            val_acc += acc_
            return  val_loss,val_acc

        class SE_Inception_resnet_v2():
            def __init__(self, x, training):
                self.training = training
                self.model = self.Build_SEnet(x)

            def Stem(self, x, scope):
                with tf.name_scope(scope) :
                    x = conv_layer(x, filter=32, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_conv1')
                    x = conv_layer(x, filter=32, kernel=[3,3], padding='VALID', layer_name=scope+'_conv2')
                    block_1 = conv_layer(x, filter=64, kernel=[3,3], layer_name=scope+'_conv3')

                    split_max_x = Max_pooling(block_1)
                    split_conv_x = conv_layer(block_1, filter=96, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv1')
                    x = Concatenation([split_max_x,split_conv_x])

                    split_conv_x1 = conv_layer(x, filter=64, kernel=[1,1], layer_name=scope+'_split_conv2')
                    split_conv_x1 = conv_layer(split_conv_x1, filter=96, kernel=[3,3], padding='VALID', layer_name=scope+'_split_conv3')

                    split_conv_x2 = conv_layer(x, filter=64, kernel=[1,1], layer_name=scope+'_split_conv4')
                    split_conv_x2 = conv_layer(split_conv_x2, filter=64, kernel=[7,1], layer_name=scope+'_split_conv5')
                    split_conv_x2 = conv_layer(split_conv_x2, filter=64, kernel=[1,7], layer_name=scope+'_split_conv6')
                    split_conv_x2 = conv_layer(split_conv_x2, filter=96, kernel=[3,3], padding='VALID', layer_name=scope+'_split_conv7')

                    x = Concatenation([split_conv_x1,split_conv_x2])

                    split_conv_x = conv_layer(x, filter=192, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv8')
                    split_max_x = Max_pooling(x)

                    x = Concatenation([split_conv_x, split_max_x])

                    x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
                    x = Relu(x)

                    return x

            def Inception_resnet_A(self, x, scope):
                with tf.name_scope(scope) :
                    init = x

                    split_conv_x1 = conv_layer(x, filter=32, kernel=[1,1], layer_name=scope+'_split_conv1')

                    split_conv_x2 = conv_layer(x, filter=32, kernel=[1,1], layer_name=scope+'_split_conv2')
                    split_conv_x2 = conv_layer(split_conv_x2, filter=32, kernel=[3,3], layer_name=scope+'_split_conv3')

                    split_conv_x3 = conv_layer(x, filter=32, kernel=[1,1], layer_name=scope+'_split_conv4')
                    split_conv_x3 = conv_layer(split_conv_x3, filter=48, kernel=[3,3], layer_name=scope+'_split_conv5')
                    split_conv_x3 = conv_layer(split_conv_x3, filter=64, kernel=[3,3], layer_name=scope+'_split_conv6')

                    x = Concatenation([split_conv_x1,split_conv_x2,split_conv_x3])
                    x = conv_layer(x, filter=384, kernel=[1,1], layer_name=scope+'_final_conv1', activation=False)

                    x = x*0.1
                    x = init + x

                    x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
                    x = Relu(x)

                    return x

            def Inception_resnet_B(self, x, scope):
                with tf.name_scope(scope) :
                    init = x

                    split_conv_x1 = conv_layer(x, filter=192, kernel=[1,1], layer_name=scope+'_split_conv1')

                    split_conv_x2 = conv_layer(x, filter=128, kernel=[1,1], layer_name=scope+'_split_conv2')
                    split_conv_x2 = conv_layer(split_conv_x2, filter=160, kernel=[1,7], layer_name=scope+'_split_conv3')
                    split_conv_x2 = conv_layer(split_conv_x2, filter=192, kernel=[7,1], layer_name=scope+'_split_conv4')

                    x = Concatenation([split_conv_x1, split_conv_x2])
                    x = conv_layer(x, filter=1152, kernel=[1,1], layer_name=scope+'_final_conv1', activation=False)
                    # 1154
                    x = x * 0.1
                    x = init + x

                    x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
                    x = Relu(x)

                    return x

            def Inception_resnet_C(self, x, scope):
                with tf.name_scope(scope) :
                    init = x

                    split_conv_x1 = conv_layer(x, filter=192, kernel=[1,1], layer_name=scope+'_split_conv1')

                    split_conv_x2 = conv_layer(x, filter=192, kernel=[1, 1], layer_name=scope + '_split_conv2')
                    split_conv_x2 = conv_layer(split_conv_x2, filter=224, kernel=[1, 3], layer_name=scope + '_split_conv3')
                    split_conv_x2 = conv_layer(split_conv_x2, filter=256, kernel=[3, 1], layer_name=scope + '_split_conv4')

                    x = Concatenation([split_conv_x1,split_conv_x2])
                    x = conv_layer(x, filter=2144, kernel=[1,1], layer_name=scope+'_final_conv2', activation=False)
                    # 2048
                    x = x * 0.1
                    x = init + x

                    x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
                    x = Relu(x)

                    return x

            def Reduction_A(self, x, scope):
                with tf.name_scope(scope) :
                    k = 256
                    l = 256
                    m = 384
                    n = 384

                    split_max_x = Max_pooling(x)

                    split_conv_x1 = conv_layer(x, filter=n, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv1')

                    split_conv_x2 = conv_layer(x, filter=k, kernel=[1,1], layer_name=scope+'_split_conv2')
                    split_conv_x2 = conv_layer(split_conv_x2, filter=l, kernel=[3,3], layer_name=scope+'_split_conv3')
                    split_conv_x2 = conv_layer(split_conv_x2, filter=m, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv4')

                    x = Concatenation([split_max_x, split_conv_x1, split_conv_x2])

                    x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
                    x = Relu(x)

                    return x

            def Reduction_B(self, x, scope):
                with tf.name_scope(scope) :
                    split_max_x = Max_pooling(x)

                    split_conv_x1 = conv_layer(x, filter=256, kernel=[1,1], layer_name=scope+'_split_conv1')
                    split_conv_x1 = conv_layer(split_conv_x1, filter=384, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv2')

                    split_conv_x2 = conv_layer(x, filter=256, kernel=[1,1], layer_name=scope+'_split_conv3')
                    split_conv_x2 = conv_layer(split_conv_x2, filter=288, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv4')

                    split_conv_x3 = conv_layer(x, filter=256, kernel=[1,1], layer_name=scope+'_split_conv5')
                    split_conv_x3 = conv_layer(split_conv_x3, filter=288, kernel=[3,3], layer_name=scope+'_split_conv6')
                    split_conv_x3 = conv_layer(split_conv_x3, filter=320, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv7')

                    x = Concatenation([split_max_x, split_conv_x1, split_conv_x2, split_conv_x3])

                    x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
                    x = Relu(x)

                    return x

            def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
                with tf.name_scope(layer_name) :

                    squeeze = Global_Average_Pooling(input_x)

                    excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
                    excitation = Relu(excitation)
                    excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
                    excitation = Sigmoid(excitation)

                    excitation = tf.reshape(excitation, [-1,1,1,out_dim])
                    scale = input_x * excitation

                    return scale

            def Build_SEnet(self, input_x):
                input_x = tf.pad(input_x, [[0, 0], [32, 32], [32, 32], [0, 0]])
                # size 32 -> 96
                print(np.shape(input_x))
                # only cifar10 architecture

                x = self.Stem(input_x, scope='stem')

                for i in range(5) :
                    x = self.Inception_resnet_A(x, scope='Inception_A'+str(i))
                    channel = int(np.shape(x)[-1])
                    x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_A'+str(i))

                x = self.Reduction_A(x, scope='Reduction_A')

                channel = int(np.shape(x)[-1])
                x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_A')

                for i in range(10)  :
                    x = self.Inception_resnet_B(x, scope='Inception_B'+str(i))
                    channel = int(np.shape(x)[-1])
                    x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_B'+str(i))

                x = self.Reduction_B(x, scope='Reduction_B')

                channel = int(np.shape(x)[-1])
                x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_B')

                for i in range(5) :
                    x = self.Inception_resnet_C(x, scope='Inception_C'+str(i))
                    channel = int(np.shape(x)[-1])
                    x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_C'+str(i))

                # channel = int(np.shape(x)[-1])
                # x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_C')

                x = Global_Average_Pooling(x)
                x = Dropout(x, rate=params['drop_out'], training=self.training)
                x = flatten(x)

                x = Fully_connected(x, layer_name='final_fully_connected')
                return x



        #tf.reset_default_graph()

        x = tf.placeholder(tf.float32, shape=[None, image_size, image_size],name='image')
        x1 = tf.reshape(x, [-1,120,120,1])
        label = tf.placeholder(tf.int32, shape=[None],name='label')
        training_flag = tf.placeholder(tf.bool,name='training_flag')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        logits = SE_Inception_resnet_v2(x1, training=training_flag).model
        cost = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=label, logits=logits))

        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #train = optimizer.minimize(cost + l2_loss * weight_decay)
        if issync == 1:
            rep_op = tf.train.SyncReplicasOptimizer(optimizer,
                                                replicas_to_aggregate=len(worker_hosts),
                                                replica_id=FLAGS.task_index,
                                                total_num_replicas=len(worker_hosts))
            train = rep_op.optimizer.minimize(cost + l2_loss * weight_decay,global_step=global_step)
            init_token_op = rep_op.get_init_tokens_op()
            chief_queue_runner = rep_op.get_chief_queue_runner()
        else:
            train = optimizer.minimize(cost + l2_loss * weight_decay,global_step=global_step)
            
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(y_train.shape[0] * min_fraction_of_examples_in_queue)
    input_queue=tf.train.slice_input_producer([X_train,y_train],shuffle=True)
    Xtrain_batch,ytrain_batch = tf.train.batch(input_queue,batch_size=params['batch_size'],num_threads=100,capacity=min_queue_examples + 3 * params['batch_size'])  
    
    print (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    

    print(params)
    
    logs='logs_unknown_p6_resort_fab8_SE_inception_resnet_gpu.txt'
    
    line = "%s, %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()), params)
    with open(logs, 'a') as f:
        f.write(line)    
    epochs = 200
    train_num_examples=y_train.shape[0]
    max_steps = int(math.ceil(train_num_examples / batch_size))
    total_sample_train = max_steps * batch_size
    best = 0
    min_val_loss = 2
    wait = 0  #counter for patience
    best_rounds =1
    counter=0
    patience=20
    if not os.path.exists('./train_logs/'):
        os.makedirs('./train_logs/')

    saver=tf.train.Saver()
    init_op=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),logdir="./train_logs/",init_op=init_op,global_step=global_step,saver=saver)
    
    if FLAGS.task_index == 0:
        print('Worker %d: Initailizing session...' % FLAGS.task_index)
    else:
        print('Worker %d: Waiting for session to be initaialized...' % FLAGS.task_index)
    
    sess = sv.prepare_or_wait_for_session(server.target,config = config)
    print('Worker %d: Session initialization  complete.' % FLAGS.task_index)

    if FLAGS.task_index == 0 and issync == 1:
        sv.start_queue_runners(sess, [chief_queue_runner])
        sess.run(init_token_op)
        
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess =sess,coord=coord)
    '''try: 
        saver
    except NameError:
        print ('saver_exists=False')
    else:
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('./model_cpu_bak/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)   '''  
   
    epoch_learning_rate = init_learning_rate
    for epoch in xrange(1,epochs+1):
        if epoch % 30 == 0 :
            epoch_learning_rate = epoch_learning_rate / 10
        true_count_train = 0
        pre_index = 0
        train_acc = 0.0
        train_loss = 0.0
        for step in xrange(1,max_steps+1):
            try:
                image_batch, label_batch = sess.run([Xtrain_batch,ytrain_batch])
                train_feed_dict = {x: image_batch,label: label_batch,learning_rate: epoch_learning_rate,
                                    training_flag: True}
                _, batch_loss,batch_logits = sess.run([train, cost,logits], feed_dict=train_feed_dict)
                train_loss += batch_loss
                train_acc += accuracy_score(np.argmax(batch_logits, axis=1),label_batch)
                pre_index += batch_size
                    
            except Exception as E_results:
                print("Exception:",E_results)
                step=step-1
                break              

        if step>0:
            train_loss /= step # average loss
            train_acc /= step # average accuracy
            
        val_acc = train_acc
        val_loss = train_loss        
        '''val_acc = 0.0
        val_loss = 0.0        
        X_val_nums=X_val.shape[0]
        for i in xrange(X_val_nums):
            val_loss_i, val_acc_i= Evaluate(sess,X_val[i].reshape([-1,image_size,image_size,1]),y_val[i].reshape([1,]))
            val_loss += val_loss_i/X_val_nums
            val_acc += val_acc_i/X_val_nums'''

        counter +=1
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best = val_acc
            best_rounds=counter
            wait = 0
            if min_val_loss<0.5:
                if not os.path.exists('./model6'):
                    os.makedirs('./model6')
                saver.save(sess,save_path='./model6/unknown_p_resort_fab8_SE_Inception_resnet_v2_gpu5.ckpt')
        else:
            wait += 1 #incremental the number of times without improvement

        line = "epoch: %d/%d, train_loss: %f, train_acc: %f, val_loss: %f, val_acc: %f \n" % (
            epoch, epochs, train_loss, train_acc, val_loss, val_acc)
        with open(logs, 'a') as f:
            f.write(line)

        if wait >= patience : #no more patience, retrieve best model
            break

    print('epoch--:%d,best_epoch: %d, min_val_acc: %f ,val_loss: %f, train_acc: %f' % 
          (epoch,best_rounds,best,val_loss, train_acc))

    try: 
        saver
    except NameError:
        print ('saver_exists=False')
    else:
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('./model6/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

    print('####################################')
    print (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    te_true_count=0
    te_preds=[]
    with sess.as_default():
        for i in xrange(te_labels_1.shape[0]-1):
            te_1_logits=logits.eval(feed_dict={x: te_imgs_1[i].reshape([-1,image_size,image_size,1]),training_flag: False})

            te_2_logits=logits.eval(feed_dict={x: te_imgs_2[i].reshape([-1,image_size,image_size,1]),training_flag: False})

            te_3_logits=logits.eval(feed_dict={x: te_imgs_3[i].reshape([-1,image_size,image_size,1]),training_flag: False})

            te_4_logits=logits.eval(feed_dict={x: te_imgs_4[i].reshape([-1,image_size,image_size,1]),training_flag: False})

            te_5_logits=logits.eval(feed_dict={x: te_imgs_5[i].reshape([-1,image_size,image_size,1]),training_flag: False})

            te_logits =(te_1_logits+te_2_logits+te_3_logits+te_4_logits+te_5_logits)/5
            te_pred = np.argmax(te_logits)
            #print(te_ture)
            te_preds.append(te_pred)
    te_acc = accuracy_score(te_preds,te_labels_1[:(te_labels_1.shape[0]-1)])
    crosst=pd.crosstab(np.array(te_preds).reshape([-1,]),te_labels_1[:(te_labels_1.shape[0]-1)], margins=True)
    print(crosst)
    print(te_acc)
    coord.request_stop()
    coord.join(threads)
    sess.close()        
      
    line = "epoch: %d/%d, best_val_so_far: %f, train_acc: %f, val_acc: %f, test_acc: %f \n %s \n" % (
        epoch, epochs, best, train_acc, val_acc, te_acc, crosst)
    with open(logs, 'a') as f:
        f.write(line)

    return {'loss': te_acc*(-1),'test_accuracy':te_acc,'min_val_accuracy':best,
            'train_accuracy':train_acc,'best_rounds':best_rounds,'status': STATUS_OK}
    print (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    
print (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
max_evals=40
trials = Trials()
best =  fmin(SE_Inception_resnet_v2_1, space, algo=tpe.suggest, trials=trials, verbose=0, max_evals=max_evals)
print ("best:",best)
print (trials.best_trial)
print (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
