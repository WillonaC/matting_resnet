#%%
import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import ndimage
import h5py
import scipy.io as sio
from gen_gradient_data import sobel_demo

#%%
read_data = False

patch_size = 64
channel_num = 4
batch_num=1
train_num=1700
epoch_num=10000000
img_height=800
img_width=600

train_dir = "../../matting_deep/train_data2/"
tfrecords_dir = "../data/"
saveName="_DirTri"
TFrecords_name="data_train"+saveName+".tfrecords"
model_dir="../model/"

#%%
def generate_trimap(trimap,alpha):
#    trimap_kernel = [val for val in range(20,40)]
#    k_size = random.choice(trimap_kernel)
    k_size = 10
    trimap[np.where((ndimage.grey_dilation(alpha[:,:,0],size=(k_size,k_size)) - ndimage.grey_erosion(alpha[:,:,0],size=(k_size*2,k_size*2)))!=0)] = 128
    #trimap[np.where((ndimage.grey_dilation(alpha[:,:,0],size=(k_size,k_size)) - alpha[:,:,0]!=0))] = 128
    return trimap

# 读数据
def read_data_from_file(data_dir,fname):
    rgb_dir = os.path.join(data_dir,"eps/0/")
    alpha_dir = os.path.join(data_dir,"alpha/0/")
    direction_dir = os.path.join(data_dir,"direction/0/")
    
    fpath_rgb = os.path.join(rgb_dir, fname)
    fpath_alpha = os.path.join(alpha_dir, fname)
    fpath_direction = os.path.join(direction_dir, fname[:-4]+'.mat')

    image_rgb = Image.open(fpath_rgb)
    image_alpha = Image.open(fpath_alpha)  
    image_direction = h5py.File(fpath_direction,'r')
    
    image_direction_keys = list(image_direction.keys())    
    data_direction = np.transpose(image_direction[image_direction_keys[0]].value,(2,1,0))     
    
    data_rgb = np.array(image_rgb)
    data_alpha = np.array(image_alpha)
    data_alpha = np.expand_dims(data_alpha,2)
    data_trimap = np.copy(data_alpha)
    data_trimap = generate_trimap(data_trimap,data_alpha)
    
#    plt.figure()
#    plt.subplot(1,3,1)
#    plt.imshow(np.concatenate((data_alpha/255,data_alpha/255,data_alpha/255),axis=2))
#    plt.subplot(1,3,2)
#    plt.imshow(data_direction)
#    plt.subplot(1,3,3)
#    plt.imshow(np.concatenate((data_trimap/255,data_trimap/255,data_trimap/255),axis=2))
#    plt.show()

    data = np.concatenate([data_direction,data_trimap],axis=2)
    label =  data_alpha
    
    return data,label,data_rgb
    
# 读取本地数据存成TFRecord文件
def read_data_to_tfrecords(data_dir):
    rgb_dir = os.path.join(data_dir,"alpha/0/")

    writer= tf.python_io.TFRecordWriter(os.path.join(tfrecords_dir, TFrecords_name)) #要生成的文件
    i=0
    for fname in os.listdir(rgb_dir): 
        i+=1
        if i>600:
            break
        if i%10==0:
            print(fname)
            
        data,label=read_data_from_file(data_dir,fname)
        
        data_raw=data.tobytes()
        label_raw=label.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
            'data_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_raw]))
        })) #example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  #序列化为字符串
    writer.close()
    
if read_data:
    read_data_to_tfrecords(train_dir)
#%%
#x=mnist.train.images
#y=mnist.train.labels
#X=mnist.test.images
#Y=mnist.test.labels
xs = tf.placeholder(tf.float32, [None, patch_size, patch_size, channel_num],name="xs")   
ys = tf.placeholder(tf.float32, [None, patch_size, patch_size, 1],name="ys")
tri = tf.placeholder(tf.float32, [None, patch_size, patch_size, 1],name="tri")
#xs = tf.placeholder(tf.float32, [None, img_height, img_width, channel_num],name="xs")   
#ys = tf.placeholder(tf.float32, [None, img_height, img_width, 1],name="ys")
#tri = tf.placeholder(tf.float32, [None, img_height, img_width, 1],name="tri")
sess = tf.InteractiveSession()

def weight_variable(shape):
#这里是构建初始变量
  initial = tf.truncated_normal(shape, mean=0,stddev=0.1)
#创建变量
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#在这里定义残差网络的id_block块，此时输入和输出维度相同
def identity_block(X_input, kernel_size, in_filter, out_filters, stage, block):
        """
        Implementation of the identity block as defined in Figure 3

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test

        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        block_name = 'res' + str(stage) + block
        f1, f2, f3 = out_filters
        with tf.variable_scope(block_name):
            X_shortcut = X_input

            #first
            W_conv1 = weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
            b_conv1 = bias_variable([f1])
            X = tf.nn.relu(X+ b_conv1)

            #second
            W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            b_conv2 = bias_variable([f2])
            X = tf.nn.relu(X+ b_conv2)

            #third

            W_conv3 = weight_variable([1, 1, f2, f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
            b_conv3 = bias_variable([f3])
            X = tf.nn.relu(X+ b_conv3)
            #final step
            add = tf.add(X, X_shortcut)
            b_conv_fin = bias_variable([f3])
            add_result = tf.nn.relu(add+b_conv_fin)

#        return add_result
        return add


#这里定义conv_block模块，由于该模块定义时输入和输出尺度不同，故需要进行卷积操作来改变尺度，从而得以相加
def convolutional_block( X_input, kernel_size, in_filter,
                            out_filters, stage, block, stride=2):
        """
        Implementation of the convolutional block as defined in Figure 4

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test
        stride -- Integer, specifying the stride to be used

        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        block_name = 'res' + str(stage) + block
        with tf.variable_scope(block_name):
            f1, f2, f3 = out_filters

            x_shortcut = X_input
            #first
            W_conv1 = weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1,strides=[1, stride, stride, 1],padding='SAME')
            b_conv1 = bias_variable([f1])
            X = tf.nn.relu(X + b_conv1)

            #second
            W_conv2 =weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1,1,1,1], padding='SAME')
            b_conv2 = bias_variable([f2])
            X = tf.nn.relu(X+b_conv2)

            #third
            W_conv3 = weight_variable([1,1, f2,f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1,1], padding='SAME')
            b_conv3 = bias_variable([f3])
            X = tf.nn.relu(X+b_conv3)
            #shortcut path
            W_shortcut =weight_variable([1, 1, in_filter, f3])
            x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')

            #final
            add = tf.add(x_shortcut, X)
            #建立最后融合的权重
            b_conv_fin = bias_variable([f3])
            add_result = tf.nn.relu(add+ b_conv_fin)


#        return add_result
        return add

#%%
x1 = xs
w_conv1 = weight_variable([2, 2, channel_num, 64])
#x1 = tf.nn.conv2d(x1, w_conv1, strides=[1, 2, 2, 1], padding='SAME')
x1 = tf.nn.conv2d(x1, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
b_conv1 = bias_variable([64])
x1 = tf.nn.relu(x1+b_conv1)
#这里操作后变成14x14x64
x1 = tf.nn.max_pool(x1, ksize=[1, 3, 3, 1],
                           strides=[1, 1, 1, 1], padding='SAME')


#stage 2
x2 = convolutional_block(X_input=x1, kernel_size=3, in_filter=64,  out_filters=[64, 64, 256], stage=2, block='a', stride=1)
#上述conv_block操作后，尺寸变为14x14x256
x2 = identity_block(x2, 3, 256, [64, 64, 256], stage=2, block='b' )
x2 = identity_block(x2, 3, 256, [64, 64, 256], stage=2, block='c')
#上述操作后张量尺寸变成14x14x256
#x2 = tf.nn.max_pool(x2, [1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')
x2 = tf.nn.max_pool(x2, [1, 2, 2, 1], strides=[1,1,1,1], padding='SAME')
#变成7x7x256
#flat = tf.reshape(x2, [-1,7*7*256])
flat = tf.reshape(x2, [-1,patch_size,patch_size,256])
#flat = tf.reshape(x2, [-1,img_height,img_width,256])

#w_fc1 = weight_variable([7 * 7 *256, 1024])
#b_fc1 = bias_variable([1024])
w_fc1 = weight_variable([1, 1, 256, 64])
b_fc1 = bias_variable([64])

#h_fc1 = tf.nn.relu(tf.matmul(flat, w_fc1) + b_fc1)
h_fc1 = tf.nn.relu(tf.nn.conv2d(flat, w_fc1, strides=[1, 1, 1, 1], padding='SAME') + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#w_fc2 = weight_variable([1024, 10])
#b_fc2 = bias_variable([10])
w_fc2 = weight_variable([1, 1, 64, 1])
b_fc2 = bias_variable([1])
y_conv = tf.nn.conv2d(h_fc1_drop, w_fc2, strides=[1, 1, 1, 1], padding='SAME') + b_fc2

b_mask = tf.reshape(xs[:,:,:,3],[-1,patch_size,patch_size,1])
#b_mask = tf.reshape(xs[:,:,:,3],[-1,img_height,img_width,1])
y_conv = tf.where(tf.equal(tri,128),y_conv,tri/255)

#建立损失函数，在这里采用交叉熵函数
wl = tf.where(tf.equal(tri,128),tf.fill([batch_num,patch_size,patch_size,1],1.),tf.fill([batch_num,patch_size,patch_size,1],0.))
#wl = tf.where(tf.equal(tri,128),tf.fill([batch_num,img_height,img_width,1],1.),tf.fill([batch_num,img_height,img_width,1],0.))
unknown_region_size = tf.reduce_sum(wl)
alpha_diff = tf.sqrt(tf.square(ys - y_conv)+ 1e-12)
cross_entropy = tf.reduce_sum(alpha_diff * wl)/(unknown_region_size)
#cross_entropy = tf.reduce_mean(
#    tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=y_conv))
#    tf.squared_difference(ys, y_conv))

tf.summary.scalar("cross_entropy", cross_entropy)

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(ys,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#%%
#初始化变量
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=4)
merged = tf.summary.merge_all() #将图形、训练过程等数据合并在一起
writer = tf.summary.FileWriter('../logs',sess.graph) #将训练日志写入到logs文件夹下

print("strat...")
#feature = {'label_raw': tf.FixedLenFeature([], tf.string),
#           'data_raw': tf.FixedLenFeature([], tf.string)}
## define a queue base on input filenames
#filename_queue = tf.train.string_input_producer([os.path.join(tfrecords_dir, TFrecords_name)], num_epochs=1)
## define a tfrecords file reader
#reader = tf.TFRecordReader()
## read in serialized example data
#_, serialized_example = reader.read(filename_queue)
## decode example by feature
#features = tf.parse_single_example(serialized_example, features=feature)
#        
#img = tf.decode_raw(features['data_raw'], tf.float64)
#label = tf.decode_raw(features['label_raw'], tf.float64)
#
## restore image to [height, width, channel]
#img = tf.reshape(img, [img_height, img_width, channel_num])
#img = tf.cast(img, tf.float64) #在流中抛出img张量
#label = tf.reshape(label, [img_height, img_width,1])
#label = tf.cast(label, tf.float64) #在流中抛出label张量
#                
## create bathch
#images, labels = tf.train.shuffle_batch(
#        [img, label], batch_size=batch_num, capacity=30, 
#        num_threads=2, min_after_dequeue=10) 
## capacity是队列的最大容量，num_threads是dequeue后最小的队列大小，num_threads是进行队列操作的线程数。
#
#sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
#coord = tf.train.Coordinator()
#threads = tf.train.start_queue_runners(coord=coord)
        
def UR_center(trimap):

    target = np.where(trimap==128)
    index = random.choice([i for i in range(len(target[0]))])
    return  np.array(target)[:,index][:2]

for epoch_index in range(epoch_num):
    for train_index in range(1,train_num+1):
    #    batch_images, batch_labels = sess.run([images, labels])
        fname="%05d.png"%train_index
        img,label, rgb = read_data_from_file(train_dir,fname)
    #    batch_images = np.expand_dims(img,axis=0)
    #    batch_labels = np.expand_dims(label,axis=0)
        batch_images_arr=[]
        batch_labels_arr=[]
        batch_tri_area_arr=[]
        
        for batch_index in range(batch_num):
            i_UR_center = UR_center(img[:,:,3])
            h_start_index = i_UR_center[0] - np.int32(patch_size/2) + 1
            w_start_index = i_UR_center[1] - np.int32(patch_size/2) + 1
            #boundary security
            if h_start_index<0:
                h_start_index = 0
            if w_start_index<0:
                w_start_index = 0
            if h_start_index+patch_size>img.shape[0]:
                h_start_index=img.shape[0]-patch_size
            if w_start_index+patch_size>img.shape[1]:
                w_start_index=img.shape[1]-patch_size
            tmp_img = img[h_start_index:h_start_index+patch_size, w_start_index:w_start_index+patch_size, :]
            tmp_label = label[h_start_index:h_start_index+patch_size, w_start_index:w_start_index+patch_size, :]/255
            tmp_rgb = rgb[h_start_index:h_start_index+patch_size, w_start_index:w_start_index+patch_size, :]
#            tmp_img = img
#            tmp_label = label/255
#            tmp_rgb = rgb
            tmp_mask = tmp_img[:,:,3]==0
            tmp_grad = sobel_demo(tmp_rgb,tmp_mask)
            tri_area = np.copy(tmp_img[:,:,3])
            tri_area = np.expand_dims(tri_area,axis=3)
#            mask=np.copy(tmp_label)
#            mask[tmp_label!=1]=0
            mask = np.int32(tmp_img[:,:,3]==255)
            tmp_img[:,:,3]=np.copy(mask.reshape(patch_size,patch_size))
#            tmp_img[:,:,3]=np.copy(mask.reshape(img_height,img_width))
#            tmp_img[:,:,0:3]=np.copy(tmp_grad/255)
            
            batch_images_arr.append(tmp_img)
            batch_labels_arr.append(tmp_label)
            batch_tri_area_arr.append(tri_area)
        
        batch_images = np.array(batch_images_arr)
        batch_labels = np.array(batch_labels_arr)
        batch_tri_area = np.array(batch_tri_area_arr)
        
        train_feed_dict={xs:batch_images,ys:batch_labels,tri:batch_tri_area, keep_prob:0.5}
        train_step.run(feed_dict=train_feed_dict)
        
        if train_index%200==0 or train_index == 1:
            result = sess.run(merged,feed_dict=train_feed_dict) #计算需要写入的日志数据
            writer.add_summary(result,train_index + epoch_index * train_num) #将日志数据写入文件
            saver.save(sess,os.path.join(model_dir,"my-model-%d"%epoch_index), global_step=(train_index))
            print("train_index: ", train_index)
            loss=cross_entropy.eval(feed_dict=train_feed_dict)
            print("loss: ",loss)
            for i in range(batch_num):
                res_ = y_conv.eval(feed_dict=train_feed_dict)[i,:,:,:]
                plt.figure(figsize=(20, 100))
                plt.subplot(1,4,1)
                plt.imshow(batch_images[i,:,:,0:3])
                plt.subplot(1,4,2)
                plt.imshow(np.concatenate((np.expand_dims(batch_images[i,:,:,3],axis=2),np.expand_dims(batch_images[i,:,:,3],axis=2),np.expand_dims(batch_images[i,:,:,3],axis=2)),axis=2))
                plt.subplot(1,4,3)
                plt.imshow(np.concatenate([res_,res_,res_],axis=2))
                plt.subplot(1,4,4)
                plt.imshow(np.concatenate((batch_labels[i],batch_labels[i],batch_labels[i]),axis=2))
                plt.show()
        
#        print("111")

#for i in range(2000):
##    batch = mnist.train.next_batch(10)
#    if i%100 == 0:
#        train_accuracy = accuracy.eval(feed_dict={
#        xs:batch[0], ys: batch[1], keep_prob: 1.0})
#        print("step %d, training accuracy %g"%(i, train_accuracy))
#    train_step.run(feed_dict={xs: batch[0], ys: batch[1], keep_prob: 0.5})

