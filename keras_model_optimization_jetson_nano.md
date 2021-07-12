# Keras model optimization in jetson nano/nx
keras model trained in PC can be optimized(speed/size) when used in jetson nano/nx. The optimization procedure is listed below: 

* build and train a keras(tensorflow 2.x) model
* convert keras model to onnx format
* convert onnx format to tensorrt model
* use tensorrt model for inference

## build and train a keras(tensorflow 2.x) model with cats/dogs datatset
First, we build our keras(tf2.x) model and save the model to tensorflow saved_model format. The datatset can be downloaded from <a href="https://www.kaggle.com/chetankv/dogs-cats-images">Here</a>

```
# this will prevent from using GPU even though a GPU installed
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#from tensorflow.python.client import device_lib
#print('GPU device: ',device_lib.list_local_devices())

import tensorflow as tf
import numpy as np
# Train a CNN classifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
#from io import BytesIO
import matplotlib.image as mpimg
import os
#import keras
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
from tensorflow.keras import losses,activations,optimizers
from tensorflow.keras import backend as K

print (tf.__version__)

# Helper function
def predict_image(classifier, img):
    
    # Flatten the image data to correct feature format. its sjape is (1,128,128,3)
    imgfeatures = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    imgfeatures = imgfeatures.astype('float32')
    imgfeatures /= 255

    # Use the classifier to predict the class
    predicted_class = classifier.predict(imgfeatures)
    # return the index of the maximun value in an array 
    i = np.argmax(predicted_class, axis=1)
    return i
# Resize image to size proportinally. size is a tuple (width,height)
def resize_image(img, size):
    
    # Convert RGBA images to RGB
    if np.array(img).shape[2] == 4:
        img = img.convert('RGB')
        
    # resize the image
    img.thumbnail(size, Image.ANTIALIAS)
    newimg = Image.new("RGB", size, (255, 255, 255))
    newimg.paste(img, (int((size[0] - img.size[0]) / 2), int((size[1] - img.size[1]) / 2)))
    
    return newimg

# expect all dataset images in ./folder directory
def load_data (folder):
    # iterate through folders, assembling feature, label, and classname data objects

    c = 0
    features = []
    labels = np.array([])
    classnames = []
    # os.walk return 3-tuple for each directory in rootfloder 
    # root: './train  dirs: all subdir in root, like ['cats','dogs']
    for root, dirs, filenames in os.walk(folder):    # root= ./train
        for d in dirs:
            # use the folder name as the class name for this label
            classnames.append(d)
            # return all files in specified directory. os.path.join(root,d) generates a full path of a directory
            files = os.listdir(os.path.join(root,d))
            print('classname={},c={}'.format(d,c))
            for f in files:
                imgFile = os.path.join(root,d, f)
#                print('file={}'.format(imgFile))
#                img = plt.imread(imgFile)   # img's shape is (128,128,3)
                img = Image.open(imgFile)
                # resize image
                img = resize_image(img,(128,128))
                # convert Image data to numpy array. its shape is (128,128,3)
                img_arr = np.array(img)
                # append data to list
                features.append(img_arr)
                # 标签数值化。append data to numpy array. labels is a one dimention numpy array [0,0,...1,...]
                labels = np.append(labels, c)   # apend c to labels numpy array.  c is number 0,1,...
            c = c + 1
    features = np.array(features)
    # features 及 labels 一一对应,classnames 为类别标签
    return features, labels, classnames
# Prepare the image data
# features in shape (8000,128,128,3), labels in shape (8000,) classnames is  a list ['cats','dogs']
features, labels, classnames = load_data('./train')
print('feature shape={}',format(features.shape))


# split into training and testing sets
# arrays : sequence of indexables with same length / shape[0] 
# Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.30)

# Format features. convert from int to float
x_train = x_train.astype('float32')
# RGB data normalization
x_train /= 255.
x_test = x_test.astype('float32')
x_test /= 255.

# labels one-hot encoding. before convertion: y_train in shape (5600,) that is [0,1,0,1,...] after concertion: y_train in shape (5600,2) that is [[1,0],[0,1],[1,0],...]
y_train = utils.to_categorical(y_train, len(classnames))
y_train = y_train.astype('float32')
y_test = utils.to_categorical(y_test, len(classnames))
y_test = y_test.astype('float32')

model = Sequential()
# filter size (3,3) step default is (1,1). input shape=(128,128,3) padding default 'valid' (no padding)
# output feature maps: (n+2p-f)/s+1 = (128+0-3)/1+1 = 126. so shape -> (126,126,32)
# parameters number = 3*3*3*32+32 = 896. filter size (3*3*3), filter number = 32
#model.add(Conv2D(32, (3, 3), input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]), activation='relu'))
model.add(Conv2D(32, (3, 3), strides=(2,2), padding='valid', input_shape=(128,128,3), activation='relu'))
# output feature maps: pool_size=(2,2) so width and height half changed shape -> (63,63,32)
# parameters numbers = 0(no convolution filter)
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# output feature maps: size=63+0-2+1  shape -> (61,61,32) 
# parameters number = 3*3*32*32+32 = 9248. pay attention the depth of the this layers' filter is the same as the last layer's filter number(32 here)
model.add(Conv2D(32, (3, 3), activation='relu'))
# output feature maps: width and height half changed shape -> (30,30,32)
# parameters numbers = 0(no convolution filter)
model.add(MaxPooling2D(pool_size=(2, 2)))    # strides default to pool_size
# feature maps no change. parameters = 0
model.add(Dropout(0.5))
# output feature maps: size=30+0-3+1  shape -> (28,28,32) parameter numbers = 3*3*32*32+32 = 9248
model.add(Conv2D(32, (3, 3), activation='relu'))
# output feature maps: width and height half changed. shape -> (14,14,32). no parameters
model.add(MaxPooling2D(pool_size=(2, 2)))
# output feature maps: no change, parameters = 0
model.add(Dropout(0.5))
# output feature maps: shape flattened from (14,14,32) to (14*14*32,) = (6272,)
model.add(Flatten())
# output feature maps: 2, parameters numbers = 2*6272+2 = 12546
model.add(Dense(len(classnames), activation='softmax'))
model.summary()
for layer in model.layers:
    print('trainable={}'.format(layer.trainable))
'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 126, 126, 32)      896          => 3*3*32+32 = 896
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 63, 63, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 61, 61, 32)        9248         => 3*3*32*32+32 = 9248  filter size (3,3,32) filter number 32
_________________________________________________________________      note the filter depth(channel) is the same as the last layer's filter number
max_pooling2d_2 (MaxPooling2 (None, 30, 30, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 30, 30, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 28, 28, 32)        9248         => 3*3*32*32+32 = 9248 filter size (3,3,32) filter number 32
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 14, 14, 32)        0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 14, 14, 32)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 6272)              0            => 14*14*32 = 6272
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 12546        => 2*6272+2 = 12546
=================================================================
Total params: 31,938
Trainable params: 31,938
Non-trainable params: 0
'''
optimizer = optimizers.Adam
loss = losses.categorical_crossentropy
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
num_epochs = 20
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=64, validation_data=(x_test, y_test))

# plot training accuracy and loss from history
plt.figure(figsize=(15,10))
plt.subplot(121)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.show()

# load predict images
predict_data = []
predict_images = []
size = (128,128)
files = os.listdir('./predict')
for f in files:
    if not 'jpg' in f:
        continue
    img = Image.open(os.path.join('./predict',f))
    # resize images
    img = resize_image(img,size)
    predict_data.append(np.array(img))


#img = Image.open('resized_images/hardshell_jackets_test/resized_10269570x1012905_zm.jpeg')
# normalize the image data
predict_data = np.array(predict_data,dtype='float')/255.
# use the model to predict
predicted_labels_encoded = model.predict(predict_data)
# predicted_labels_encoded is an array ,so np.argmax must specify the axis, so the result is also an arrar                                 
predicted_labels = np.argmax(predicted_labels_encoded,axis=1) 
print('predicted digits={}'.format(predicted_labels))                                
#predicted_labels = encoder.inverse_transform(predicted_labels)
#print('predicted labels=',predicted_labels)
i = 0
#fig = plt.figure(figsize=(10, 10))
for data in predict_data:
    ax = plt.subplot(1,4,i+1)
    # image data should be converted back to its original value in order to show image
    data = (data * 255.).astype(np.uint8)
    imgplot = plt.imshow(data)
    # classnames is a dictionary [0:'cats',1:'dogs']
    ax.set_title(classnames[predicted_labels[i]])
    i = i+1
plt.show()

# save Model Weights and Architecture Together
model.save('catsdogs.h5')
tf.saved_model.save(model, 'catsdogs_saved_model')
# load the saved model
model2 = load_model('catsdogs.h5')
model2.summary()

```

the trained model is saved as tensorflow saved_model format, this is the favorite model format when converted to onnx

## convert keras model to onnx format

## convert onnx format to tensorrt model

## use tensorrt model for inference

