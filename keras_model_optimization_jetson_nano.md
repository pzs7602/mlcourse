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
For convenience, we do the conversion task in jetson nano/nx, cause the tensorrt 7.x is already installed in jetson nano/nx with jetsdk 4.5.1.  So copy the keras model to jetson nano/nx. 

install tf2onnx python package:

```
pip3 install tf2onnx
```

### convert saved_model to onnx
This is the easiest way for the conversion. We use tf2onnx:

```
python3 -m tf2onnx.convert --saved-model catsdogs_saved_model --output catsdogs.onnx --opset 9
```

Every library is versioned. scikit-learn may change the implementation of a specific model. That happens for example with the SVC model where the parameter break_ties was added in 0.22. ONNX does also have a version called opset number. Operator ArgMin was added in opset 1 and changed in opset 11, 12, 13. Sometimes, it is updated to extend the list of types it supports, sometimes, it moves a parameter into the input list. The runtime used to deploy the model does not implement a new version, in that case, a model must be converted by usually using the most recent opset supported by the runtime, we call that opset the targeted opset. An ONNX graph only contains one unique opset, every node must be described following the specifications defined by the latest opset below the targeted opset. tf2onnx.convert use opset=9 for default

the output is catsdogs.onnx

## convert onnx format to tensorrt model

for onnx model to tensorrt engine conversion, we use 

```
python3 buildEngine.py --onnx_file catsdogs.onnx --plan_file catsdogs.engine
```

## use tensorrt model for inference

we use both the converted tensorrt engine and the original keras h5 model for inference:
```
python3 trt_catsdogs.py catsdogs.engine ../../catsdogs2/predict/dogs34.jpg
python3 trt_catsdogs.py ../../catsdogs2/catsdogs.h5 ../../catsdogs2/predict/dogs34.jpg
```
trt_catsdogs.py is listed below:

```
"""trt_catsdogs.py

This script is for testing a trained Keras ImageNet model.  The model
could be one of the following 2 formats:

    1. tf.keras model (.h5)
    2. optimized TensorRT engine (.engine)

Example usage #1:
$ python3 trt_catsdogs.py ../../catsdogs2/catsdogs.h5 ../../catsdogs2/predict/dogs34.jpg

Example usage #2:
$ python3 trt_catsdogs.py catsdogs.engine ../../catsdogs2/predict/dogs34.jpg
"""


import argparse
import time
import numpy as np
import cv2
import tensorflow as tf

WIDTH = 128
HEIGHT = 128
CHANNELS = 3
NUM_CLASSES = 2

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        help='a tf.keras model or a TensorRT engine, e.g. saves/googlenet_bn-model-final.h5 or tensorrt/googlenet_bn.engine')
    parser.add_argument('jpg',
                        help='an image file to be predicted')
    args = parser.parse_args()
    return args


def preprocess(img):
    """Preprocess an image for Keras ImageNet model inferencing."""
    if img.ndim != 3:
        raise TypeError('bad ndim of img')
    if img.dtype != np.uint8:
        raise TypeError('bad dtype of img')
    img = cv2.resize(img, (WIDTH, HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img *= (2.0/255)  # normalize to: 0.0~2.0
    img -= 1.0        # subtract mean to make it: -1.0~1.0
    img = np.expand_dims(img, axis=0)
    return img


def infer_with_tf(img, model):
    """Inference the image with TensorFlow model."""
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#    from utils.utils import config_keras_backend, clear_keras_session
#    from models.adamw import AdamW

#    config_keras_backend()

    # load the trained model
    net = tf.keras.models.load_model(model)
    predictions = net.predict(img)[0]

#    clear_keras_session()

    return predictions


def init_trt_buffers(cuda, trt, engine):
    """Initialize host buffers and cuda buffers for the engine."""
#    assert engine[0] == 'input_1:0'
#    assert engine[0] == 'input_1:0'
    assert engine.get_binding_shape(0)[1:] == (WIDTH, HEIGHT, CHANNELS)
    size = trt.volume((1, WIDTH, HEIGHT, CHANNELS)) * engine.max_batch_size
    host_input = cuda.pagelocked_empty(size, np.float32)
    cuda_input = cuda.mem_alloc(host_input.nbytes)
#    assert engine[1] == 'dense/Softmax:0'
    print(f'engine[0]={engine[0]}, engine[1]={engine[1]}')
    assert engine.get_binding_shape(1)[1:] == (NUM_CLASSES,)
    size = trt.volume((1, NUM_CLASSES)) * engine.max_batch_size
    host_output = cuda.pagelocked_empty(size, np.float32)
    cuda_output = cuda.mem_alloc(host_output.nbytes)
    return host_input, cuda_input, host_output, cuda_output


def infer_with_trt(img, model):
    """Inference the image with TensorRT engine."""
    import pycuda.autoinit
    import pycuda.driver as cuda
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    with open(model, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    assert len(engine) == 2, 'ERROR: bad number of bindings'
    host_input, cuda_input, host_output, cuda_output = init_trt_buffers(
        cuda, trt, engine)
    stream = cuda.Stream()
    context = engine.create_execution_context()
    context.set_binding_shape(0, (1, WIDTH, HEIGHT, CHANNELS))
    np.copyto(host_input, img.ravel())
    cuda.memcpy_htod_async(cuda_input, host_input, stream)
    if trt.__version__[0] >= '7':
        context.execute_async_v2(bindings=[int(cuda_input), int(cuda_output)],
                                 stream_handle=stream.handle)
    else:
        context.execute_async(bindings=[int(cuda_input), int(cuda_output)],
                              stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, cuda_output, stream)
    stream.synchronize()
    return host_output


def main():
    args = parse_args()

    # load the cls_list (index to class name)
    # with open('data/synset_words.txt') as f:
    #     cls_list = sorted(f.read().splitlines())
    cls_list = ['cat','dog']

    # load and preprocess the test image
    img = cv2.imread(args.jpg)
    if img is None:
        raise SystemExit('cannot load the test image: %s' % args.jpg)
    img = preprocess(img)

    fps = 0.0
    tic = time.time()
    # predict the image
    if args.model.endswith('.h5'):
        predictions = infer_with_tf(img, args.model)
    elif args.model.endswith('.engine'):
        predictions = infer_with_trt(img, args.model)
    else:
        raise SystemExit('ERROR: bad model')
    toc = time.time()
    curr_fps = 1.0 / (toc - tic)
    # calculate an exponentially decaying average of fps number
    fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
    print(f'toc-tic={toc - tic:.2f}, fps={fps:.2f}')
    # postprocess
    top5_idx = predictions.argsort()[::-1][:5]  # take the top 5 predictions
    for i in top5_idx:
        print('%5.2f   %s' % (predictions[i], cls_list[i]))


if __name__ == '__main__':
    main()

```