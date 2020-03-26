import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, ZeroPadding3D , BatchNormalization, Flatten, Conv2D , Conv3D , Bidirectional , GRU , SpatialDropout3D , TimeDistributed , AveragePooling2D, AveragePooling3D ,MaxPooling2D, MaxPooling3D , GlobalMaxPooling3D ,GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
#matplotlib inline
import tensorflow as tf
import keras.backend as K

class LipNet(object):
    def __init__ (self , img_h = 50 , img_w = 100 , img_c = 3 , frame_n = 75, max_str_len = 32 , output_siz = 28):
        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        self.frame_n = frame_n
        self.max_str_len = max_str_len
        self.output_siz = output_siz
        self.create()
        
    def create(self):
        
        # check input shape
        
        if K.image_data_format() == 'channels_first' :
            self.input_shape = (self.img_c,self.frame_n,self.img_h,self.img_w)
        else :
            self.input_shape = (self.frame_n,self.img_h,self.img_w,self.img_c)
        
        # X_input at the start 
        self.input_data = Input(shape = self.input_shape , dtype='float32')
        
        
        #Stage 1
        #Set 1
        self.X = ZeroPadding3D(padding=(1,2,2), name ='Z1' )(self.input_data)
        self.X = Conv3D(32,(3,5,5),strides = (1,2,2),name='C1',kernel_initializer = 'he_normal')(self.X)
        self.X = BatchNormalization(name = 'B1')(self.X)
        self.X = Activation('relu' , name = 'A1')(self.X)
        self.X = SpatialDropout3D(0.5)(self.X)
        self.X = MaxPooling3D(pool_size=(1,2,2) , strides = (1,2,2) , name = 'M1')(self.X)
        
        #Stage 1
        #Set 2
        self.X = ZeroPadding3D(padding=(1,2,2), name ='Z2' )(self.X)
        self.X = Conv3D(64,(3,5,5),strides = (1,1,1),name='C2',kernel_initializer = 'he_normal')(self.X)
        self.X = BatchNormalization(name = 'B2')(self.X)
        self.X = Activation('relu' , name = 'A2')(self.X)
        self.X = SpatialDropout3D(0.5)(self.X)
        self.X = MaxPooling3D(pool_size=(1,2,2) , strides = (1,2,2) , name = 'M2')(self.X)
        
        #Stage 1
        #Set 3
        self.X = ZeroPadding3D(padding=(1,1,1), name ='Z3' )(self.X)
        self.X = Conv3D(96,(3,3,3),strides = (1,1,1),name='C3',kernel_initializer = 'he_normal')(self.X)
        self.X = BatchNormalization(name = 'B3')(self.X)
        self.X = Activation('relu' , name = 'A3')(self.X)
        self.X = SpatialDropout3D(0.5)(self.X)
        self.X = MaxPooling3D(pool_size=(1,2,2) , strides = (1,2,2) , name = 'M3')(self.X)
        
        #Stage 2
        # turn to sequential model get rid of the time
        self.X = TimeDistributed(Flatten())(self.X)
        
        #Stage 2
        # set 1
        self.X = Bidirectional(GRU(256,return_sequences = True , kernel_initializer='Orthogonal', name='R1'), merge_mode='concat')(self.X)
        
        #Stage 2
        # set 2
        self.X = Bidirectional(GRU(256,return_sequences = True , kernel_initializer='Orthogonal', name='R2'), merge_mode='concat')(self.X)
        
        #Stage 3
        #convert to a fully connected layer
        self.X = Dense(self.output_siz,kernel_initializer='he_normal', name='D1')(self.X)
        
        #Applying softmax
        self.y_pred = Activation('softmax',name = 'SM1')(self.X)
        
        #Model
        self.model = Model(inputs = self.input_data, outputs = self.y_pred, name='LipNet')
        
    def summary(self):
        Model(inputs=self.input_data, outputs=self.y_pred).summary()



model = LipNet().model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
