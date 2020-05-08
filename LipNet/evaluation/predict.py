import sys
sys.path.append('../')
# For the video class
from lipnet.lipreading.videos import Video
#For decoding the encoding labels
from lipnet.core.decoders import Decoder
from lipnet.lipreading.helpers import labels_to_text
from lipnet.utils.spell import Spell
#For the lipnet model
from lipnet.model2 import LipNet
#Adam optimizer
from keras.optimizers import Adam
#keras backend
from keras import backend as K
import numpy as np
#For os commands
import os

np.random.seed(55)
#getting the current path
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
#Face predictor path
FACE_PREDICTOR_PATH = os.path.join(CURRENT_PATH,'..','dic_pred','predictors','shape_predictor_68_face_landmarks.dat')

#For gready approach "used in training for finding the fastest path to the predicted label in the dictionary"
#True : means activate , False : use the beam search
PREDICT_GREEDY      = False
PREDICT_BEAM_WIDTH  = 200
#Getting dictionary path " The dectionary is the encoder stage which used to search for the right label "
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'..','dic_pred','dictionaries','grid.txt')

def predict(weight_path, video_path, absolute_max_string_len=32, output_size=28):
    # W_path : path to the weights file
    # Vid_path : path to the input video
    # max_string_len : the num of letters in the sentence best computation and effiency at 32
    # output_size : the number of labels 'output of the model'
    # create video object
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
    if os.path.isfile(video_path):
        video.from_video(video_path)

    #modeling the data format
    if K.image_data_format() == 'channels_first':
        img_c, frames_n, img_w, img_h = video.data.shape
    else:
        frames_n, img_w, img_h, img_c = video.data.shape

    #initializing Model
    lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                    absolute_max_string_len=absolute_max_string_len, output_size=output_size)
    #initializing the optimizer
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #Compiling the model and decressing the loss
    lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    #loading weights
    lipnet.model.load_weights(weight_path)

    spell = Spell(path=PREDICT_DICTIONARY)
    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                      postprocessors=[labels_to_text, spell.sentence])

    # Transforming the video to frame and to array 
    X_data       = np.array([video.data]).astype(np.float32) / 255
    input_length = np.array([len(video.data)])

    y_pred         = lipnet.predict(X_data)
    result         = decoder.decode(y_pred, input_length)[0]

    return (video, result)

video , result = predict("E:\\Graduation Project\\LipNet__\\LipNet\\evaluation\\models\\overlapped-weigh.h5","E:\\Graduation Project\\LipNet__\\LipNet\\evaluation\\models\\id2_vcd_swwp2s.mpg")
print(result)