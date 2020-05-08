#For os manipulation
import os
import numpy as np
#For keras
from keras import backend as K
#For handling Videos and images
from scipy import ndimage
from scipy.misc import imresize
import skvideo.io
from skimage.transform import resize
import cv2
#For face and mouth detector
import dlib


class Video(object):
    def __init__(self, vtype='mouth', face_predictor_path=None):
	#v_type : used for indicator for the video if only the mouth exists
        #Face_pred_path : the path for the predictor
        if vtype == 'face' and face_predictor_path is None:
	    # in case of mouth type videos there is no need to the predictor data but the opposite in the face type
            raise AttributeError('Face video need to be accompanied with face predictor')
        self.face_predictor_path = face_predictor_path
        self.vtype = vtype
    #handling the video
    def from_video(self, path):
	#path to the video
        #get_video_frames is function below
        frames = self.get_video_frames(path)
	#We need to handle the type of video
        self.handle_type(frames)
	#since type will handle all return itself
        return self

    #Solving type issue
    def handle_type(self, frames):
        if self.vtype == 'mouth':
            self.process_frames_mouth(frames)
        elif self.vtype == 'face':
            self.process_frames_face(frames)
        else:
            raise Exception('Video type not found')

    #Face Handler
    def process_frames_face(self, frames):
        # Face detection using dlib , https://www.pyimagesearch.com/2018/04/02/faster-facial-landmark-detector-with-dlib/
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.face_predictor_path)
        #extract mouth
        mouth_frames = self.get_frames_mouth(detector, predictor, frames)
        # face images
        self.face = np.array(frames)
        #mouth images
        self.mouth = np.array(mouth_frames)
        #Setting the data
        self.set_data(mouth_frames)

    #mouth handler
    def process_frames_mouth(self, frames):
        # the frame contains only mouth
        self.face = np.array(frames)
        self.mouth = np.array(frames)
        self.set_data(frames)

    def get_frames_mouth(self, detector, predictor, frames):
        #New seeting for the processed frame
        MOUTH_WIDTH = 100
        MOUTH_HEIGHT = 50
        HORIZONTAL_PAD = 0.19
        normalize_ratio = None
        mouth_frames = []
        #for every frame
        #first detect the face points
        for frame in frames:
            dets = detector(frame, 1)
            #dets is the frame with points that can be detected
            shape = None
            for k, d in enumerate(dets):
                shape = predictor(frame, d)
                # if he can recognize these points
                i = -1
            if shape is None: # Detector doesn't detect face, just return as is ,No face detected or can't recongnize the face points
                return frames
            mouth_points = []
            for part in shape.parts(): # 
                i += 1
                if i < 48: # Only take mouth region
                    continue
                mouth_points.append((part.x,part.y))
            np_mouth_points = np.array(mouth_points)
            # get the center of the mouth
            mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)
            #Normalizing frames and resizing
            if normalize_ratio is None:
                mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
                mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

                normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

            new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))
            resized_img = imresize(frame, new_img_shape)

            mouth_centroid_norm = mouth_centroid * normalize_ratio

            mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
            mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
            mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
            mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)

            mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]

            mouth_frames.append(mouth_crop_image)
        return mouth_frames
    
    def get_video_frames(self, path):
        #reading the video
        vid_reader = cv2.VideoCapture(path)
        #Turn it to frames
        suc , frame = vid_reader.read()
        frames = np.array(frame)
        frames = np.reshape(frames,(1,frames.shape[0],frames.shape[1],frames.shape[2]))
        while(suc):
            suc , frame = vid_reader.read()
            if not suc:
                break
            frame = np.array(frame)
            frame = np.reshape(frame,(1,frame.shape[0],frame.shape[1],frame.shape[2]))
            frames = np.concatenate([frames , frame],axis=0)
        print(frames.shape)
        return frames
    #setting the final data
    def set_data(self, frames):
        data_frames = []
        for frame in frames:
            frame = frame.swapaxes(0,1) # swap width and height to form format W x H x C
            if len(frame.shape) < 3:
                frame = np.array([frame]).swapaxes(0,2).swapaxes(0,1) # Add grayscale channel
            data_frames.append(frame)
        frames_n = len(data_frames)
        data_frames = np.array(data_frames) # T x W x H x C
        if K.image_data_format() == 'channels_first':
            data_frames = np.rollaxis(data_frames, 3) # C x T x W x H
        self.data = data_frames
        self.length = frames_n