from keras import backend as k
from keras.layers.core import Lambda


def ctc_lambda_func(args):
    labels , y_pred,num_inputs,num_labels = args
    return k.ctc_batch_cost(labels , y_pred,num_inputs,num_labels)


def CTC(name , args ):
    return Lambda(ctc_lambda_func,output_shape=(1,),name = name)(args)
