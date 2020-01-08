import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.signal import butter, savgol_filter
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import keras.layers as L
import tensorflow as tf
from sklearn import linear_model
from sklearn import svm 
from keras import callbacks
import keras.models as M
import keras.backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint

tf.set_random_seed(103)

def rmse(ytrue,ypred):
    #ytrue = tf.reshape(ytrue,[-1,1])
    #ypred = tf.reshape(ypred,[-1,1])
    return K.sqrt(K.mean(K.square(ytrue - ypred)))

def create_time_stepped_data(df, time_steps,num_features):
    d = []
    for i in range(time_steps):
        d.append(df.shift(-i).values[:-time_steps].reshape(-1,num_features+1))
    return np.transpose(np.array(d),(1,0,2))

def create_formatted_data(df, time_steps, num_features, fut_type=1):
    assert (time_steps%2 !=0), "Time steps should be odd!"
    print ("Create formatted data : (df shape)", df.shape)
    d = create_time_stepped_data(df, time_steps, num_features)
    print ("Create time-stepped data : (d shape)", d.shape)
    past = d[:,:int(time_steps/2),:]
    if(fut_type == 1):
        fut = np.flip(d[:,int(time_steps/2)+1:,:],1)
#         fut = np.flip(d[:,int(time_steps/2):,:],1)
    else:
        fut = np.flip(d[:,:int(time_steps/2),:],1)
    y = d[:,int((time_steps)/2),-1]
#     for i in range(past.shape[0]):
#         for j in range(past.shape[1]):
#             past[i][j] = (past[i][j] - df.mean())/df.var()
#             fut[i][j] = (fut[i][j] - df.mean())/df.var()
            
    print ("Create formatted data : ", past.shape, fut.shape, y.shape)
    return past,fut,y

def train_test_split(past,fut,y,df,ratio):
    train_split = int(ratio*y.shape[0])
    return past[:train_split,:], past[train_split:,:], fut[:train_split,:], np.zeros_like(past[train_split:,:]), y[:train_split], y[train_split:]

def model_creation(input_shape1, input_shape2, model_version, optim, problem_type=0):
    """
    op_sequence : False, by default. If True, then also uses the num_nodes parameter.
    num_nodes   : The number that denotes that how many values to predict at the output layer.
    """
    print ("Inside model_creation")
    
    past_inp = L.Input(shape=(input_shape1))
    fut_inp = L.Input(shape=(input_shape2))
    
    if(model_version == "M1V1"):
        cnn1 = L.Conv1D(filters=32,kernel_size=5)(past_inp)
        cnn1 = L.Conv1D(filters=32,kernel_size=2)(cnn1)
    
        cnn2 = L.Conv1D(filters=32,kernel_size=5)(fut_inp)
        cnn2 = L.Conv1D(filters=32,kernel_size=2)(cnn2)
    
        lstm_inp = L.Average()([cnn1,cnn2])
    
        lstm_out = L.LSTM(32, recurrent_dropout=0.2, return_sequences=True, bias_initializer='ones')(lstm_inp)
    
        x1 = L.Average()([lstm_out,lstm_inp])
        x1 = L.Flatten()(x1)
        
    elif(model_version == "M1V2"):
        cnn1 = L.Conv1D(filters=32,kernel_size=5)(past_inp)
        cnn1 = L.Conv1D(filters=32,kernel_size=3)(cnn1)
    
        cnn2 = L.Conv1D(filters=32,kernel_size=5)(fut_inp)
        cnn2 = L.Conv1D(filters=32,kernel_size=3)(cnn2)
    
        x1 = L.Average()([cnn1,cnn2])
        x1 = L.Flatten()(x1)
    
    elif(model_version == "M1V3"):
        x1 = L.LSTM(32, recurrent_dropout=0.2, return_sequences=True, bias_initializer='ones')(past_inp)
        #x1 = L.LSTM(32, recurrent_dropout=0.2, return_sequences=True, bias_initializer='ones')(x1)
        x1 = L.LSTM(32, recurrent_dropout=0.2, bias_initializer='ones')(x1)
        
    elif(model_version == "M2V1"):
        cnn1 = L.Conv1D(filters=32,kernel_size=5)(past_inp)
        cnn1 = L.Conv1D(filters=32,kernel_size=2)(cnn1)

        lstm_out1 = L.LSTM(32, recurrent_dropout=0.3, return_sequences=True, bias_initializer='ones')(cnn1)
        lstm_out1 = L.Average()([cnn1,lstm_out1])

        cnn2 = L.Conv1D(filters=32,kernel_size=5)(fut_inp)
        cnn2 = L.Conv1D(filters=32,kernel_size=2)(cnn2)

        lstm_out2 = L.LSTM(32, recurrent_dropout=0.3, return_sequences=True, bias_initializer='ones')(cnn2)
        lstm_out2 = L.Average()([cnn2,lstm_out2])

        x1 = L.Average()([lstm_out1,lstm_out2])
        x1 = L.Flatten()(x1)
        
    elif(model_version == "M2V2"):
        lstm_out1 = L.LSTM(32, recurrent_dropout=0.3, return_sequences=True, bias_initializer='ones')(past_inp)
        lstm_out2 = L.LSTM(32, recurrent_dropout=0.3, return_sequences=True, bias_initializer='ones')(fut_inp)
        x1 = L.Average()([lstm_out1,lstm_out2])
        x1 = L.Flatten()(x1)
        
    elif(model_version == "M3V1"):
#         cnn_inp = L.Concatenate(axis=1)([past_inp,fut_inp])
        cnn = L.Conv1D(filters=32,kernel_size=5)(past_inp)
        cnn = L.Conv1D(filters=32,kernel_size=2)(cnn)
        
        layer = L.Lambda(lambda cnn: K.reverse(cnn,axes=1))
        cnn2 = layer(cnn)
        
        lstm1 = L.LSTM(32, recurrent_dropout=0.3, return_sequences=True, bias_initializer='ones')(cnn)
        lstm2 = L.LSTM(32, recurrent_dropout=0.3, return_sequences=True, bias_initializer='ones')(cnn2)
        lstm = L.Average()([lstm1,lstm2])
        x1 = L.Average()([cnn,lstm])
        x1 = L.Flatten()(x1)
    
    elif(model_version == "M3V2"):
        cnn_inp = L.Concatenate(axis=1)([past_inp,fut_inp])
        cnn = L.Conv1D(filters=32,kernel_size=5)(cnn_inp)
        cnn = L.Conv1D(filters=32,kernel_size=2)(cnn)
        
        layer = L.Lambda(lambda cnn: K.reverse(cnn,axes=1))
        cnn2 = layer(cnn)
        
        lstm1 = L.LSTM(32, recurrent_dropout=0.3, return_sequences=True, bias_initializer='ones')(cnn)
        lstm2 = L.LSTM(32, recurrent_dropout=0.3, return_sequences=True, bias_initializer='ones')(cnn2)
        lstm = L.Average()([lstm1,lstm2])
        x1 = L.Average()([cnn,lstm])
        x1 = L.Flatten()(x1)
    
    elif(model_version == "M4"):
        cnn1 = L.Conv1D(filters=32,kernel_size=5)(past_inp)
        cnn1 = L.Conv1D(filters=32,kernel_size=2)(cnn1)
        
        cnn2 = L.Conv1D(filters=32,kernel_size=5)(fut_inp)
        cnn2 = L.Conv1D(filters=32,kernel_size=2)(cnn2)

        lstm1 = L.Bidirectional(L.LSTM(16, recurrent_dropout=0.3, return_sequences=True, bias_initializer='ones'))(past_inp)
        lstm2 = L.Bidirectional(L.LSTM(16, recurrent_dropout=0.3, return_sequences=True, bias_initializer='ones'))(fut_inp)

        out1 = L.Concatenate(axis=1)([cnn1,lstm1])
        out2 = L.Concatenate(axis=1)([cnn2,lstm2])
        
#         out1 = L.Average()([cnn1,cnn2])
#         ousst2 = L.Average()([lstm1,lstm2])
        
#         x1 = L.Concatenate(axis=1)([out1,out2])
        x1 = L.Average()([out1,out2])
        x1 = L.Flatten()(x1)
        
    x1 = L.Dense(256)(x1)
    x1 = L.advanced_activations.LeakyReLU(0.2)(x1)
    x1 = L.Dense(256)(x1)
    x1 = L.advanced_activations.LeakyReLU(0.2)(x1)
    x1 = L.Dense(256)(x1)
    x1 = L.advanced_activations.LeakyReLU(0.2)(x1)
    x1 = L.Dense(256)(x1)
    x1 = L.advanced_activations.LeakyReLU(0.2)(x1)
    x1 = L.Dense(256)(x1)
    x1 = L.advanced_activations.LeakyReLU(0.2)(x1)
    x1 = L.Dense(256)(x1)
    x1 = L.advanced_activations.LeakyReLU(0.2)(x1)
    
    #Classification Part.
    if(problem_type == 1):
        main_out = L.Dense(2,activation='softmax')(x1)
        model = M.Model(inputs=[past_inp,fut_inp], outputs=[main_out], name=model_version)
        model.summary()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    #Regression Part.
    else:
        x1 = L.Dense(1)(x1)
        main_out = L.advanced_activations.LeakyReLU(0.2)(x1)
        model = M.Model(inputs=[past_inp,fut_inp], outputs=[main_out], name=model_version)   
        model.compile(optimizer = optim, loss=tf.losses.huber_loss, metrics=['mae','mse',rmse])
    
    model.summary()
    
    return model

def linear_regression(model_pred, data):
    lin_model = linear_model.LinearRegression()
    split = int(0.7 * data.shape[0])
    training_data_x, training_data_y = model_pred[:split], data[:split]
    testing_data_x, testing_data_y = model_pred[split:], data[split:]
    lin_model.fit(training_data_x, training_data_y)
    lin_model_pred = lin_model.predict(testing_data_x)
    print("MAE %.4f\nR2 %.4f\nRMSE %.4f"%(mean_absolute_error(testing_data_y,lin_model_pred),                            r2_score(testing_data_y,lin_model_pred),np.sqrt(mean_squared_error(testing_data_y,lin_model_pred))))
    return lin_model

def help():
    print ("You're on your own.")