import numpy as np
np.random.seed(2020)
import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#tf.random.set_seed(2020)
from scipy import io

#%% system parameters
Nt = 128
p = 1
SNR = 15 #dB
SNR_linear = 10**(SNR/10)
sigma_2 = p/SNR_linear

RF_chain_ratio = 1/4
R = int(RF_chain_ratio*Nt)

H_list = io.loadmat('./data/H_list_360.mat')['H_list']

# keep part of the data
H_list = H_list[:200000]

print(H_list.shape)

# apply the sensing matrix
def DFT_matrix(N):
    F = np.zeros((N,N))+1j*np.zeros((N,N))
    # DFT矩阵每一行对应将角度的sin值进行N等分
    for i in range(-int(N/2),int(N/2)):
        for k in range(N): 
            F[i+int(N/2),k] = np.exp(-1j*2*np.pi*(i+1/2)*k/N)/np.sqrt(N)
    return F

F = DFT_matrix(Nt)
B = np.transpose(np.conjugate(F))

# use angular domain channel
angular_channels = np.transpose(F.dot(np.transpose(H_list)))
 
# use channel in the original domain 
#angular_channels = H_list

print(angular_channels.shape)

data_num = len(angular_channels)

# add noise to angular channel
LS_noise = np.sqrt(1/2*sigma_2)*(np.random.randn(data_num,Nt)+1j*np.random.randn(data_num,Nt))        
angular_channels_noisy = angular_channels + np.transpose(F.dot(np.transpose(LS_noise)))

# load ZC matrix
W = io.loadmat('./data/W_%d_%d.mat'%(Nt,R))['W2']
fai = W.dot(B)
dataset = np.transpose(fai.dot(np.transpose(angular_channels_noisy)))
#print(fai.dot(angular_channels_noisy[0]))

dataset = np.expand_dims(dataset,axis=-1)
dataset = np.concatenate([np.real(dataset),np.imag(dataset)],axis=-1)

labelset = np.concatenate([np.real(angular_channels),np.imag(angular_channels)],axis=-1)

print(dataset.shape)
print(labelset.shape)


process = 'train'
    

if process == 'test':
    #%% save data for matlab VBI test
    test_num = 500
    received_signal = dataset[-test_num:,:,0]+1j*dataset[-test_num:,:,1]
    io.savemat('./data/test_dataset_%d_dB.mat'%SNR,{'dataset':received_signal,'labelset':angular_channels[-test_num:],'fai':fai})


#%% network training
from tensorflow.keras.layers import GlobalAvgPool1D,Multiply,Activation,Dense,Conv1D,Conv2D,Flatten,Permute,Reshape,Input,BatchNormalization,Concatenate,Add,Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras import backend as K
import os

# limit memory usage
os.environ["CUDA_VISIBLE_DEVICES"]='0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0],True)
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)


def naive_attention(x):
    '''
        squeeze and excitation
    '''
    reduction_ratio = 1
    num_filter = int(x.shape[-1])
    num_neurons = num_filter//reduction_ratio
    
    # squeeze
    x1 = GlobalAvgPool1D()(x)  

    # attention map prediction
    x2 = Dense(num_neurons,activation='relu')(x1)
    attention_map = Dense(num_filter,activation='sigmoid')(x2)
    
    # feature recalibration
    x = Multiply()([x,attention_map])

    return x



def feature_attention(x,original_input):
    '''
        squeeze and excitation
    '''
    reduction_ratio = 1
    num_filter = int(x.shape[-1])
    num_neurons = num_filter//reduction_ratio
    
    # squeeze
    x1 = GlobalAvgPool1D()(x)  

    # concatenate with extracted features from the original input
    original_input = Flatten()(original_input)
    features = Dense(32,activation='relu')(original_input)
    features = BatchNormalization()(features)
    features = Dense(16,activation='linear')(features)
    x1 = Concatenate()([x1,features])  


    # conv layers for feature extraction
    #features = Conv1D(filters=16,kernel_size=7,padding='same')(original_input)
    #features = BatchNormalization()(features)
    #features = Activation('relu')(features)
    #features = Conv1D(filters=4,kernel_size=5,padding='same')(features)
    #features = BatchNormalization()(features)
    #features = Activation('relu')(features)
    #features = Flatten()(features)
    #features = Dense(32,activation='linear')(features)
    
    # attention map prediction
    x1 = Concatenate()([x1,features])  
    x2 = Dense(num_neurons,activation='relu')(x1)
    attention_map = Dense(num_filter,activation='sigmoid')(x2)
    
    # feature recalibration
    x = Multiply()([x,attention_map])

    return x

def est_net(lr,Nt,net,mode):

    # Second part, channel estimator
    x0 = Input(shape=(R,2))

    if net=='CNN':

        # change the order
        x1 = Permute((2,1))(x0)
        x1 = Flatten()(x1)
        x1 = Reshape((R,2))(x1)

        x1 = Conv1D(filters=96,kernel_size=7,padding='same')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        if mode == 'naive':
            x1 = naive_attention(x1)
        if mode == 'feature':
            x1 = feature_attention(x1,x0)     
 
        x2 = Conv1D(filters=96,kernel_size=5,padding='same')(x1)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
        if mode == 'naive':
            x2 = naive_attention(x2)
        if mode == 'feature':
            x2 = feature_attention(x2,x0) 

        x3 = Flatten()(x2)
        prediction = Dense(2*Nt,activation='linear')(x3) 


    if net=='CNN2':

        x1 = Conv1D(filters=96,kernel_size=7,padding='same')(x0)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x2 = Conv1D(filters=96,kernel_size=7,padding='same')(x1)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
        x2 = Conv1D(filters=96,kernel_size=5,padding='same')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
        x2 = Conv1D(filters=96,kernel_size=5,padding='same')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
        x2 = Conv1D(filters=96,kernel_size=3,padding='same')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)

        x3 = Flatten()(x2)
        prediction = Dense(2*Nt,activation='linear')(x3)


    if net=='CNN3':
        x1 = Flatten()(x0)
        #x1 = Dense(4*Nt,activation='relu')(x1)
        #x1 = BatchNormalization()(x1)
        x1 = Dense(2*Nt,activation='linear')(x1)
        x1 = Reshape((Nt,2))(x1)

        x1 = Conv1D(filters=96,kernel_size=7,padding='same')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        if mode == 'naive':
            x1 = naive_attention(x1)
        if mode == 'feature':
            x1 = feature_attention(x1,x0)     
 
        x2 = Conv1D(filters=96,kernel_size=5,padding='same')(x1)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
        if mode == 'naive':
            x2 = naive_attention(x2)
        if mode == 'feature':
            x2 = feature_attention(x2,x0) 

        x3 = Flatten()(x2)
        #prediction = Conv1D(filters=2,kernel_size=3,padding='same')(x2)
        #prediction = Flatten()(prediction)
        prediction = Dense(2*Nt,activation='linear')(x3)

    if net == 'res_CNN':
        x = Conv1D(filters=96,kernel_size=7,padding='same')(x0)
        x = BatchNormalization()(x)
        x_init = Activation('relu')(x)
        if mode=='naive':
            x_init = naive_attention(x_init)
        if mode == 'feature':
            x_init = feature_attention(x_init,x0)  
        for i in range(3): # number of residual blocks
            x = Conv1D(filters=96,kernel_size=5,padding='same')(x_init)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv1D(filters=96,kernel_size=3,padding='same')(x)
            x = BatchNormalization()(x)
            if mode=='naive':
                x = naive_attention(x)
            if mode == 'feature':
                x = feature_attention(x,x0)   
            x_init = Add()([x,x_init])
            x_init = Activation('relu')(x_init)

        x2 = Conv1D(filters=96,kernel_size=3,padding='same')(x_init)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
        if mode =='naive':
            x2 = naive_attention(x2)
        if mode == 'feature':
            x2 = feature_attention(x2,x0)  

        x3 = Flatten()(x2)        
        prediction = Dense(2*Nt,activation='linear')(x3)

    if net=='FNN':
        x1 = Flatten()(x0) # 256,512,256
        x1 = Dense(256,activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dense(512,activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dense(256,activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        x3 = Flatten()(x1)
        prediction = Dense(2*Nt,activation='linear')(x3)

    if net=='FNN3':
        x1 = Flatten()(x0)
        x1 = Dense(96*96//2,activation='relu')(x1)
        x1 = BatchNormalization()(x1)

        x3 = Flatten()(x1)
        prediction = Dense(2*Nt,activation='linear')(x3)

    if net=='FNN2':
        x1 = Flatten()(x0)

        #x1 = Dense(128,activation='relu')(x1)
        #x1 = BatchNormalization()(x1)

        x1 = Dense(16*192,activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        # grouping
        x1 = Reshape((16,192))(x1)
        # attention
        x1 = naive_attention(x1)
        #x1 = feature_attention(x1,x0)
        
        #x1 = Flatten()(x1)
        #x1 = Dense(512,activation='relu')(x1)
        #x1 = BatchNormalization()(x1)

        x3 = Flatten()(x1)
        prediction = Dense(2*Nt,activation='linear')(x3)

    model = Model(inputs=x0,outputs=prediction)
    model.compile(loss='mse',optimizer=Adam(lr=lr))
    model.summary()
    return model


lr = 1e-3
epochs = 1000
batch_size = 500

net = 'FNN2'

mode = 'no'

if net=='FNN':
    mode = 'no'

best_model_path = './models/best_%s_attention_%s_%d_dB.h5'%(mode,net,SNR)

checkpointer = ModelCheckpoint(best_model_path,verbose=1,save_best_only=True,save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10,verbose=1, mode='auto',min_delta=0.00001,min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss',min_delta=0.00001,patience=25)

model = est_net(lr,Nt,net,mode)

if process == 'train':
    model.fit(dataset,labelset,epochs=epochs,batch_size=batch_size,callbacks=[checkpointer,reduce_lr,early_stopping],validation_split=0.2)

if process == 'test':
    model.load_weights(best_model_path)
    prediction = model.predict(dataset[-test_num:])
    truth = labelset[-test_num:]
    error = prediction - truth
    mse = np.mean(np.linalg.norm(error,axis=-1)**2)/Nt/2
    print(mse)
    #nmse = np.mean((np.linalg.norm(error,axis=-1)/np.linalg.norm(truth,axis=-1))**2)
    #print(nmse)


