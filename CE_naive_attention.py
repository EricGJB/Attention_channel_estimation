import numpy as np
np.random.seed(2020)
import tensorflow as tf
tf.random.set_seed(2020)
from scipy import io

#%% system parameters
Nt = 128
p = 1
SNR = 20 # dB
SNR_linear = 10**(SNR/10) 
sigma_2 = p/SNR_linear

# load data
H_list = io.loadmat('./data/H_list.mat')['H_list']

print('Data loaded, generating dataset for learning')

# keep part of the data
H_list = H_list[:200000]

data_num = len(H_list)

# add LS channel estimation error here, channel with unit energy
LS_noise = np.sqrt(1/2*sigma_2)*(np.random.randn(data_num,Nt)+1j*np.random.randn(data_num,Nt))        
H_list_noisy = H_list+LS_noise



def DFT_matrix(N):
    F = np.zeros((N,N))+1j*np.zeros((N,N))
    # DFT矩阵每一行对应将角度的sin值进行N等分
    for i in range(-int(N/2),int(N/2)):
        for k in range(N): 
            F[i+int(N/2),k] = np.exp(-1j*2*np.pi*(i+1/2)*k/N)/np.sqrt(N)
    return F

F = DFT_matrix(Nt)

# transform channels to angular domain
H_list_angular = np.transpose(F.dot(np.transpose(H_list)))
H_list_noisy_angular = np.transpose(F.dot(np.transpose(H_list_noisy)))

# use the original antenna domain channel
#H_list_angular = H_list
#H_list_noisy_angular = H_list_noisy


H_list_angular = np.expand_dims(H_list_angular,axis=-1)
H_list_noisy_angular = np.expand_dims(H_list_noisy_angular,axis=-1)

dataset = np.concatenate([np.real(H_list_noisy_angular),np.imag(H_list_noisy_angular)],axis=-1)
labelset = np.concatenate([np.real(H_list_angular),np.imag(H_list_angular)],axis=-1)

print(dataset.shape)
print(labelset.shape)

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

def est_net(lr,Nt,net,mode):

    channel = Input(shape=(Nt,2))

    if net == 'CNN':

        x = Conv1D(filters=96,kernel_size=7,padding='same')(channel)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if mode=='naive':
            x = naive_attention(x)
        x = Conv1D(filters=96,kernel_size=5,padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # comment if only use single attention layer 
        #if mode=='naive':
        #    x = naive_attention(x) 
        refined_channel = Conv1D(filters=2,kernel_size=3,padding='same')(x)

    if net == 'CNN_deep':

        x = Conv1D(filters=96,kernel_size=7,padding='same')(channel)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if mode=='naive':
            x = naive_attention(x)

        x = Conv1D(filters=96,kernel_size=5,padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if mode=='naive':
            x = naive_attention(x)

        x = Conv1D(filters=96,kernel_size=5,padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if mode=='naive':
            x = naive_attention(x)

        x = Conv1D(filters=96,kernel_size=5,padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if mode=='naive':
            x = naive_attention(x)

        refined_channel = Conv1D(filters=2,kernel_size=1,padding='same')(x)

    if net == 'CNN_wide':

        x = Conv1D(filters=192,kernel_size=7,padding='same')(channel)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if mode=='naive':
            x = naive_attention(x)

        x = Conv1D(filters=192,kernel_size=5,padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        refined_channel = Conv1D(filters=2,kernel_size=3,padding='same')(x)

    if net == 'res_CNN':

        x = Conv1D(filters=96,kernel_size=7,padding='same')(channel)
        x = BatchNormalization()(x)
        x_init = Activation('relu')(x)
        if mode == 'naive':
            x_init = naive_attention(x_init)
        for i in range(3): # number of residual blocks
            x = Conv1D(filters=96,kernel_size=5,padding='same')(x_init)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv1D(filters=96,kernel_size=3,padding='same')(x)
            x = BatchNormalization()(x)
            if mode=='naive':
                x = naive_attention(x)
            x_init = Add()([x,x_init])
            x_init = Activation('relu')(x_init)
        
        refined_channel = Conv1D(filters=2,kernel_size=3,padding='same')(x_init)


    if net=='FNN':
        x1 = Flatten()(channel)
        x1 = Dense(512,activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dense(1024,activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dense(256,activation='relu')(x1)
        x1 = BatchNormalization()(x1)

        refined_channel = Flatten()(x1)
        refined_channel = Reshape((Nt,2))(refined_channel) 

    model = Model(inputs=channel,outputs=refined_channel)
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    model.summary()
    
    return model

lr = 1e-2
epochs = 1000
batch_size = 500

net = 'CNN_deep'
mode = 'naive'

if net=='FNN':
    mode = 'no'

best_model_path = './models/best_%s_attention_angular_%s_%d_dB.h5'%(mode,net,SNR)

checkpointer = ModelCheckpoint(best_model_path,verbose=1,save_best_only=True,save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10,verbose=1, mode='auto',min_delta=0.00001,min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss',min_delta=0.00001,patience=25)

model = est_net(lr,Nt,net,mode)

model.fit(dataset,labelset,epochs=epochs,batch_size=batch_size,callbacks=[checkpointer,reduce_lr,early_stopping],validation_split=0.2)

