import numpy as np
np.random.seed(2020)
import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#tf.random.set_seed(2020)
from scipy import io

#%% system parameters
Nt = 128
p = 1
SNR = 20 # dB
SNR_linear = 10**(SNR/10) 
sigma_2 = p/SNR_linear

RF_chain_ratio = 1/4
R = int(RF_chain_ratio*Nt)

# load data
H_list = io.loadmat('./data/H_list.mat')['H_list']
mean_AoAs = io.loadmat('./data/H_list.mat')['mean_AoA']

mean_AoAs = np.expand_dims(np.squeeze(mean_AoAs),-1)

print('Data loaded, generating dataset for learning')

# keep part of the data
H_list = H_list[-20000:]
mean_AoAs = mean_AoAs[-20000:]

data_num = len(H_list)

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
    #% save data for matlab VBI test
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
    x2 = Dense(num_neurons,activation='relu',use_bias=False)(x1)
    attention_map = Dense(num_filter,activation='sigmoid',name='att_map',use_bias=False)(x2)
    
    # feature recalibration
    x = Multiply()([x,attention_map])

    return x



def feature_attention(x,original_input,i):
    '''
        squeeze and excitation
    '''
    reduction_ratio = 1
    num_filter = int(x.shape[-1])
    num_neurons = num_filter//reduction_ratio
    
    # squeeze
    x1 = GlobalAvgPool1D()(x)  

    # concatenate with extracted features from the original input
    #original_input = Flatten()(original_input)
    #features = Dense(32,activation='relu')(original_input)
    #features = Dense(3,activation='linear')(features)
    #x1 = Concatenate()([x1,features])  


    # conv layers for feature extraction
    features = Conv1D(filters=16,kernel_size=7,padding='same')(original_input)
    features = BatchNormalization()(features)
    features = Activation('relu')(features)
    features = Conv1D(filters=4,kernel_size=5,padding='same')(features)
    features = BatchNormalization()(features)
    features = Activation('relu')(features)
    features = Flatten()(features)
    features = Dense(32,activation='linear')(features)
    
    # attention map prediction
    x1 = Concatenate()([x1,features])  
    x2 = Dense(num_neurons,activation='relu',name='att_input_%d'%i)(x1)
    attention_map = Dense(num_filter,activation='sigmoid',name='att_map_%d'%i)(x2)
    
    # feature recalibration
    x = Multiply()([x,attention_map])

    return x

def est_net(lr,Nt,net,mode):

    # Second part, channel estimator
    x0 = Input(shape=(R,2))

    if net=='CNN':

        # change the order
        #x1 = Permute((2,1))(x0)
        #x1 = Flatten()(x1)
        #x1 = Reshape((R,2))(x1)

        x1 = Conv1D(filters=96,kernel_size=7,padding='same')(x0)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        if mode == 'naive':
            x1 = naive_attention(x1)
        if mode == 'feature':
            x1 = feature_attention(x1,x0,1)     
 
        x2 = Conv1D(filters=96,kernel_size=5,padding='same')(x1)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
        if mode == 'naive':
            x2 = naive_attention(x2)
        if mode == 'feature':
            x2 = feature_attention(x2,x0,2) 

        x3 = Flatten()(x2)

    if net=='CNN_circular':
        # expand the input data
        kernel_size1 = 7
        x0 = Concatenate(axis=-2)([x0[:,-(kernel_size1//2-1):],x0,x0[:,:kernel_size1//2+1]])
        x1 = Conv1D(filters=96,kernel_size=kernel_size1,padding='valid')(x0)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        if mode == 'naive':
            x1 = naive_attention(x1)
 
        kernel_size2 = 7
        x1 = Concatenate(axis=-2)([x1[:,-(kernel_size2//2-1):],x1,x1[:,:kernel_size2//2+1]])    
    
        x2 = Conv1D(filters=96,kernel_size=kernel_size2,padding='valid')(x1)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
        if mode == 'naive':
            x2 = naive_attention(x2)

        x3 = Flatten()(x2)

    if net == 'res_CNN':
        x = Conv1D(filters=96,kernel_size=7,padding='same')(x0)
        x = BatchNormalization()(x)
        x_init = Activation('relu')(x)
        if mode=='naive':
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

        x2 = Conv1D(filters=96,kernel_size=3,padding='same')(x_init)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
        if mode =='naive':
            x2 = naive_attention(x2)

        x3 = Flatten()(x2)        


    if net=='FNN':
        x1 = Flatten()(x0)
        x1 = Dense(256,activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dense(512,activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dense(256,activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        x3 = Flatten()(x1)

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

    prediction = Dense(2*Nt,activation='linear')(x3)
    model = Model(inputs=x0,outputs=prediction)
    model.compile(loss='mse',optimizer=Adam(lr=lr))
    model.summary()
    return model


lr = 1e-3
epochs = 1000
batch_size = 500

net = 'FNN2'

mode = 'naive'

if net=='FNN':
    mode = 'no'

best_model_path = './models/best_%s_attention_%s_%d_dB.h5'%(mode,net,SNR)

checkpointer = ModelCheckpoint(best_model_path,verbose=1,save_best_only=True,save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10,verbose=1, mode='auto',min_delta=0.00001,min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss',min_delta=0.00001,patience=25)

model = est_net(lr,Nt,net,mode)

#%%
model.load_weights(best_model_path)

for i in range(len(model.layers)):
    layer = model.layers[i]
    if layer.name == 'att_map':
        count1 = i  

get_att_map1 = K.function([model.input],[model.layers[count1].output])

#%% 
att_map1 = get_att_map1([dataset])[0]

def get_partial_att_map(att_map,mean_AoAs,min_value,max_value):
    partial_indexs = np.where((np.sin(mean_AoAs*2*np.pi) > min_value) & (np.sin(mean_AoAs*2*np.pi)<max_value))[0]
    partial_att_map = att_map[partial_indexs]    
    return partial_att_map


from matplotlib import pyplot as plt

num_filter = 192
filters = np.arange(16,49)

min_value1 = 0.15
max_value1 = 0.2

min_value2 = 0.2
max_value2 = 0.25

min_value3 = 0.75
max_value3 = 0.8

lw = 1

#%% Att_map_1
att_map1 = get_att_map1([dataset])[0]
partial_att_map11 = get_partial_att_map(att_map1,mean_AoAs,min_value=min_value1,max_value=max_value1)
partial_att_map12 = get_partial_att_map(att_map1,mean_AoAs,min_value=min_value2,max_value=max_value2)
partial_att_map13 = get_partial_att_map(att_map1,mean_AoAs,min_value=min_value3,max_value=max_value3)

ax=plt.subplot(111)
plt.xlim(16,48)
plt.ylim(0,1)
plt.xlabel('Channel index')
plt.ylabel('Scale factor')
#plt.plot(filters,np.mean(att_map1,axis=0))
plt.plot(filters,np.mean(partial_att_map11,axis=0)[16:49],'r-',linewidth=lw)
plt.plot(filters,np.mean(partial_att_map12,axis=0)[16:49],'y--',linewidth=lw)
plt.plot(filters,np.mean(partial_att_map13,axis=0)[16:49],'b-.',linewidth=lw)

legends = ['(%.2f,%.2f)'%(min_value1,max_value1),'(%.2f,%.2f)'%(min_value2,max_value2),'(%.2f,%.2f)'%(min_value3,max_value3)]

plt.legend(legends,loc='upper right')
#plt.grid()

plt.xticks([16,20,24,28,32,36,40,44,48])

plt.axis([16,48,0,1])

plt.savefig('F:/research/AI+WC/attention/paper_version_20210203/CS_att_map.eps') 

#%% plot att maps with close mean AoA
plt.plot(filters,partial_att_map11[0,16:49],'r-',linewidth=lw)
plt.plot(filters,partial_att_map11[16,16:49],'b--',linewidth=lw)
plt.legend([r'Sample 1 with $\bar{\theta}=90.69^\circ$',r'Sample 2 with $\bar{\theta}=90.70^\circ$'],loc='upper right')
#plt.grid()

plt.xlabel('Channel index')
plt.ylabel('Scale factor')

plt.xticks([16,20,24,28,32,36,40,44,48])

plt.axis([16,48,0,1])

plt.savefig('F:/research/AI+WC/attention/paper_version_20210203/CS_single_att_map.eps') 

