import numpy as np
np.random.seed(2020)
import tensorflow as tf
from scipy import io


#%% system parameters
Nt = 128
p = 1
SNR = 20 # dB
SNR_linear = 10**(SNR/10) 
sigma_2 = p/SNR_linear

# load data
H_list = io.loadmat('./data/H_list.mat')['H_list']
mean_AoAs = io.loadmat('./data/H_list.mat')['mean_AoA']

mean_AoAs = np.expand_dims(np.squeeze(mean_AoAs),-1)

print('Data loaded, generating dataset for learning')

# keep part of the data
H_list = H_list[-int(0.1*len(H_list)):]
mean_AoAs = mean_AoAs[-int(0.1*len(mean_AoAs)):]

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


# original F matrix
#def DFT_matrix(N):
#    F = np.zeros((N,N))+1j*np.zeros((N,N))
#    # DFT矩阵每一行对应将角度的sin值进行N等分
#    for i in range(1,2*N,2):
#        for k in range(N): 
#            F[(i-1)//2,k] = np.exp(-1j*np.pi*i*k/N)/np.sqrt(N)
#    return F

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

def naive_attention(x,i):
    '''
        squeeze and excitation
    '''
    reduction_ratio = 1
    num_filter = int(x.shape[-1])
    num_neurons = num_filter//reduction_ratio
    
    # squeeze
    x1 = GlobalAvgPool1D()(x)  

    # attention map prediction
    x2 = Dense(num_neurons,activation='relu',use_bias=False,name='att_input_%d'%i)(x1)
    attention_map = Dense(num_filter,activation='sigmoid',use_bias=False,name='att_map_%d'%i)(x2)
    
    # feature recalibration
    x = Multiply()([x,attention_map])

    return x

def est_net(lr,Nt,net):

    trainable = True

    channel = Input(shape=(Nt,2))

    if net == 'CNN_deep':

        x = Conv1D(filters=96,kernel_size=7,padding='same')(channel)
        x = BatchNormalization(trainable=trainable)(x)
        x = Activation('relu')(x)
        x = naive_attention(x,1)
        x = Conv1D(filters=96,kernel_size=5,padding='same')(x)
        x = BatchNormalization(trainable=trainable)(x)
        x = Activation('relu')(x)
        x = naive_attention(x,2)
        x = Conv1D(filters=96,kernel_size=5,padding='same')(x)
        x = BatchNormalization(trainable=trainable)(x)
        x = Activation('relu')(x)
        x = naive_attention(x,3)
        x = Conv1D(filters=96,kernel_size=5,padding='same')(x)
        x = BatchNormalization(trainable=trainable)(x)
        x = Activation('relu')(x)
        x = naive_attention(x,4) 
        refined_channel = Conv1D(filters=2,kernel_size=1,padding='same')(x)


    model = Model(inputs=channel,outputs=refined_channel)
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    model.summary()
    
    return model

lr = 0.01
epochs = 1000
batch_size = 500

net = 'CNN_deep'

best_model_path = './models/best_naive_attention_angular_%s_%d_dB.h5'%(net,SNR)

checkpointer = ModelCheckpoint(best_model_path,verbose=1,save_best_only=True,save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10,verbose=1, mode='auto',min_delta=0.00001,min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss',min_delta=0.00001,patience=25)

model = est_net(lr,Nt,net)

#model.fit(dataset,labelset,epochs=epochs,batch_size=batch_size,callbacks=[checkpointer,reduce_lr,early_stopping],validation_split=0.2)


#%% observe learned attention map
model.load_weights(best_model_path)

for i in range(len(model.layers)):
    layer = model.layers[i]
    if layer.name == 'att_map_1':
        count1 = i
    if layer.name == 'att_map_2':
        count2 = i     
    if layer.name == 'att_map_3':
        count3 = i   
    if layer.name == 'att_map_4':
        count4 = i     

get_att_map1 = K.function([model.input],[model.layers[count1].output])
get_att_map2 = K.function([model.input],[model.layers[count2].output])
get_att_map3 = K.function([model.input],[model.layers[count3].output])
get_att_map4 = K.function([model.input],[model.layers[count4].output])

#%% 
att_map1 = get_att_map1([dataset])[0]
att_map2 = get_att_map2([dataset])[0]
att_map3 = get_att_map3([dataset])[0]
att_map4 = get_att_map4([dataset])[0]

def get_partial_att_map(att_map,mean_AoAs,min_value,max_value):
    partial_indexs = np.where((np.sin(mean_AoAs*2*np.pi) > min_value) & (np.sin(mean_AoAs*2*np.pi)<max_value))[0]
    partial_att_map = att_map[partial_indexs]    
    return partial_att_map


from matplotlib import pyplot as plt

num_filter = 96
filters = np.arange(16,49)

min_value1 = 0.15
max_value1 = 0.2

min_value2 = 0.2
max_value2 = 0.25

min_value3 = 0.75
max_value3 = 0.8

lw = 1

lap = 1

figsize = (6,2)

#%% Att_map_1
att_map1 = get_att_map1([dataset])[0]
partial_att_map11 = get_partial_att_map(att_map1,mean_AoAs,min_value=min_value1,max_value=max_value1)
partial_att_map12 = get_partial_att_map(att_map1,mean_AoAs,min_value=min_value2,max_value=max_value2)
partial_att_map13 = get_partial_att_map(att_map1,mean_AoAs,min_value=min_value3,max_value=max_value3)

plt.figure()
#plt.figure(figsize=(6,2),dpi=100)
plt.xlabel('Channel index')
plt.ylabel('Scale factor')
#plt.plot(filters,np.mean(att_map1,axis=0))
plt.plot(filters,np.mean(partial_att_map11,axis=0)[16:49],'r-',linewidth = lw+0.3)
plt.plot(filters,np.mean(partial_att_map12,axis=0)[16:49],'y--',linewidth = lw)
plt.plot(filters,np.mean(partial_att_map13,axis=0)[16:49],'b-.',linewidth = lw-0.3)

legends = ['(%.2f,%.2f)'%(min_value1,max_value1),'(%.2f,%.2f)'%(min_value2,max_value2),'(%.2f,%.2f)'%(min_value3,max_value3)]

plt.legend(legends,loc='upper right')
#plt.grid()

plt.xticks([16,20,24,28,32,36,40,44,48])

plt.axis([16,48,0,1])

plt.savefig('F:/research/AI+WC/attention/paper_version_20210203/CE_att_map1.eps')


#%% Att_map_3
att_map1 = get_att_map3([dataset])[0]
partial_att_map11 = get_partial_att_map(att_map1,mean_AoAs,min_value=min_value1,max_value=max_value1)
partial_att_map12 = get_partial_att_map(att_map1,mean_AoAs,min_value=min_value2,max_value=max_value2)
partial_att_map13 = get_partial_att_map(att_map1,mean_AoAs,min_value=min_value3,max_value=max_value3)

plt.figure()
#plt.figure(figsize=(6,2),dpi=100)
plt.xlabel('Channel index')
plt.ylabel('Scale factor')
#plt.plot(filters,np.mean(att_map1,axis=0))
plt.plot(filters,np.mean(partial_att_map11,axis=0)[16:49],'r-',linewidth = lw+0.3)
plt.plot(filters,np.mean(partial_att_map12,axis=0)[16:49],'y--',linewidth = lw)
plt.plot(filters,np.mean(partial_att_map13,axis=0)[16:49],'b-.',linewidth = lw-0.3)

legends = ['(%.2f,%.2f)'%(min_value1,max_value1),'(%.2f,%.2f)'%(min_value2,max_value2),'(%.2f,%.2f)'%(min_value3,max_value3)]

plt.legend(legends,loc='upper right')
#plt.grid()

plt.xticks([16,20,24,28,32,36,38,40,44,48])

plt.axis([16,48,0,1])

plt.savefig('F:/research/AI+WC/attention/paper_version_20210203/CE_att_map3.eps')


#%% Att_map_4
att_map1 = get_att_map4([dataset])[0]
partial_att_map11 = get_partial_att_map(att_map1,mean_AoAs,min_value=min_value1,max_value=max_value1)
partial_att_map12 = get_partial_att_map(att_map1,mean_AoAs,min_value=min_value2,max_value=max_value2)
partial_att_map13 = get_partial_att_map(att_map1,mean_AoAs,min_value=min_value3,max_value=max_value3)

plt.figure()
#plt.figure(figsize=(6,2),dpi=100)
plt.xlabel('Channel index')
plt.ylabel('Scale factor')
#plt.plot(filters,np.mean(att_map1,axis=0))
plt.plot(filters,np.mean(partial_att_map11,axis=0)[16:49],'r-',linewidth = lw+0.3)
plt.plot(filters,np.mean(partial_att_map12,axis=0)[16:49],'y--',linewidth = lw)
plt.plot(filters,np.mean(partial_att_map13,axis=0)[16:49],'b-.',linewidth = lw-0.3)

legends = ['(%.2f,%.2f)'%(min_value1,max_value1),'(%.2f,%.2f)'%(min_value2,max_value2),'(%.2f,%.2f)'%(min_value3,max_value3)]

plt.legend(legends,loc='upper right')
#plt.grid()

plt.xticks([16,20,24,28,32,36,40,44,48])

plt.axis([16,48,0,1])

plt.savefig('F:/research/AI+WC/attention/paper_version_20210203/CE_att_map4.eps')