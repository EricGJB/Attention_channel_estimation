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
dataset = io.loadmat('./data/H_list_128_5.mat')
H_list = dataset['H_list']
mean_AoAs = dataset['mean_AoA']

mean_AoAs = np.expand_dims(np.squeeze(mean_AoAs),-1)

print('Data loaded, generating dataset for learning')

# only keep a part of channel
indexs = np.where((mean_AoAs>0.3) & (mean_AoAs<0.31))[0]
H_list = H_list[indexs]

# Rayleigh channel
#H_list = np.sqrt(1/2)*(np.random.randn(20000,Nt)+1j*np.random.randn(20000,Nt)) 

data_num = len(H_list)

print('Select %d channel samples'%data_num)

LS_noise = np.sqrt(1/2*sigma_2)*(np.random.randn(data_num,Nt)+1j*np.random.randn(data_num,Nt))        
channels_noisy = H_list + LS_noise

# LS performance
LS_mse = np.linalg.norm(channels_noisy-H_list)**2/np.product(H_list.shape)


#%% MMSE
# estimate the correlation matrix
RHH_hat = np.zeros((Nt,Nt),dtype=complex)
RH_hat2 = np.zeros((Nt,Nt),dtype=complex)
RHH = np.zeros((Nt,Nt),dtype=complex)

for i in range(data_num):
    if i%1000==0:
        print('%d/%d'%(i,data_num))
    RHH_hat = RHH_hat + np.expand_dims(H_list[i],axis=1).dot(np.expand_dims(np.conjugate(channels_noisy[i]),axis=0))
    RH_hat2 = RH_hat2 + np.expand_dims(channels_noisy[i],axis=1).dot(np.expand_dims(np.conjugate(channels_noisy[i]),axis=0))
    RHH = RHH + np.expand_dims(H_list[i],axis=1).dot(np.expand_dims(np.conjugate(H_list[i]),axis=0))
    
RHH_hat = RHH_hat/data_num
RH_hat2 = RH_hat2/data_num
RHH = RHH/data_num

W1 = RHH_hat.dot(np.linalg.inv(RH_hat2))

# another kind of way to compute W
W2 = RHH.dot(np.linalg.inv(RHH+1/SNR_linear*np.eye(Nt)))
W3 = RHH_hat.dot(np.linalg.inv(RHH+1/SNR_linear*np.eye(Nt)))

# modify the LS estimation
mmse_est = np.zeros((data_num,Nt),dtype=complex)
for i in range(data_num):
    if i%1000==0:
        print('%d/%d'%(i,data_num))
    mmse_est[i] = np.squeeze(W1.dot(np.expand_dims(channels_noisy[i],axis=-1)))

# MMSE performance
MMSE_mse = np.linalg.norm(mmse_est-H_list)**2/np.product(H_list.shape)

print('LS mse is %.4f'%LS_mse)
print('MMSE mse is %.5f'%MMSE_mse)


