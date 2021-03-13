import numpy as np
np.random.seed(2020)
from scipy import io

#%% system parameters
Nt = 192
p = 1
SNR = 20 # dB
SNR_linear = 10**(SNR/10) 
sigma_2 = p/SNR_linear

# load data
dataset = io.loadmat('./data/H_list_192_5.mat')
H_list_all = dataset['H_list']
mean_AoAs = dataset['mean_AoA']
mean_AoAs = np.expand_dims(np.squeeze(mean_AoAs),-1)
print('Data loaded!')

LS_noise = np.sqrt(1/2*sigma_2)*(np.random.randn(len(H_list_all),Nt)+1j*np.random.randn(len(H_list_all),Nt))        
channels_noisy_all = H_list_all + LS_noise

# split the training set and testing set
H_list_all_test = H_list_all[-int(0.2*len(H_list_all)):]
channels_noisy_all_test = channels_noisy_all[-int(0.2*len(channels_noisy_all)):]

H_list_all_train = H_list_all[:-int(0.2*len(H_list_all))]
channels_noisy_all_train = channels_noisy_all[:-int(0.2*len(channels_noisy_all))]

mean_AoAs_test = mean_AoAs[-int(0.2*len(mean_AoAs)):]
mean_AoAs_train = mean_AoAs[:-int(0.2*len(mean_AoAs))]

def MMSE(H_list,H_list_test,channels_noisy,channels_noisy_test,group,groups,Nt,train_num,test_num):
    RHH_hat = np.zeros((Nt,Nt),dtype=complex)
    RH_hat2 = np.zeros((Nt,Nt),dtype=complex)
    RHH = np.zeros((Nt,Nt),dtype=complex)

    for i in range(train_num):
        if i%1000==0:
            print('CCM  for  Group %d/%d, %d/%d'%(group,groups,i,train_num))
        RHH_hat = RHH_hat + np.expand_dims(H_list[i],axis=1).dot(np.expand_dims(np.conjugate(channels_noisy[i]),axis=0))
        RH_hat2 = RH_hat2 + np.expand_dims(channels_noisy[i],axis=1).dot(np.expand_dims(np.conjugate(channels_noisy[i]),axis=0))
        RHH = RHH + np.expand_dims(H_list[i],axis=1).dot(np.expand_dims(np.conjugate(H_list[i]),axis=0))
            
    RHH_hat = RHH_hat/train_num
    RH_hat2 = RH_hat2/train_num
    RHH = RHH/train_num
    
    W1 = RHH_hat.dot(np.linalg.inv(RH_hat2))
    
    # another kind of way to compute W
#    W2 = RHH.dot(np.linalg.inv(RHH+1/SNR_linear*np.eye(Nt)))
#    W3 = RHH_hat.dot(np.linalg.inv(RHH+1/SNR_linear*np.eye(Nt)))
    
    # modify the LS estimation
    mmse_est = np.zeros((test_num,Nt),dtype=complex)
    for i in range(test_num):
        if i%200==0:
            print('Calibrate Group %d/%d, %d/%d'%(group,groups,i,test_num))
        mmse_est[i] = np.squeeze(W1.dot(np.expand_dims(channels_noisy_test[i],axis=-1)))
    
    # MMSE performance
    MMSE_mse = np.linalg.norm(mmse_est-H_list_test)**2/np.product(H_list_test.shape)
    
    return MMSE_mse

delta = 3/360 # 间隔
groups = int(1/delta)

mse_list = []
start = 0
for group in range(groups):
    end = start + delta
    # only keep a part of channel
    indexs_train = np.where((mean_AoAs_train>=start) & (mean_AoAs_train<=end))[0]
    indexs_test = np.where((mean_AoAs_test>=start) & (mean_AoAs_test<=end))[0]
    H_list = H_list_all[indexs_train]
    H_list_test = H_list_all_test[indexs_test]
    channels_noisy = channels_noisy_all[indexs_train]
    channels_noisy_test = channels_noisy_all_test[indexs_test]
    train_num = len(H_list)
    test_num = len(H_list_test)
    mse = MMSE(H_list,H_list_test,channels_noisy,channels_noisy_test,group,groups,Nt,train_num,test_num)
    mse_list.append(mse)
    
    start = start + delta

MMSE_mse = np.mean(mse_list)
print('MMSE mse is %.5f'%MMSE_mse)