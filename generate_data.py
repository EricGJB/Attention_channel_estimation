import numpy as np
np.random.seed(2020)
from scipy import io

# system parameters
Nt = 128
Lp = 20
AS = 5
# generate dataset
data_num = 200000
H_list = np.zeros((data_num,Nt))+1j*np.zeros((data_num,Nt))

mean_angle_list = np.random.uniform(0,1,data_num)*360
#mean_angle_list = np.random.choice([-48.59,-30,-14.48,0,14.48,30,48.59],data_num)

Gains_list = np.sqrt(1/2)*(np.random.randn(data_num,Lp)+1j*np.random.randn(data_num,Lp))
H_list = np.zeros((data_num,Nt))+1j*np.zeros((data_num,Nt))

for i in range(data_num):   
    if i%1000==0:
        print('%d/%d'%(i,data_num))
    mean_angle = mean_angle_list[i]
    DoAs = np.random.uniform(mean_angle-AS,mean_angle+AS,Lp)/180*np.pi
    #DoAs = np.random.uniform(0,360,Lp)/180*np.pi
    for lp in range(Lp):
        H_list[i] = H_list[i] + Gains_list[i,lp]*np.exp(1j*2*np.pi*1/2*np.arange(Nt)*np.sin(DoAs[lp]))    

H_list = H_list/np.sqrt(Lp)

print(H_list.shape)

io.savemat('./data/H_list.mat',{'H_list':H_list,'mean_AoA':mean_angle_list/360})

print('Data saved!')



