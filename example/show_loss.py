import matplotlib.pyplot as plt
import numpy as np

dgd_acc = []
dgd_loss = []
dzo_acc_1 =[]
dzo_acc_5 =[]
dzo_acc_10 = []
dzo_acc_20 = []
dzo_acc_40 = []
dzo_acc_f = []
dzo_loss_1 = []
dzo_loss_5 = []
dzo_loss_10 = []
dzo_loss_20 = []
dzo_loss_40 = []
dzo_loss_f=[]
epoch = np.arange(2000)
    
f = open("/home/qyq/bfl/results/num_perturb=1_accuracy.txt","r")
lines = f.readlines()
for line in lines:
    dzo_acc_1.append(float(line.replace('\n', ''))*100)
    
f = open("/home/qyq/bfl/results/num_perturb=5_accuracy.txt","r")
lines = f.readlines()
for line in lines:
    dzo_acc_5.append(float(line.replace('\n', ''))*100)

f = open("/home/qyq/bfl/results/perturb10_accuracy.txt","r")
lines = f.readlines()
for line in lines:
    dzo_acc_10.append(float(line.replace('\n', ''))*100)
    
f = open("/home/qyq/bfl/results/perturb20_accuracy.txt","r")
lines = f.readlines()
for line in lines:
    dzo_acc_20.append(float(line.replace('\n', ''))*100)

f = open("/home/qyq/bfl/results/perturb40_accuracy.txt","r")
lines = f.readlines()
for line in lines:
    dzo_acc_40.append(float(line.replace('\n', ''))*100)
    
f = open("/home/qyq/bfl/results/forward_accuracy.txt","r")
lines = f.readlines()
for line in lines:
    dzo_acc_f.append(float(line.replace('\n', ''))*100)

f = open("/home/qyq/bfl/results/num_perturb=1_loss.txt","r")
lines = f.readlines()
for line in lines:
    dzo_loss_1.append(float(line.replace('\n', '')))
    
f = open("/home/qyq/bfl/results/num_perturb=5_loss.txt","r")
lines = f.readlines()
for line in lines:
    dzo_loss_5.append(float(line.replace('\n', '')))

f = open("/home/qyq/bfl/results/perturb10_loss.txt","r")
lines = f.readlines()
for line in lines:
    dzo_loss_10.append(float(line.replace('\n', '')))
    
f = open("/home/qyq/bfl/results/perturb20_loss.txt","r")
lines = f.readlines()
for line in lines:
    dzo_loss_20.append(float(line.replace('\n', '')))

f = open("/home/qyq/bfl/results/perturb40_loss.txt","r")
lines = f.readlines()
for line in lines:
    dzo_loss_40.append(float(line.replace('\n', '')))

f = open("/home/qyq/bfl/results/forward_loss.txt","r")
lines = f.readlines()
for line in lines:
    dzo_loss_f.append(float(line.replace('\n', '')))


# plt.plot(epoch,dzo_loss_1,color='yellow',label='perturbation=1')
plt.plot(epoch,dzo_loss_5,color='pink',label='perturbation=5')
plt.plot(epoch,dzo_loss_10,color='red',label='perturbation=10')
plt.plot(epoch,dzo_loss_20,color='blue',label='perturbation=20')
plt.plot(epoch,dzo_loss_40,color='green',label='perturbation=40')
plt.plot(epoch,dzo_loss_f,color='orange',label='forward,perturb=5')
plt.legend()  #显示上面的label
plt.xlabel('epochs') #x_label
plt.ylabel('train_loss')#y_label
plt.ylim(0,0.5)
plt.savefig("/home/qyq/bfl/results/loss.png")

    
    