#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt


# In[2]:


# evaluate the accuracy of the weight
# input: accuracy(nparray,nparray,nparray)
# output: float
def accuracy(x,y,weight):
    prediction = np.matmul(x,weight)
    prediction = np.sign(prediction)
    number_of_correct = np.sum(prediction==y)
    total_number = prediction.shape[0]
    
    return number_of_correct/total_number


# In[3]:


# read data from the specific path
# input: get_data(string)
# output: nparray
def get_data(path):
    # read train data from txt 
    train_set = np.loadtxt(path)
    #print(train_set.shape)
    num_train_data = train_set.shape[0]
    num_dimension = train_set.shape[1] - 1 
    
    append_one = np.ones(shape=(num_train_data,1))
    
    train_feature = train_set[:,0:num_dimension].astype(float)
    train_feature = np.concatenate([append_one,train_feature],axis=1)
    print(train_feature.shape)
    train_label=train_set[:,num_dimension:num_dimension+1].astype(float)
    print(train_label.shape)
    
    return train_feature,train_label


# In[4]:


# sigmoid function
# input: sigmoid(int/float)
# output: float
def sigmoid(x):
    return (1.0 / (1.0 + math.exp(-x)))


# In[5]:


# sigmoid function for numpy
# input: sigmoid_matrix(nparray)
# output: nparray
def sigmoid_matrix(M):
    return (1.0/(1.0 + np.exp(-M)))


# In[6]:


# count gradient for stochastic gradient descent
# input: stochastic_grad(nparray,nparray,nparray,int)
# output: nparray
def stochastic_grad(x,y,weight,index):
    dim = x.shape[1]
    example = x[index,:].reshape(1,dim)
    gradient = sigmoid((-1)*y[index][0]*np.matmul(example,weight))*((-1)*np.transpose(example)*y[index][0])
    
    return gradient


# In[7]:


# count geadient for gradient descent
# input: stochastic_grad(nparray,nparray,nparray)
# output: nparray
def batch_grad_parallel(x,y,weight):
    N = x.shape[0]
    dim = x.shape[1]
    
    # sigmoid裡面:N*1   sigmoid右邊:N*d
    gradient = np.sum(sigmoid_matrix((-1)*y*np.matmul(x,weight)) * ((-1)*x*y),axis=0).reshape(dim,1)
    
    gradient = gradient/N
    
    return gradient  


# In[8]:


# main function of logistic regression
# input: logistic_regression(nparray,nparray,float,int,boolean,nparray,nparray)
# output: nparray,list,list
def logistic_regression(feature,label,lr,epoch,stochastic,test_feature,test_label):
    # initialize weight
    weight = np.zeros(shape=(feature.shape[1],1),dtype = float)
    train_acc = []
    test_acc = []
    for i in range(epoch):
        if i%100==0:
            print(i,"/",epoch)
        if stochastic:
            gradient = stochastic_grad(feature,label,weight,i%feature.shape[0])
        else:
            gradient = batch_grad_parallel(feature,label,weight)
            
        weight = weight - (lr*gradient)
        train_acc.append(1-accuracy(feature,label,weight))
        test_acc.append(1-accuracy(test_feature,test_label,weight))
        
    return weight,train_acc,test_acc


# In[9]:


train_feature,train_label = get_data("https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw3_train.dat")
test_feature,test_label = get_data("https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw3_test.dat")


# In[10]:


lr = 0.01
epoch = 2000


# In[11]:


#19
_,HW_7_1,HW8_1 = logistic_regression(train_feature,train_label,lr,epoch,False,test_feature,test_label)


# In[12]:


lr = 0.001
epoch = 2000


# In[13]:


#20
_,HW_7_2,HW8_2 = logistic_regression(train_feature,train_label,lr,epoch,True,test_feature,test_label)


# In[14]:


y_axis = list(range(epoch))


# In[15]:


# Homework7
plt.figure(figsize=(15,10),dpi=100,linewidth = 2)
plt.plot(y_axis,HW_7_1,'s-',color = 'r',label="Gradient descent")
plt.plot(y_axis,HW_7_2,'o-',color = 'g', label="Stochastic gradient descent")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# 標示x軸(labelpad代表與圖片的距離)
plt.xlabel("Epoch", fontsize=30, labelpad = 15)
# 標示y軸(labelpad代表與圖片的距離)
plt.ylabel("Error rate", fontsize=30, labelpad = 20)
# 顯示出線條標記位置
plt.legend(loc = "best", fontsize=20)
# 畫出圖片
plt.show()
plt.savefig('HW3_7.png')


# In[16]:


# Homework8
plt.figure(figsize=(15,10),dpi=100,linewidth = 2)
plt.plot(y_axis,HW8_1,'s-',color = 'r',label="Gradient descent")
plt.plot(y_axis,HW8_2,'o-',color = 'g', label="Stochastic gradient descent")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# 標示x軸(labelpad代表與圖片的距離)
plt.xlabel("Epoch", fontsize=30, labelpad = 15)
# 標示y軸(labelpad代表與圖片的距離)
plt.ylabel("Error rate", fontsize=30, labelpad = 20)
# 顯示出線條標記位置
plt.legend(loc = "best", fontsize=20)
# 畫出圖片
plt.show()
plt.savefig('HW3_8.png')

