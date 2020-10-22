#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import random
import matplotlib.pyplot as plt


# In[18]:


# get sign
def get_sign(x):
    if x>0 :
        return 1.0
    else:
        return -1.0


# In[19]:


# read data from URL

train_set=np.loadtxt('https://www.csie.ntu.edu.tw/~htlin/course/mlfound19fall/hw1/hw1_6_train.dat')
train_feature=train_set[:,0:4].astype(float)
print(train_feature.shape)
train_label=train_set[:,4:5].astype(float)
print(train_label.shape)


# In[20]:


# PLA algo

num_train_data=train_feature.shape[0]
train_time=1126
learn_rate=1
epoch_list=[]

append_one=np.ones(shape=(num_train_data,1),dtype=float)

for i in range(train_time):
    print("model: ",i)
    # random the training order
    random_train_list=random.sample(range(num_train_data), num_train_data)
    
    # get the train feature
    temp_train_feature=train_feature[random_train_list]
    temp_train_feature=np.concatenate([append_one,temp_train_feature],axis=1)
    
    # get the label
    temp_train_label=train_label[random_train_list]

    
    # initialize 
    weight=np.zeros(shape=(5,1))
    epoch_count=0
    
    finish_flag=False
    finish_checking=-1
    current_example=0
    next_example=0
    
    while not finish_flag:
        inner_product=np.dot(temp_train_feature[current_example,:],weight)[0]
        next_example=(current_example+1)%num_train_data
        
        # correct 
        if get_sign(inner_product)*temp_train_label[current_example]>0 or (inner_product==0 and temp_train_label[current_example]==-1):
            if current_example==finish_checking:
                finish_flag=True
        # incorrect
        else:
            weight=weight+learn_rate*(temp_train_label[current_example]*temp_train_feature[current_example]).reshape(5,1)
            finish_checking=current_example
            epoch_count=epoch_count+1
            
        current_example=next_example
        
    
    epoch_list.append(epoch_count)




# In[21]:


total=sum(epoch_list)
print(total/train_time)

epoch_list = np.array(epoch_list)
plt.hist(epoch_list, bins=100)
plt.gca().set(xlabel = "number of updates",ylabel='Frequency');

