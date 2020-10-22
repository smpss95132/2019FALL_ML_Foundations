#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import matplotlib.pyplot as plt


# In[2]:


# get sign
def get_sign(x):
    if x>0 :
        return 1.0
    else:
        return -1.0


# In[3]:


# calculate error rate
def evaluate_error_rate(feature,label,weight):
    test_size=feature.shape[0]
    test_result=np.matmul(feature,weight).reshape(test_size,1)
    compare=np.multiply(test_result,label)

    error_count=0
    
    for i in range(test_size):
        if compare[i,0]<0 or (compare[i,0]==0 and label[i,0]==1):
            error_count=error_count+1
                
    return error_count/test_size


# In[4]:


# test whether the specific index is correct
#def correctness(temp_train_feature,temp_train_label,current_weight,index):
#    inner_product=np.dot(temp_train_feature[index,:],current_weight)[0]
#
#    if inner_product*temp_train_label[index]>0 or (inner_product==0 and temp_train_label[index]==-1):
#        return True
#    else:
#        return False


# In[5]:


# read train data from txt 
train_set=np.loadtxt('https://www.csie.ntu.edu.tw/~htlin/course/mlfound19fall/hw1/hw1_7_train.dat')
num_train_data=train_set.shape[0]
append_one=np.ones(shape=(num_train_data,1))
train_feature=train_set[:,0:4].astype(float)
train_feature=np.concatenate([append_one,train_feature],axis=1)
print(train_feature.shape)
train_label=train_set[:,4:5].astype(float)
print(train_label.shape)


# In[6]:


# read test data from txt 
test_set=np.loadtxt('https://www.csie.ntu.edu.tw/~htlin/course/mlfound19fall/hw1/hw1_7_test.dat')
test_feature=test_set[:,0:4].astype(float)
test_feature=np.concatenate([append_one,test_feature],axis=1)
print(test_feature.shape)
test_label=test_set[:,4:5].astype(float)
print(test_label.shape)


# In[7]:


# pocket algo(base) pocket algo

num_train_data=train_feature.shape[0]
train_time=1126
update_time=100
error_list=[]

for i in range(train_time):
    print("model: ",i)
    # random the training order
    random_train_list=random.sample(range(num_train_data), num_train_data)
    
    # get the train feature
    temp_train_feature=train_feature[random_train_list]
    
    # get the label
    temp_train_label=train_label[random_train_list]
    
    # initialize 
    current_weight=np.zeros(shape=(5,1))
    temp_weight=np.zeros(shape=(5,1))
    
    # use to check whether the data is linear sperable
    finish_flag=False
    finish_checking=-1
    
    current_example=0
    next_example=0
    num_update=0
    
    while not finish_flag:
        inner_product=np.dot(temp_train_feature[current_example,:],temp_weight)[0]
        next_example=(current_example+1)%num_train_data
        
        if get_sign(inner_product)*temp_train_label[current_example]>0 or (inner_product==0 and temp_train_label[current_example]==-1):
            # the data is linear seperable
            if current_example==finish_checking:
                finish_flag=True
        else:
            temp_weight=temp_weight+(temp_train_label[current_example]*temp_train_feature[current_example]).reshape(5,1)
            finish_checking=current_example
            if evaluate_error_rate(temp_train_feature, temp_train_label,current_weight)>evaluate_error_rate(temp_train_feature, temp_train_label,temp_weight):
                current_weight=temp_weight
            num_update=num_update+1
            
        current_example=next_example
        
        if num_update==update_time:
            break
    
    error_list.append(evaluate_error_rate(test_feature,test_label,current_weight))
    
        

 


# In[8]:


print(sum(error_list)/len(error_list))

error_list = np.array(error_list)
plt.hist(error_list, bins=100)
plt.gca().set(xlabel="error rate", ylabel='Frequency');

