#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import matplotlib.pyplot as plt


# In[2]:


def Data_generator(size,noise):
    x = np.random.uniform(-1,1,size)
    
    y = np.sign(x)   
    choose_noise = np.random.uniform(0,1,size)  
    noise_position = (choose_noise > (1.0-noise))
    y[noise_position] = y[noise_position] * (-1)

    return x,y


# In[3]:


def Evaluate_Eout(theta,s):
    if s == 1:
        return 0.5 + 0.3 * (abs(theta) - 1)
    else:
        return 0.5 - 0.3 * (abs(theta) - 1)


# In[4]:


# hypothesis set: h(x) = s · sign(x − θ), find best θ and s
def Decision_stump_algorithm(x,y):
    sorted_x = np.sort(x)
    one = np.ones(1)
    first_x = np.concatenate((-1 * one,sorted_x))
    second_x = np.concatenate((sorted_x,one))
    
    cut_point_list = (first_x + second_x)/2
    sign_list = [1,-1]
    
    best_theta = cut_point_list[0]
    best_sign = sign_list[0] 
    best_error_count = x.shape[0]
    
    for i in range(len(sign_list)):
        sign = sign_list[i]
        for j in range(cut_point_list.shape[0]):
            theta = cut_point_list[j]
            result = np.sign(x - theta)
            result[result == 0] = sign
            result = result * sign
            
            error_count = np.sum(result != y)
            
            if error_count < best_error_count:
                best_error_count = error_count
                best_theta = theta
                best_sign = sign
    
    Eout = Evaluate_Eout(best_theta,best_sign)
    
    return (best_error_count/len(x)),Eout


# In[5]:


total_Ein = 0
total_Eout = 0

for i in range(5000):
    x,y = Data_generator(20,0.2)
    tempEin,tempEout = Decision_stump_algorithm(x,y)

    total_Ein += tempEin
    total_Eout += tempEout
    
print(total_Ein/5000)
print(total_Eout/5000)


# In[6]:


# Homework7
Err_diff_list = []

for i in range(1000):
    x,y = Data_generator(20,0.2)
    tempEin,tempEout = Decision_stump_algorithm(x,y)
    
    Err_diff_list.append(tempEin-tempEout)
    
Err_diff_np = np.array(Err_diff_list)

plt.hist(Err_diff_np, bins=100)
plt.gca().set(xlabel="Ein-Eout", ylabel='quantity');


# In[7]:


# Homework8
Err_diff_list = []

for i in range(1000):
    x,y = Data_generator(2000,0.2)
    tempEin,tempEout = Decision_stump_algorithm(x,y)
    
    Err_diff_list.append(tempEin-tempEout)

Err_diff_np = np.array(Err_diff_list)
plt.hist(Err_diff_np, bins=100)
plt.gca().set(xlabel="Ein-Eout", ylabel='quantity');

