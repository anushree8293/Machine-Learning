#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


###################################Import of data#########################
#import data
file1="E:\\Study\\ML\\HW_1\\Dataset (2)\\Dataset\\Training\\Features_Variant_1.csv"
file2="E:\\Study\\ML\\HW_1\\Dataset (2)\\Dataset\\Training\\Features_Variant_5.csv"
file3="E:\Study\ML\HW_1\Dataset (2)\Dataset\Testing\TestSet\Test_Case_1.csv"
file4="E:\Study\ML\HW_1\Dataset (2)\Dataset\Testing\TestSet\Test_Case_2.csv"
file5="E:\Study\ML\HW_1\Dataset (2)\Dataset\Testing\TestSet\Test_Case_3.csv"
file6="E:\Study\ML\HW_1\Dataset (2)\Dataset\Testing\TestSet\Test_Case_4.csv"
file7="E:\Study\ML\HW_1\Dataset (2)\Dataset\Testing\TestSet\Test_Case_5.csv"
file8="E:\Study\ML\HW_1\Dataset (2)\Dataset\Testing\TestSet\Test_Case_6.csv"
file9="E:\Study\ML\HW_1\Dataset (2)\Dataset\Testing\TestSet\Test_Case_7.csv"
file10="E:\Study\ML\HW_1\Dataset (2)\Dataset\Testing\TestSet\Test_Case_8.csv"
file11="E:\Study\ML\HW_1\Dataset (2)\Dataset\Testing\TestSet\Test_Case_9.csv"
file12="E:\Study\ML\HW_1\Dataset (2)\Dataset\Testing\TestSet\Test_Case_10.csv"
head=["Page_Likes","Page_Checkins","Page talking about","Page Category","Derived5","Derived6","Derived7","Derived8",
          "Derived9","Derived10","Derived11","Derived12","Derived13","Derived14","Derived15","Derived16","Derived17",
          "Derived18","Derived19","Derived20","Derived21","Derived22","Derived23","Derived24","Derived25","Derived26","Derived27","Derived28","Derived29",
          "CC1","CC2","CC3","CC4","CC5","Base Time","Post_Length","Post Share Count","Post Promotion Status","H Local","PSun","PMon","PTue","PWed","PThurs","PFri","PSat",
          "BSun","BMon","BTue","BWed","BThurs","BFri","BSat","Target"]

training1 = pd.read_csv(file1,names=head)
training5 = pd.read_csv(file2,names=head)
test1 = pd.read_csv(file3,names=head)
test2 = pd.read_csv(file4,names=head)
test3 = pd.read_csv(file5,names=head)
test4 = pd.read_csv(file6,names=head)
test5 = pd.read_csv(file7,names=head)
test6 = pd.read_csv(file8,names=head)
test7 = pd.read_csv(file9,names=head)
test8 = pd.read_csv(file10,names=head)
test9 = pd.read_csv(file11,names=head)
test10 = pd.read_csv(file12,names=head)


# In[3]:


#combine into master data
frames = [training1, training5,test1,test2,test3,test4,test5,test6,test7,test8,test9,test10]
result = pd.concat(frames)


# In[4]:


# split master data into  test and train
msk = np.random.rand(len(result)) < 0.7
train_data = result[msk]
test_data = result[~msk]


# In[5]:


##################################################Parameter Setting and parameter scaling############################################

#Train parameter definition and standardization
X2=train_data.iloc[:, [0,1,29,30,31,32,34,35,36,38]]
X3 = (X2 - X2.mean()) / X2.std()
X4=train_data.iloc[:, [37,40,41,42,43,44,45]]
X=np.concatenate((X3,X4),axis=1)
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)


# In[6]:


#Train target definition
y = train_data.iloc[:,53:54].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray
#y=(y1-y1.mean())/y1.std()
theta = np.zeros([1,18])


# In[7]:


#Test parameter definition and standardization
testX2=test_data.iloc[:, [0,1,29,30,31,32,34,35,36,38]]
testX3 = (testX2 - testX2.mean()) / testX2.std()
testX4=test_data.iloc[:, [37,40,41,42,43,44,45]]
testX=np.concatenate((testX3,testX4),axis=1)
testones = np.ones([testX.shape[0],1])
testX = np.concatenate((testones,testX),axis=1)


# In[8]:


#Test target definition
testy = test_data.iloc[:,53:54].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray
#testy=(testy1-testy1.mean())/testy1.std()
testtheta = np.zeros([1,18])


# In[9]:


#set hyper parameters
alpha = [0.001,0.01,0.1,0.2,0.3]
iters = 1000
threshold=[0.000001,0.00001,0.0001,0.001,0.01,0.1]


# In[10]:


###########################Function Definition############################################

#predicted number of comments
def yhatcalc(testX,final_theta):
     return testX @ final_theta.T


# In[11]:


#computecost
def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))


# In[28]:


#gradientdescent
def gradientDescent(X, y, theta, iters, alpha,isThreshold, thresholdValue):
    cost = []
    for i in range(iters):
        theta = theta - (alpha / len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost.append(computeCost(X, y, theta))
        if  isThreshold==True and i!=0 and (cost[i - 1] - cost[i]) / cost[i - 1]<= thresholdValue :
            break
    return cost, theta, i+1


# In[15]:


#plotting
def plot(xvariable, yvariable, xaxis, yaxis,title,color):
    fig, ax = plt.subplots()
    ax.plot(xvariable, yvariable,color)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    ax.set_title(title)
    return fig


# In[17]:


############################################Execution################################################

print("=================================="+"Train data for learning rate"+"==================================")

#Running gradient descent for TRAINING data to observe optimum alpha without threshold
intialcost=computeCost(X,y,theta)
final_theta=[]
final_alpha=[]
final_iteration=[]
final_cost=[]
final_cost_collection=np.zeros(len(alpha))
for i in range(len(alpha)):
    c,t,j=gradientDescent(X,y,theta,iters,alpha[i],False,0)
    fig, ax = plt.subplots()
    ax.plot(np.arange(iters), c, 'r')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title('Error Vs Iterations for Learning rate ' + str(alpha[i]) + ' in Train')
    plt.show()
    costty=computeCost(X,y,t)
    print("For alpha:"+str(alpha[i])+" Mininum Cost="+str(costty))
    final_cost_collection[i]=costty
    if costty<intialcost:
        final_theta=t
        final_alpha=alpha[i]
        final_iteration=j
        final_cost=c
        intialcost=costty

train_alphaCost=plot(alpha, final_cost_collection,'Alpha','Cost','Learning Rate v/s Cost for Training Data','r')
train_alphaCost.show()


# In[18]:


#Running gradient descent for TEST data to observe optimum alpha without threshold

print("=================================="+"Test data for learning rate"+"==================================")

test_cost_collection=np.zeros(len(alpha))
for i in range(len(alpha)):
    test_cost,b1,iteration = gradientDescent(testX, testy, testtheta, iters, alpha[i], False,0)
    final_test_cost = computeCost(testX, testy, b1)
    print("For alpha=" + str(alpha[i]) + " Minimum cost= " + str(final_test_cost))
    test_cost_collection[i] = final_test_cost
    fig, ax = plt.subplots()
    ax.plot(np.arange(iters), test_cost, 'b')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title('Error Vs Iterations for Learning rate ' + str(alpha[i]) + ' in Test')
    fig.show()

test_alphaCost = plot(alpha, test_cost_collection, 'Alpha', 'Cost','Learning Rate v/s Cost for Test Data','b')
test_alphaCost.show()


# In[40]:


#Running gradient descent for TRAIN data with threshold

print("=================================="+"Train data for learning rate with threshold"+"==================================")

threshold_train_cost_collection=np.zeros(len(threshold))
for i in range(len(threshold)):
    thresh_train_cost, betas_train, iteration = gradientDescent(X, y, theta, iters, 0.3, True, threshold[i])
    threshold_train_cost_collection[i] = computeCost(X, y, betas_train)
    print("For alpha=0.3 and threshold=" + str(threshold[i]) + ", Minimum cost= " + str(threshold_train_cost_collection[i]))


# In[41]:


#Running gradient descent for TEST data with threshold

print("=================================="+"Test data for learning rate with threshold"+"==================================")

threshold_test_cost_collection=np.zeros(len(threshold))
for i in range(len(threshold)):
    thresh_test_cost, betas_test, iteration = gradientDescent(testX, testy, testtheta, iters, 0.3, True, threshold[i])
    threshold_test_cost_collection[i] = computeCost(testX, testy, betas_test)
    print("For alpha=0.3 and threshold=" + str(threshold[i]) + ", Minimum cost= " + str(threshold_test_cost_collection[i]))


# In[42]:


#Plot Threshold Vs Cost for Test and Train
plt.plot(threshold,threshold_train_cost_collection,'r',label='train')
plt.plot(threshold,threshold_test_cost_collection,'b',label='test')
plt.xlabel("Threshold")
plt.ylabel("Cost")
plt.legend(loc='upper right')
plt.title("Cost as a function of Threshold")
plt.show()


# In[43]:


train_cost_end,train_betas_end , train_iteration_end = gradientDescent(X, y, theta, iters, 0.3, True, 0.00001)
test_cost_end, test_betas_end, test_iteration_end = gradientDescent(testX, testy, testtheta, iters, 0.3, True, 0.00001 )


# #Plot Iterations Vs Cost for Test and Train
plt.plot(train_cost_end,'r',label='train')
plt.plot(test_cost_end,'b',label='test')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.legend(loc='upper right')
plt.title("Cost as a function of Number of Iterations for Best Threshold")
plt.show()


# In[57]:


#########################################Experimenting with 5 random variables###############################################################

Xrandom=X[:,[0,1,2,3,5,10]]
thetanew= np.zeros([1,6])
rcost, rtheta, ri =gradientDescent(Xrandom,y,thetanew,iters,0.3,False,0.00001)
a=computeCost(Xrandom,y,rtheta)
print(a)


# In[61]:


print(final_cost_collection)
Xtestrandom=testX[:,[0,1,2,3,5,10]]
thetanewtest= np.zeros([1,6])
rcosttest, rthetatest, ritest =gradientDescent(Xtestrandom,testy,thetanewtest,iters,0.3,False,0.00001)
b=computeCost(Xtestrandom,testy,rthetatest)
print(b)


# In[62]:


Xpicked=X[:,[0,1,2,6,7,10]]
thetapicked=np.zeros([1,6])
pcost, ptheta, pi=gradientDescent(Xpicked,y,thetapicked,iters,0.3,False,0.00001)
d=computeCost(Xpicked,y,ptheta)
print(d)
print(ptheta)


# In[60]:


Xpickedtest=testX[:,[0,1,2,6,7,10]]
thetapickedtest=np.zeros([1,6])
pcosttest, pthetatest, pitest=gradientDescent(Xpickedtest,testy,thetapickedtest,iters,0.3,False,0.00001)
z=computeCost(Xpickedtest,testy,pthetatest)
print(z)


# In[ ]:




