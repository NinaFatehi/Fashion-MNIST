#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# In[29]:


from scipy.spatial.distance import cdist 
import numpy as np
import math
from scipy.special import comb


def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


def IsOptimal(np1, np2):
    if np.sum((np2-np1)/np1*100) > 0.001:
        return False
    else:
        return True
    

def kmeans(x,k, no_of_iterations):
    idx = np.random.choice(len(x), k, replace=False)
    #Randomly choosing Centroids 
    centroids = x[idx, :] #Step 1
     
    #finding the distance between centroids and all the data points
    #distances = np.linalg.norm(x - centroids[0,:],axis=1).reshape(-1,1)
    distances = cdist(x, centroids ,'euclidean')
    #distances = euclidean(x, centroids) #Step 2
    print(distances)
    #Centroid with the minimum Distance
    points = np.array([np.argmin(i) for i in distances]) #Step 3
    
    #Repeating the above steps for a defined number of iterations
    #Step 4
    for _ in range(no_of_iterations): 
        centroids1 = []
        for idx in range(k):
            #Updating Centroids by taking mean of Cluster it belongs to
            temp_cent = x[points==idx].mean(axis=0) 
            centroids1.append(temp_cent)
            if IsOptimal(centroids, centroids1) == True:
                break
            else:
                centroids = np.vstack(centroids1) #Updated Centroids 
         
        
        #Repeating the above steps for a defined number of iterations
    #Step 4
                #distances = euclidean(x, centroids)
                distances = cdist(x, centroids ,'euclidean')
                #distances = np.linalg.norm(x - centroids[0,:],axis=1).reshape(-1,1)
                points = np.array([np.argmin(i) for i in distances])    
                temp_cent = x[points==idx].mean(axis=0)
    print(points)
    return points 


# In[30]:


data= np.random.randn(3000).reshape(1000,3)


# In[31]:


data.shape


# In[32]:


label = kmeans(data,5,100)
 
#Visualize the results
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(data[label == i , 0] , data[label == i , 1] , label = i)
plt.legend()
plt.show()


# In[33]:


label = kmeans(data,10,2000)
 
#Visualize the results
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(data[label == i , 0] , data[label == i , 1] , label = i)
plt.legend()
plt.show()


# In[34]:


label = kmeans(data,10,500)
 
#Visualize the results
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(data[label == i , 0] , data[label == i , 1] , label = i)
plt.legend()
plt.show()


# ## Q.2. Mnist Dataset
# 

# In[35]:


mnist = pd.read_csv('fashion-mnist_train.csv')


# In[36]:


mnist.head()


# In[37]:


mnist['label'].value_counts()


# In[38]:


features = mnist.drop(['label'], axis=1)


# In[39]:


features.head()


# In[40]:


data1 = np.array(features)


# In[42]:


label = kmeans(data1,10,100)


# Following is the distribution of classes in the clustr

# In[43]:


true_labels = mnist['label']


# In[44]:


pd.Series(label).value_counts()


# Following is the distribution of 9 classes in the dataself itsel, which is referred as true labels

# In[45]:


pd.Series(true_labels).value_counts()


# In[46]:


pd.DataFrame({'True Labels':true_labels, 'Cluster Labels':label})


# In[47]:


from sklearn.metrics.cluster import adjusted_rand_score


# In[48]:


adjusted_rand_score(true_labels, label)


# In[49]:


label.shape


# In[50]:


from sklearn.decomposition import PCA


# In[51]:


pca = PCA(10)


# In[52]:


data2= pca.fit_transform(data1)


# In[53]:


true_labels = mnist['label']


# ### k=10

# In[249]:


label = kmeans(data2,10,1000)


# In[250]:


label.shape


# In[287]:


pd.DataFrame({'True Labels':true_labels, 'Cluster Labels':label})


# In[252]:


from sklearn.metrics.cluster import adjusted_rand_score


# In[253]:


adjusted_rand_score(true_labels, label)


# In[61]:


rand_index_score(true_labels, label)


# In[254]:


u_labels = np.unique(label)
u_labels


# In[255]:


label


# In[256]:


#Visualize the results
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(data2[label == i , 0] , data2[label == i , 1] , label = i)
plt.legend()
plt.show()


# In[257]:


adjusted_rand_score(true_labels, label)


# ### k=5

# In[62]:


label = kmeans(data2,5,1000)
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(data2[label == i , 0] , data2[label == i , 1] , label = i)
plt.legend()
plt.show()


# In[336]:


adjusted_rand_score(true_labels, label)


# In[63]:


rand_index_score(true_labels, label)


# ### k=7

# In[65]:


label = kmeans(data2,7,1000)
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(data2[label == i , 0] , data2[label == i , 1] , label = i)
plt.legend()
plt.show()


# In[261]:


adjusted_rand_score(true_labels, label)


# In[66]:


rand_index_score(true_labels, label)


# ### k=8

# In[67]:


label = kmeans(data2,8,100)
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(data2[label == i , 0] , data2[label == i , 1] , label = i)
plt.legend()
plt.show()


# In[263]:


adjusted_rand_score(true_labels, label)


# In[68]:


rand_index_score(true_labels, label)


# ### k=12

# In[69]:


label = kmeans(data2,12,1000)
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(data2[label == i , 0] , data2[label == i , 1] , label = i)
plt.legend()
plt.show()


# In[338]:


adjusted_rand_score(true_labels, label)


# In[70]:


rand_index_score(true_labels, label)


# ### k=15

# In[71]:


label = kmeans(data2,15,1000)
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(data2[label == i , 0] , data2[label == i , 1] , label = i)
plt.legend()
plt.show()


# In[340]:


adjusted_rand_score(true_labels, label)


# In[72]:


rand_index_score(true_labels, label)


# ### k=13

# In[73]:


label = kmeans(data2,13,100)
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(data2[label == i , 0] , data2[label == i , 1] , label = i)
plt.legend()
plt.show()


# In[74]:


rand_index_score(true_labels, label)


# In[342]:


adjusted_rand_score(true_labels, label)


# ### k=14

# In[75]:


label = kmeans(data2,14,1000)
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(data2[label == i , 0] , data2[label == i , 1] , label = i)
plt.legend()
plt.show()


# In[344]:


adjusted_rand_score(true_labels, label)


# In[76]:


rand_index_score(true_labels, label)


# In[345]:


true_labels.unique()


# In[346]:


type(true_labels)


# In[347]:


pd.Series(label).unique()


# In[348]:


pd.DataFrame({'True Labels':true_labels, 'Predicted Labels':pd.Series(label)})


# The best k in terms of adjustable rand seems to be k=15 

# ## Q.3

# We apply k means with k=15 to the dataset for 5 times.

# # 1

# In[77]:


label = kmeans(data2,15,1000)
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(data2[label == i , 0] , data2[label == i , 1] , label = i)
plt.legend()
plt.show()


# In[352]:


adjusted_rand_score(true_labels, label)


# In[78]:


rand_index_score(true_labels, label)


# # 2

# In[79]:


label = kmeans(data2,15,1000)
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(data2[label == i , 0] , data2[label == i , 1] , label = i)
plt.legend()
plt.show()


# In[360]:


adjusted_rand_score(true_labels, label)


# In[80]:


rand_index_score(true_labels, label)


# # 3

# In[81]:


label = kmeans(data2,15,1000)
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(data2[label == i , 0] , data2[label == i , 1] , label = i)
plt.legend()
plt.show()


# In[362]:


adjusted_rand_score(true_labels, label)


# In[82]:


rand_index_score(true_labels, label)


# # 4

# In[83]:


label = kmeans(data2,15,1000)
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(data2[label == i , 0] , data2[label == i , 1] , label = i)
plt.legend()
plt.show()


# In[364]:


adjusted_rand_score(true_labels, label)


# In[84]:


rand_index_score(true_labels, label)


# # 5

# In[86]:


label = kmeans(data2,15,1000)
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(data2[label == i , 0] , data2[label == i , 1] , label = i)
plt.legend()
plt.show()


# In[366]:


adjusted_rand_score(true_labels, label)


# In[87]:


rand_index_score(true_labels, label)


# # Q.4

# From the above results, one may conclude that the kmeans is highly dependent on the initialization. With each rerun of the algorithm, we get new results.

# ### Above steps for mnist test data

# In[89]:


mnist_test = pd.read_csv('C:\\Users\\Hana\\Downloads\\fashion-mnist_test.csv')


# In[90]:


features_test = mnist_test.drop(['label'], axis=1)


# In[91]:


true_labels_test = mnist_test['label']


# In[103]:


data_test = np.array(features_test)


# In[104]:


pca = PCA(10)
data_pca= pca.fit_transform(data_test)


# # k=10

# In[107]:


label_test = kmeans(data_pca,10,1000)


# In[108]:


label_test.shape


# In[109]:


adjusted_rand_score(true_labels_test, label_test)


# In[111]:


rand_index_score(true_labels_test, label_test)


# # k=9

# In[112]:


label_test = kmeans(data_pca,9,1000)


# In[113]:


adjusted_rand_score(true_labels_test, label_test)


# In[114]:


rand_index_score(true_labels_test, label_test)


# # k=8

# In[115]:


label_test = kmeans(data_pca,8,1000)


# In[116]:


adjusted_rand_score(true_labels_test, label_test)


# In[117]:


rand_index_score(true_labels_test, label_test)


# # k=7

# In[118]:


label_test = kmeans(data_pca,7,1000)


# In[119]:


adjusted_rand_score(true_labels_test, label_test)


# In[120]:


rand_index_score(true_labels_test, label_test)


# # k=6

# In[121]:


label_test = kmeans(data_pca,6,1000)


# In[122]:


adjusted_rand_score(true_labels_test, label_test)


# In[123]:


rand_index_score(true_labels_test, label_test)


# # k=5

# In[127]:


label_test = kmeans(data_pca,5,1000)


# In[128]:


adjusted_rand_score(true_labels_test, label_test)


# In[129]:


rand_index_score(true_labels_test, label_test)


# # k=11

# In[136]:


label_test = kmeans(data_pca,11,1000)


# In[137]:


adjusted_rand_score(true_labels_test, label_test)


# In[138]:


rand_index_score(true_labels_test, label_test)


# # k=12

# In[139]:


label_test = kmeans(data_pca,12,1000)


# In[140]:


adjusted_rand_score(true_labels_test, label_test)


# In[141]:


rand_index_score(true_labels_test, label_test)


# # k=13

# In[145]:


label_test = kmeans(data_pca,13,1000)


# In[146]:


adjusted_rand_score(true_labels_test, label_test)


# In[147]:


rand_index_score(true_labels_test, label_test)


# # k=14

# In[148]:


label_test = kmeans(data_pca,14,1000)


# In[149]:


adjusted_rand_score(true_labels_test, label_test)


# In[150]:


rand_index_score(true_labels_test, label_test)


# # k=15

# In[151]:


label_test = kmeans(data_pca,15,1000)


# In[152]:


adjusted_rand_score(true_labels_test, label_test)


# In[153]:


rand_index_score(true_labels_test, label_test)


# k=8 shows the best performance in terms of rand index, so we run the algorithm five times on the test data with k=8

# ## 1

# In[156]:


label_test = kmeans(data_pca,8,1000)
print("Adjusted Rand Score: {}".format(adjusted_rand_score(true_labels_test, label_test)))
print("Rand Index: {}".format(rand_index_score(true_labels_test, label_test)))


# ## 2

# In[157]:


label_test = kmeans(data_pca,8,1000)
print("Adjusted Rand Score: {}".format(adjusted_rand_score(true_labels_test, label_test)))
print("Rand Index: {}".format(rand_index_score(true_labels_test, label_test)))


# ## 3

# In[158]:


label_test = kmeans(data_pca,8,1000)
print("Adjusted Rand Score: {}".format(adjusted_rand_score(true_labels_test, label_test)))
print("Rand Index: {}".format(rand_index_score(true_labels_test, label_test)))


# ## 4

# In[159]:


label_test = kmeans(data_pca,8,1000)
print("Adjusted Rand Score: {}".format(adjusted_rand_score(true_labels_test, label_test)))
print("Rand Index: {}".format(rand_index_score(true_labels_test, label_test)))


# ## 5

# In[160]:


label_test = kmeans(data_pca,8,1000)
print("Adjusted Rand Score: {}".format(adjusted_rand_score(true_labels_test, label_test)))
print("Rand Index: {}".format(rand_index_score(true_labels_test, label_test)))

