#import Image 
#import ImageOps
import numpy as np
from time import time
from sklearn.decomposition import MiniBatchDictionaryLearning
from keras.datasets import mnist
from function import *
import matplotlib.pyplot as plt

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

iteration=500
patch_hight=14
patch_width=14
num_basis=100

#initialization of dictionary class
t0=time()
dict=MiniBatchDictionaryLearning(n_components=num_basis,alpha=1.0,transform_algorithm='lasso_lars',transform_alpha=1.0,fit_algorithm='lars',n_iter=iteration)

M=np.mean(train_images,axis=0)[np.newaxis,:]
white_image=train_images-M
white_image/=np.std(white_image,axis=0)
print(white_image)
patches=np.empty((0,patch_hight,patch_width))

for i in range(white_image[0].size):
    patches_temp=extract_patch(train_images[i],patch_hight,patch_width)
    patches=np.concatenate((patches,patches_temp),axis=0)

patches=patches.reshape(patches.shape[0],-1)

V=dict.fit(patches).components_
np.save("dictionary.npy",V)

plt.figure(figsize=(6,6))

for i,comp in enumerate(V[:100]):
    plt.subplot(10,10,i+1)
    plt.imshow(comp.reshape(14,14),cmap=plt.cm.gray_r,interpolation='nearest')
    plt.xticks(())
    plt.yticks(())

plt.show()
