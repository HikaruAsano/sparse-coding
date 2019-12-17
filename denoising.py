#import Image 
#import ImageOps
import numpy as np
from time import time
#from sklearn.cross_decomposition import MiniBatchDictionaryLearning
from keras.datasets import mnist
from function import *
import matplotlib.pyplot as plt

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

iteration=500
patch_size=(8,8)
num_basis=100

#initialization of dictionary class
t0=time()
#dict=MiniBatchDictionaryLearning(n_atom=num_basis,alpha=1.0,transform_algorithm='lasso_lars',transform_alpha=1.0,fit_algorithm='lars',n_iter=iteration)

M=np.mean(train_images,axis=0)[np.newaxis,:]
white_image=train_images-M
white_image/=np.std(white_image,axis=0)

patches=extract_patch(white_image[0],7,7)
plt.imshow(pathcs[0])
plt.show()
#V=dict.fit(white_image).components_
