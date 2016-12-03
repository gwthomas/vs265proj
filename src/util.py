import pickle
import numpy as np
import os

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def one_hot(labels, num_classes):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10}'''
    return np.eye(num_classes)[labels]

def load_cifar(dir, reshape=True):
    def reshape_image(img):
        shape = (32,32)
        r = img[:1024].reshape(shape)
        g = img[1024:2048].reshape(shape)
        b = img[2048:].reshape(shape)
        return np.stack([r,g,b])
    if not reshape:
        reshape_image = lambda x: x

    Xtrain, Ytrain, Xtest, Ytest = [], [], [], []
    for i in range(1,6):
        batch_dict = unpickle(os.path.join(dir, 'data_batch_%i' % i))
        Xtrain.extend([reshape_image(img) for img in batch_dict['data']])
        Ytrain.extend(batch_dict['labels'])
    Xtrain = np.stack(Xtrain).astype(float)
    Ytrain = np.array(Ytrain)

    test_dict = unpickle(os.path.join(dir, 'test_batch'))
    Xtest = np.stack([reshape_image(img) for img in test_dict['data']])
    Ytest = np.array(test_dict['labels'])

    return Xtrain, Ytrain, Xtest, Ytest
