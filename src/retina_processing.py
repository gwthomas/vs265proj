import os
import cv2
from cv2 import bioinspired
import pickle
import numpy as np
import theano
import theano_stuff, util
import matplotlib.pyplot as plt
from visual_system import VisualSystem
from simretina import retina, dataset

option = "data"#"test-setup-function"#"test-bgr2rgb-sequence"
if option == "test-bgr2rgb-sequence":
    frames, size = dataset.get_horse_riding()

    new_frames = gui.trans_bgr2rgb_seq(frames)

    for frame in new_frames:
        cv2.imshow("test", frame)
        cv2.waitKey(delay=0)

    print(len(new_frames))


if option == "test-setup-function":
    eye = retina.init_retina((300, 200))

    print(type(eye.setupOPLandIPLParvoChannel))
    print(type(eye.setupIPLMagnoChannel))
    print(eye.getInputSize())



if option == "data":
    """frame, size = dataset.get_lenna()
    retina = cv2.bioinspired.createRetina((size[0], size[1]))
    retina.clearBuffers()
    retina.run(frame)
    lenna_output_parvo = retina.getParvo()
    lenna_output_magno = retina.getMagno()"""
    
    Xtrain, Ytrain, Xtest, Ytest = util.load_cifar('cifar-10-batches-py', reshape=True)
    img_shape = Xtrain[0].shape
    retina = cv2.bioinspired.createRetina((img_shape[0], img_shape[1]))
    retina.clearBuffers()
    frame = Xtrain[70]
    retina.run(frame)
    #plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #plt.imshow(frame)
    #plt.show()
    output_parvo = retina.getParvo()
    #print(output_parvo)
    #plt.imshow(cv2.cvtColor(output_parvo, cv2.COLOR_BGR2RGB))
    #plt.imshow(output_parvo)
    #plt.show()
    #plt.imshow(cv2.cvtColor(output_magno, cv2.COLOR_BGR2RGB))
    #plt.imshow(output_magno, cmap='gray')
    #plt.show()
    retina = cv2.bioinspired.createRetina((img_shape[0], img_shape[1]))
    data = Xtrain
    Xtrain_parvo = []
    print("Outputting Xtrain parvo files")
    for i in range(Xtrain.shape[0]):
        retina.clearBuffers()
        retina.run(Xtrain[i])
        Xtrain_parvo.append(retina.getParvo())
        if i % 1000 == 0:
            print("progress %d" %i)
    util.write_npy('cifar-10-parvo-out', Xtrain_parvo, 'Xtrain')
    
    Xtest_parvo = []
    print("Outputting Xtest parvo files")
    for i in range(Xtest.shape[0]):
        retina.clearBuffers()
        retina.run(Xtest[i])
        Xtest_parvo.append(retina.getParvo())
        if i % 1000 == 0:
            print("progress %d" %i)

    util.write_npy('cifar-10-parvo-out', Xtest_parvo, 'Xtest')
