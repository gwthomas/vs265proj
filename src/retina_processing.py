import os
import cv2
from cv2 import bioinspired
import pickle
import numpy as np
import theano
import theano_stuff, util
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
    frame, size = dataset.get_lenna()
    retina = cv2.bioinspired.createRetina((size[0], size[1]))
    retina.clearBuffers()
    retina.run(frame)
    lenna_output_parvo = retina.getParvo()
    print(lenna_output_parvo.shape)
    lenna_output_magno = retina.getMagno()
    
    Xtrain, Ytrain, Xtest, Ytest = util.load_cifar('cifar-10-batches-py', reshape=True)
    
