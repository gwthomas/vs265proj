import os
import cv2
import pickle
import lasagne
import numpy as np
import theano_stuff, util
from visual_system import VisualSystem
from simretina import retina, dataset
from nose.tools import assert_equal, assert_not_equal

option = "test-setup-function"#"test-bgr2rgb-sequence"
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
