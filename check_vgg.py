import os.path
import numpy as np
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time

with tf.Session() as sess:
    vgg = tf.saved_model.loader.load(sess, ['vgg16'], './data/vgg')
#    print(sess.graph.get_operations())
    for op in sess.graph.get_operations():
        for ts in op.outputs:
            print(ts.name + ' \t ' + str(ts.shape))
