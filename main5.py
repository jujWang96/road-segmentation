import os.path
import numpy as np
import time
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    vgg_meta_graph = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # print(sess.graph.get_operations())

    image_input = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    # Added layers
    layer1_out = sess.graph.get_tensor_by_name('pool1:0')
    layer2_out = sess.graph.get_tensor_by_name('pool2:0')
    layer5_out = sess.graph.get_tensor_by_name('pool5:0')
    layer6_out = sess.graph.get_tensor_by_name('dropout/mul:0')
    
    return image_input, keep_prob, [layer1_out, layer2_out, layer3_out, layer4_out, layer5_out, layer6_out, layer7_out]

def layers(loaded_layers, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    '''Encoder'''
    encoded_1 = tf.layers.conv2d(loaded_layers[0], num_classes, 1, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    encoded_2 = tf.layers.conv2d(loaded_layers[1], num_classes, 1, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    encoded_3 = tf.layers.conv2d(loaded_layers[2], num_classes, 1, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    encoded_4 = tf.layers.conv2d(loaded_layers[3], num_classes, 1, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    encoded_5 = tf.layers.conv2d(loaded_layers[4], num_classes, 1, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    encoded_6 = tf.layers.conv2d(loaded_layers[5], num_classes, 1, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    encoded_7 = tf.layers.conv2d(loaded_layers[6], num_classes, 1, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    '''Decoder'''
    deconv_1 = tf.layers.conv2d_transpose(encoded_7, num_classes, 4, 16, "same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)) # padding?
    skip_1 = tf.add(deconv_1, encoded_1)

    deconv_3 = tf.layers.conv2d_transpose(skip_1, num_classes, 16, 2, "same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    return deconv_3

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, [-1, num_classes])

    crossent_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
        logits = logits,
        labels = tf.reshape(correct_label, [-1, num_classes])
    )
    )

    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(crossent_loss)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]

    train_op = optimizer.apply_gradients(capped_gradients)

    return logits, train_op, crossent_loss

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, losslog, timelog):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    
    training_info = '1+7\nNumber of epochs:\t{0}\n'.format(str(epochs)) + \
                    'Batch size:\t{0}\n'.format(str(batch_size)) + \
                    'Keep_prob:\t0.5\n' + \
                    'Learning rate:\t0.0001\n'
    losslog.write(training_info)
    timelog.write(training_info)

    for epoch in range(epochs):
        print("### EPOCH %d"%(epoch+1))
        t0 = time.time()

        for batch_img, batch_gt in get_batches_fn(batch_size):
            _, crossent_loss = sess.run(
               [train_op, cross_entropy_loss], 
              {
                input_image: batch_img,
                correct_label: batch_gt,
                keep_prob: 0.5,
                learning_rate: 0.0001,
            })
            print("Loss:", np.round(crossent_loss, 3), "  Time used:", time.time()-t0)
            losslog.write(str(crossent_loss) + '\n')
            timelog.write(str(time.time()-t0) + '\n')

        print("Time used:", time.time()-t0, "\n")


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    timestamp = time.time()
    losslog = open('./logs/loss_{0}.txt'.format(str(timestamp)), 'w')
    timelog = open('./logs/time_{0}.txt'.format(str(timestamp)), 'w')

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, loaded_layers = load_vgg(sess, vgg_path)
        nn_last_layer = layers(loaded_layers, num_classes)
        correct_label = tf.placeholder(tf.int32, shape=[None, None, None, None])
        learning_rate = tf.placeholder(tf.float32)
        logits, train_op, crossent_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        epochs = 10
        batch_size = 4
        sess.run(tf.global_variables_initializer())
        train_nn(
            sess, epochs, batch_size, get_batches_fn, train_op, crossent_loss, 
            image_input, correct_label, keep_prob, learning_rate, losslog, timelog)

        losslog.close()
        timelog.close()

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input, timestamp)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
