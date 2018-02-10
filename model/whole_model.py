"""
input: a model tensor feature
output: softmax_cross_entropy
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.contrib.quantize.python import quant_ops
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from densenet import DenseNet, create_densenet_info

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def save_graph_to_file(sess, graph, graph_file_name, final_tensor_name):
  output_graph_def = graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [final_tensor_name])

  with gfile.FastGFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  return


def get_batch_image(sess, jpeg_data_tensor, decoded_image_tensor, image_datas):
  """Retrieves resized input values.

  If no distortions are being applied, this function can retrieve the resized input
  bottleneck values directly from disk for images.

  Args:
    sess: Current TensorFlow Session.
    jpeg_data_tensor: The layer to feed jpeg image data into.
    decoded_image_tensor: The output of decoding and resizing the image.
    image_datas: List of data which need to convert tensor

  Returns:
    List of resized input arrays
  """
  resized_inputs = []
  for image_data in image_datas:
      resized_input= run_resize_input_on_image(sess, image_data, jpeg_data_tensor,
                                           decoded_image_tensor)
      resized_inputs.append(resized_input)

  return resized_inputs


def run_resize_input_on_image(sess, image_data, image_data_tensor,
                                            decoded_image_tensor):
  """Runs inference on an jpeg image to resize.

  Args:
    sess: Current active TensorFlow Session.
    image_data: String of raw JPEG data.
    image_data_tensor: Input data layer in the graph.
    decoded_image_tensor: Output of initial image resizing and preprocessing.

  Returns:
    Numpy array of resize image values.
  """
  # First decode the JPEG image, resize it, and rescale the pixel values.
  resized_input_values = sess.run(decoded_image_tensor,
                                  {image_data_tensor: image_data})

  resized_input_values = np.squeeze(resized_input_values)
  return resized_input_values

def add_jpeg_decoding(input_width, input_height, input_depth, input_mean,
                      input_std):
  """Adds operations that perform JPEG decoding and resizing to the graph..

  Args:
    input_width: Desired width of the image fed into the recognizer graph.
    input_height: Desired width of the image fed into the recognizer graph.
    input_depth: Desired channels of the image fed into the recognizer graph.
    input_mean: Pixel value that should be zero in the image for the graph.
    input_std: How much to divide the pixel values by before recognition.

  Returns:
    Tensors for the node to feed JPEG data into, and the output of the
      preprocessing steps.
  """
  jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  resize_shape = tf.stack([input_height, input_width])
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)
  offset_image = tf.subtract(resized_image, input_mean)
  mul_image = tf.multiply(offset_image, 1.0 / input_std)
  resized_input_tensor = tf.placeholder_with_default(mul_image, shape = (None,
                                input_height, input_width, input_depth ),
                                    name='ResizedInputPlaceholder')
  return jpeg_data, mul_image, resized_input_tensor


def create_model_info(architecture):
  """Given the name of a model architecture, returns information about it.

  There are different base image recognition models that can be trained
  , and this function translates from the name of a model to the attributes

  Args:
    architecture: Name of a model architecture.

  Returns:
    Dictionary of information about the model, or None if the name isn't
    recognized

  Raises:
    ValueError: If architecture name is unknown.
  """
  architecture = architecture.lower()
  is_quantized = False
  if architecture == 'densenet':
    # pylint: disable=line-too-long
    data_url = None
    # pylint: enable=line-too-long
    bottleneck_tensor_name = 'merge_linear'
    bottleneck_tensor_size = 2048
    input_width = 300
    input_height = 300
    input_depth = 3
    resized_input_tensor_name = 'ResizedInputPlaceholder'
    model_file_name = None
    input_mean = 128
    input_std = 128

  elif architecture == 'inception_v3':
    # pylint: disable=line-too-long
    data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    # pylint: enable=line-too-long
    bottleneck_tensor_name = 'pool_3/_reshape:0'
    bottleneck_tensor_size = 2048
    input_width = 299
    input_height = 299
    input_depth = 3
    resized_input_tensor_name = 'Mul:0'
    model_file_name = 'classify_image_graph_def.pb'
    input_mean = 128
    input_std = 128

  elif architecture.startswith('mobilenet_'):
    parts = architecture.split('_')
    if len(parts) != 3 and len(parts) != 4:
      tf.logging.error("Couldn't understand architecture name '%s'",
                       architecture)
      return None
    version_string = parts[1]
    if (version_string != '1.0' and version_string != '0.75' and
        version_string != '0.50' and version_string != '0.25'):
      tf.logging.error(
          """"The Mobilenet version should be '1.0', '0.75', '0.50', or '0.25',
  but found '%s' for architecture '%s'""",
          version_string, architecture)
      return None
    size_string = parts[2]
    if (size_string != '224' and size_string != '192' and
        size_string != '160' and size_string != '128'):
      tf.logging.error(
          """The Mobilenet input size should be '224', '192', '160', or '128',
 but found '%s' for architecture '%s'""",
          size_string, architecture)
      return None
    if len(parts) == 3:
      is_quantized = False
    else:
      if parts[3] != 'quantized':
        tf.logging.error(
            "Couldn't understand architecture suffix '%s' for '%s'", parts[3],
            architecture)
        return None
      is_quantized = True

    if is_quantized:
      data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
      data_url += version_string + '_' + size_string + '_quantized_frozen.tgz'
      bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
      resized_input_tensor_name = 'Placeholder:0'
      model_dir_name = ('mobilenet_v1_' + version_string + '_' + size_string +
                        '_quantized_frozen')
      model_base_name = 'quantized_frozen_graph.pb'

    else:
      data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
      data_url += version_string + '_' + size_string + '_frozen.tgz'
      bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
      resized_input_tensor_name = 'input:0'
      model_dir_name = 'mobilenet_v1_' + version_string + '_' + size_string
      model_base_name = 'frozen_graph.pb'

    bottleneck_tensor_size = 1001
    input_width = int(size_string)
    input_height = int(size_string)
    input_depth = 3
    model_file_name = os.path.join(model_dir_name, model_base_name)
    input_mean = 127.5
    input_std = 127.5
  else:
    tf.logging.error("Couldn't understand architecture name '%s'", architecture)
    raise ValueError('Unknown architecture', architecture)

  return {
      'data_url': data_url,
      'bottleneck_tensor_name': bottleneck_tensor_name,
      'bottleneck_tensor_size': bottleneck_tensor_size,
      'input_width': input_width,
      'input_height': input_height,
      'input_depth': input_depth,
      'resized_input_tensor_name': resized_input_tensor_name,
      'model_file_name': model_file_name,
      'input_mean': input_mean,
      'input_std': input_std,
      'quantize_layer': is_quantized,
  }


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor,
                           bottleneck_tensor_size, quantize_layer, FLAGS):
  """Adds a new softmax and fully-connected layer for training.

  We need to retrain the top layer to identify our new classes, so this function
  adds the right operations to the graph, along with some variables to hold the
  weights, and then sets up all the gradients for the backward pass.

  The set up for the softmax and fully-connected layers is based on:
  https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

  Args:
    class_count: Integer of how many categories of things we're trying to
        recognize.
    final_tensor_name: Name string for the new final node that produces results.
    bottleneck_tensor: The output of the main CNN graph.
    bottleneck_tensor_size: How many entries in the bottleneck vector.
    quantize_layer: Boolean, specifying whether the newly added layer should be
        quantized.

  Returns:
    The tensors for the training and cross entropy results, and tensors for the
    bottleneck input and ground truth input.
  """
  with tf.name_scope('input'):
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor,
        shape=[None, bottleneck_tensor_size],
        name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(
        tf.int64, [None], name='GroundTruthInput')

  # Organizing the following ops as `final_training_ops` so they're easier
  # to see in TensorBoard
  layer_name = 'final_training_ops'
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      initial_value = tf.truncated_normal(
          [bottleneck_tensor_size, class_count], stddev=0.001)
      layer_weights = tf.Variable(initial_value, name='final_weights')
      if quantize_layer:
        quantized_layer_weights = quant_ops.MovingAvgQuantize(
            layer_weights, is_training=True)
        variable_summaries(quantized_layer_weights)

      variable_summaries(layer_weights)
    with tf.name_scope('biases'):
      layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
      if quantize_layer:
        quantized_layer_biases = quant_ops.MovingAvgQuantize(
            layer_biases, is_training=True)
        variable_summaries(quantized_layer_biases)

      variable_summaries(layer_biases)

    with tf.name_scope('Wx_plus_b'):
      if quantize_layer:
        logits = tf.matmul(bottleneck_input,
                           quantized_layer_weights) + quantized_layer_biases
        logits = quant_ops.MovingAvgQuantize(
            logits,
            init_min=-32.0,
            init_max=32.0,
            is_training=True,
            num_bits=8,
            narrow_range=False,
            ema_decay=0.5)
        tf.summary.histogram('pre_activations', logits)
      else:
        logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
        tf.summary.histogram('pre_activations', logits)

  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

  tf.summary.histogram('activations', final_tensor)

  with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
        labels=ground_truth_input, logits=logits)

  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
          final_tensor)


def create_bottleneck_info(train_tuple):
  return { 'train_step': train_tuple[0], 'cross_entropy_mean': train_tuple[1],
 'bottleneck_input': train_tuple[2], 'ground_truth_input': train_tuple[3],
 'final_tensor': train_tuple[4] }


def create_evaluation_info(evaluation_tuple):
  return { 'evaluation_step': evaluation_tuple[0],
            'prediction': evaluation_tuple[1] }


def add_evaluation_step(result_tensor, ground_truth_tensor):
  """Inserts the operations we need to evaluate the accuracy of our results.

  Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data
    into.

  Returns:
    Tuple of (evaluation step, prediction).
  """
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      prediction = tf.argmax(result_tensor, 1)
      correct_prediction = tf.equal(prediction, ground_truth_tensor)
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)
  return (evaluation_step, prediction)


class Model:
  def __init__(self, FLAGS, dataset, sess, graph):
    self.dataset = dataset
    self.FLAGS = FLAGS
    self.sess = sess
    self.model_info = create_model_info(FLAGS.architecture)
    if not self.model_info:
      tf.logging.error('Did not recognize architecture flag')
      return -1
    self.jpeg_data_tensor = None
    self.input_tensor = None
    self.training_flag = tf.placeholder(tf.bool)
    self.bottleneck_dim = self.model_info['bottleneck_tensor_size']
    self.create_model()
    self.graph = graph

  def create_model(self):
    FLAGS = self.FLAGS
    model_info = self.model_info

    # Set up the image decoding sub-graph.
    self.jpeg_data_tensor, self.decoded_image_tensor, self.input_tensor = add_jpeg_decoding(
        model_info['input_width'], model_info['input_height'],
        model_info['input_depth'], model_info['input_mean'],
        model_info['input_std'])

    # Set up the bottleneck graph(main graph).
    self.feature_tensor = self.create_bottleneck()

    # Add the new layer that we'll be training.
    self.bottleneck_info = create_bottleneck_info(
                        add_final_training_ops(self.dataset.class_num,
                            self.FLAGS.final_tensor_name, self.feature_tensor,
                            self.bottleneck_dim, model_info['quantize_layer'], self.FLAGS))

    # Create the operations we need to evaluate the accuracy of our new layer.
    self.evaluation_info = create_evaluation_info(
            add_evaluation_step( self.bottleneck_info['final_tensor']
                                ,self.bottleneck_info['ground_truth_input']))
    # Merge all the summaries and write them out to the summaries_dir
    self.merged = tf.summary.merge_all()
    self.train_writer = tf.summary.FileWriter(self.FLAGS.summaries_dir + '/train',
                                         self.sess.graph)
    self.validation_writer = tf.summary.FileWriter(
        self.FLAGS.summaries_dir + '/validation')

    # Set up all our weights to their initial default values.
    init = tf.global_variables_initializer()
    self.sess.run(init)

  def create_bottleneck(self):
    densenet_info = create_densenet_info(nb_blocks_layers = [6, 12, 48, 32],
                            filters = 24,
                            output_dim = self.bottleneck_dim)
    densenet =  DenseNet(self.input_tensor, self.training_flag,
                                self.FLAGS.dropout_rate, densenet_info)
    feature_tensor = densenet.model
    return feature_tensor

  def train(self):
    # Run the training for as many cycles as requested on the command line.
    train_generator = self.dataset.create_batch(
                            batch_size = self.FLAGS.train_batch_size,
                            category = 'training')
    validation_generator = self.dataset.create_batch(
                            batch_size = self.FLAGS.validation_batch_size,
                            category = 'validation')
    for i in range(self.FLAGS.how_many_training_steps):

      # Get a batch of input values, either calculated fresh every
      (train_image_datas, train_ground_truths, _)=train_generator.next()
      train_input_datas = get_batch_image(self.sess, self.jpeg_data_tensor,
                                          self.decoded_image_tensor,
                                          train_image_datas)

      # Feed the bottlenecks and ground truth into the graph, and run a training
      # step. Capture training summaries for TensorBoard with the `merged` op.
      train_summary, _ = self.sess.run(
          [self.merged, self.bottleneck_info['train_step']],
          feed_dict={self.input_tensor: train_input_datas,
                self.bottleneck_info['ground_truth_input']: train_ground_truths,
                self.training_flag: True })
      self.train_writer.add_summary(train_summary, i)

      # Every so often, print out how well the graph is training.
      is_last_step = (i + 1 == self.FLAGS.how_many_training_steps)
      if (i % self.FLAGS.eval_step_interval) == 0 or is_last_step:
        train_accuracy, cross_entropy_value = self.sess.run(
            [self.evaluation_info['evaluation_step'],
            self.bottleneck_info['cross_entropy_mean']],
            feed_dict={self.input_tensor: train_input_datas,
                 self.bottleneck_info['ground_truth_input']: train_ground_truths,
                 self.training_flag: False })
        tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                        (datetime.now(), i, train_accuracy * 100))
        tf.logging.info('%s: Step %d: Cross entropy = %f' %
                        (datetime.now(), i, cross_entropy_value))

        # Get a batch of input values, either calculated fresh every
        (validation_image_datas, validation_ground_truths, _
                                                ) = validation_generator.next()
        validation_input_datas = get_batch_image(self.sess, self.jpeg_data_tensor,
                                            self.decoded_image_tensor,
                                            validation_image_datas)

        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary, validation_accuracy = self.sess.run(
            [self.merged, self.evaluation_info['evaluation_step']],
            feed_dict={self.input_tensor: validation_input_datas,
            self.bottleneck_info['ground_truth_input']: validation_ground_truths,
            self.training_flag: False })
        self.validation_writer.add_summary(validation_summary, i)
        tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                        (datetime.now(), i, validation_accuracy * 100,
                         self.model_info['bottleneck_tensor_size']))

      # Store intermediate results
      intermediate_frequency = self.FLAGS.intermediate_store_frequency

      if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
          and i > 0):
        intermediate_file_name = (self.FLAGS.intermediate_output_graphs_dir +
                                  'intermediate_' + str(i) + '.pb')
        tf.logging.info('Save intermediate result to : ' +
                        intermediate_file_name)
        save_graph_to_file(self.sess, self.graph, intermediate_file_name, self.FLAGS.final_tensor_name)

  def test(self):
    # We've completed all our training, so run a final test evaluation on
    # some new images we haven't used before.
    test_generator = self.dataset.create_batch(
                            batch_size = self.FLAGS.test_batch_size,
                            category = 'testing')
    # Get a batch of input values, either calculated fresh every
    (test_image_datas, test_ground_truths, test_filenames
     ) = test_generator.next()
    test_input_datas = get_batch_image(self.sess, self.jpeg_data_tensor,
                                       self.decoded_image_tensor,
                                       test_image_datas)

    test_accuracy, predictions = self.sess.run(
        [self.evaluation_info['evaluation_step'],
        self.evaluation_info['prediction']],
        feed_dict={self.input_tensor: test_input_datas,
        self.bottleneck_info['ground_truth_input']: test_ground_truths,
        self.training_flag: False })
    tf.logging.info('Final test accuracy = %.1f%% (N=%d)' %
                    (test_accuracy * 100, self.model_info['bottleneck_tensor_size']))

    if self.FLAGS.print_misclassified_test_images:
      tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
      for i, test_filename in enumerate(test_filenames):
        if predictions[i] != test_ground_truths[i]:
          tf.logging.info('%70s  %s' %
                        (test_filename,
                         list(self.dataset.image_lists.keys())[predictions[i]]))
  def save_graph_to_file(self):
    # Write out the trained graph and labels with the weights stored as
    # constants.
    save_graph_to_file(self.sess, self.graph, self.FLAGS.output_graph,
                       self.FLAGS.final_tensor_name)
    with gfile.FastGFile(self.FLAGS.output_labels, 'w') as f:
      f.write('\n'.join(self.dataset.image_lists.keys()) + '\n')
