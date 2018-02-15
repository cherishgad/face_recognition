"""
Read the dest_directory and make batch and sample
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import os
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import pandas as pd

from tensorflow.contrib.quantize.python import quant_ops
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1
def save_index_numpys(index_numpys, output_path):
  """cashed index_numpys
  Args:
    index_numpys: List of numpys which contain label_index and image_index
    output_path: String path to a file containig label_index and image_index
  """
  indexs_numpy = np.asarray(index_numpys)
  with open( output_path, 'wb') as f:
    np.save(f, indexs_numpy)
def load_index_numpys(input_path):
  """cashed index_numpys
  Args:
    input_path: String path to a file containig label_index and image_index
  Return:
    index_numpy: List of numpys which contain label_index and image_index
  """
  if os.path.exists(input_path):
    indexs_numpy = np.load(input_path)
    return indexs_numpy.tolist()
  else:
    return None
def save_image_lists(image_lists, output_dir):
  """cashed dictionary
  Args:
    image_lists: Dictionary of training images for each label.
    output_dir: String path to a folder containing subfolders holding cached files of image_list values.
  """
  sub_dirs = []
  sub_dirs_path = os.path.join(output_dir, 'sub_dirs.csv')
  for image_lists_key in image_lists.keys():
    sub_dict = image_lists[image_lists_key]
    sub_dir = sub_dict['dir']
    sub_dirs.append(sub_dir)

    training_images = sub_dict['training']
    testing_images = sub_dict['testing']
    validation_images = sub_dict['validation']
    save_dir = os.path.join(output_dir, sub_dir)
    #check the dirctory
    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
    training_path = os.path.join(save_dir, 'training.csv')
    validation_path = os.path.join(save_dir, 'validation.csv')
    testing_path = os.path.join(save_dir, 'testing.csv')
    save_list_in_csv(training_path, training_images)
    save_list_in_csv(validation_path, validation_images)
    save_list_in_csv(testing_path, testing_images)
  save_list_in_csv(sub_dirs_path,sub_dirs)

def load_image_lists(image_dir):
  """cashed dictionary
  Args:
    image_dir: String path to a folder containing subfolders holding cached files of image_list values.
  Return:
    image_lists: Dictionary of training images for each label.
  """
  if not gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
    return None
  result = {}
  sub_dirs = []
  sub_dirs_path = os.path.join(image_dir, 'sub_dirs.csv')
  if not os.path.exists(sub_dirs_path):
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
  else:
    sub_dirs = load_list_in_one_column_file(sub_dirs_path)
  # The root directory comes first, so skip it.
  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue
    dir_name = os.path.basename(sub_dir)
    if dir_name == image_dir:
      continue
    tf.logging.info("Loading image list in '" + dir_name + "'")

    save_dir = os.path.join(image_dir, dir_name)
    #check the dirctory
    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
    training_path = os.path.join(save_dir, 'training.csv')
    validation_path = os.path.join(save_dir, 'validation.csv')
    testing_path = os.path.join(save_dir, 'testing.csv')

    training_images = load_list_in_one_column_file(training_path)
    validation_images = load_list_in_one_column_file(validation_path)
    testing_images = load_list_in_one_column_file(testing_path)

    if (len(training_images)+len(validation_images) + len(testing_images)) < 20:
      tf.logging.warning(
          'WARNING: Folder has less than 20 images, which may cause issues.' +
          str(len(training_images)) + ' : '+  str(len(validation_images)) + ':'
          + str(len(testing_images)))
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    result[label_name] = {
        'dir': dir_name,
        'training': training_images,
        'testing': testing_images,
        'validation': validation_images,
    }
  return result

def load_list_in_one_column_file(file_path):
  """save list in csv
  Args:
    file_path: String path to save file
  Returns:
    save_list: List
  """
  if not isinstance(file_path, str):
    print (type(file_path))
    print ('path should be string')
    return []
  if not gfile.Exists(file_path):
    print("Image directory '" + file_path + "' not found.")
    return []

  with open (file_path, 'r') as f:
    save_list = []
    for line in f:
        save_list.append(str(line).rstrip('\r\n'))
    return save_list

def save_list_in_csv(file_path, save_list):
  """save list in csv
  Args:
    file_path: String path to save file
    save_list: List or List of Lists

  """
  if not isinstance(file_path, str):
    print (type(file_path))
    print ('path should be string')
    return

  if not isinstance(save_list, list):
    print (type(save_list))
    print ('save_list should be list type')
    return
  if len(save_list) == 0:
    with open(file_path, "wb") as f:
      print('save_list have no row')
    return
  if isinstance(save_list[0], list):
    #List of List
    with open(file_path, "wb") as f:
      df = pd.DataFrame(save_list)
      df.to_csv(file_path, index=False, header=False)
  elif isinstance(save_list, list):
    #List
    with open(file_path, "wb") as f:
      df = pd.DataFrame(save_list, columns=["colummn"])
      df.to_csv(file_path, index=False, header=False)
  else:
    print (type(save_list))
    print (' is not avaliable input for save_list')

def create_image_lists_by_percentage(image_dir, testing_percentage,
                                             validation_percentage):
  """Builds a list of training images from the file system.

  Analyzes the sub folders in the image directory, splits them into stable
  training, testing, and validation sets, and returns a data structure
  describing the lists of images for each label and their paths.

  Args:
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.

  Returns:
    A dictionary containing an entry for each label subfolder, with images split
    into training, testing, and validation sets within each label.
  """
  if not gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
    return None
  result = {}
  sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
  # The root directory comes first, so skip it.
  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    dir_name = os.path.basename(sub_dir)
    if dir_name == image_dir:
      continue
    tf.logging.info("Looking for images in '" + dir_name + "'")
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      file_list.extend(gfile.Glob(file_glob))
    if not file_list:
      tf.logging.warning('No files found')
      continue
    if len(file_list) < 20:
      tf.logging.warning(
          'WARNING: Folder has less than 20 images, which may cause issues.')
    elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
      tf.logging.warning(
          'WARNING: Folder {} has more than {} images. Some images will '
          'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    training_images = []
    testing_images = []
    validation_images = []
    for file_name in file_list:
      base_name = os.path.basename(file_name)
      # We want to ignore anything after '_nohash_' in the file name when
      # deciding which set to put an image in, the data set creator has a way of
      # grouping photos that are close variations of each other. For example
      # this is used in the plant disease data set to group multiple pictures of
      # the same leaf.
      hash_name = re.sub(r'_nohash_.*$', '', file_name)
      # This looks a bit magical, but we need to decide whether this file should
      # go into the training, testing, or validation sets, and we want to keep
      # existing files in the same set even if more files are subsequently
      # added.
      # To do that, we need a stable way of deciding based on just the file name
      # itself, so we do a hash of that and then use that to generate a
      # probability value that we use to assign it.
      hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
      percentage_hash = ((int(hash_name_hashed, 16) %
                          (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                         (100.0 / MAX_NUM_IMAGES_PER_CLASS))
      if percentage_hash < validation_percentage:
        validation_images.append(base_name)
      elif percentage_hash < (testing_percentage + validation_percentage):
        testing_images.append(base_name)
      else:
        training_images.append(base_name)
    result[label_name] = {
        'dir': dir_name,
        'training': training_images,
        'testing': testing_images,
        'validation': validation_images,
    }
  return result


def get_image_path(image_lists, label_name, index, image_dir, category):
  """"Returns a path to an image for a label at the given index.

  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Int offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of set to pull images from - training, testing, or
    validation.

  Returns:
    File system path string to an image that meets the requested parameters.

  """
  if label_name not in image_lists:
    tf.logging.fatal('Label does not exist %s.', label_name)
  label_lists = image_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal('Category does not exist %s.', category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal('Label %s has no images in the category %s.',
                     label_name, category)
  mod_index = index % len(category_list)
  base_name = category_list[mod_index]
  sub_dir = label_lists['dir']
  full_path = os.path.join(image_dir, sub_dir, base_name)
  return full_path


def create_dataset_info(image_dir, testing_percentage, validation_percentage
                            ,cash_dic_dir):
  """Given the dataset setting for trining, returns information about it.

  Args:
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.
    cash_dic_dir: String path to a folder containing subfolder
                                          which containing dictionary
  Returns:
    Dictionary of information about the setting
  """
  if not isinstance(type(testing_percentage), int):
    tf.logging.fatal('testing_percentage does not int type.')
  if not isinstance(type(validation_percentage), int):
    tf.logging.fatal('validation_percentage does not int type.')
  if testing_percentage >= 100 or testing_percentage < 0:
    tf.logging.fatal('testing_percentage is out of range %d.',
                                                testing_percentage)
  if validation_percentage >= 100 or validation_percentage < 0:
    tf.logging.fatal('validation_percentage is out of range %d.',
                                                validation_percentage)

  return {'image_dir': image_dir,
        'testing_percentage' : testing_percentage,
        'validation_percentage': validation_percentage,
        'cash_dic_dir': cash_dic_dir}


def make_index_numpys(image_lists, category):
  """make the index_numpys
  Args:
    image_lists: A dictionary containing an entry for each label subfolder, with
    images split into training, testing, and validation sets within each label.
    category: Name string of which set of images to fetch - training, testing,
    or validation
  return:
    List of numpys which contain label_index and image_index
  """
  index_numpys = []
  class_keys = image_lists.keys()
  class_count = len(class_keys)
  for label_index in range(class_count):
    label_name = list(class_keys)[label_index]
    label_lists = image_lists[label_name]
    if category not in label_lists:
      tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
      tf.logging.fatal('Label %s has no images in the category %s.',
                       label_name, category)
    for image_index in range(len(category_list)):
      index_numpys.append(np.array([label_index, image_index]))

  return index_numpys


class Dataset:
  """ make the batch or sample from image directory
  this class need dataset_info which contain parameter of Dataset

  """
  def __init__(self, dataset_info):
    self.use_cash = False
    self.image_dir = dataset_info['image_dir']
    self.dataset_info = dataset_info
    self.image_lists = self.create_image_lists()
    self.class_num = len(self.image_lists.keys())
    if self.class_num == 0:
      tf.logging.error('No valid folders of images found at ' + self.image_dir)
    if self.class_num == 1:
      tf.logging.error('Only one valid folder of images found at ' +
                       self.image_dir +
                       ' - multiple classes are needed for classification.')

  def create_image_lists(self):
    # Look at the folder structure, and create lists of all the images.
    image_lists = load_image_lists(self.dataset_info['cash_dic_dir'])
    if image_lists != None:
      self.use_cash = True
      return image_lists
    else:
      if True:
        image_lists = create_image_lists_by_percentage(
                                    self.dataset_info['image_dir'],
                                    self.dataset_info['testing_percentage'],
                                    self.dataset_info['validation_percentage'])
        save_image_lists(image_lists, self.dataset_info['cash_dic_dir'])
        return image_lists
      else:
        image_lists = create_image_lists_by_percentage(
                                    self.dataset_info['image_dir'],
                                    self.dataset_info['testing_percentage'],
                                    self.dataset_info['validation_percentage'])
        save_image_lists(image_lists, self.dataset_info['cash_dic_dir'])
        return iamge_lists


  def create_batch(self, batch_size = 1, category = 'training', is_shuffle = True):
    """make the batch generator
    Args:
      batch_size: The integer number of image datas to yield
      category: Name string of which set of images to fetch - training, testing,
      or validation
      is_shuffle: Boolean whether to randomly shuffle images.
    yields:
      List of image_datas and their corresponding ground truths
    """
    image_lists = self.image_lists
    image_dir = self.image_dir
    cash_path = os.path.join(
                    self.dataset_info['cash_dic_dir'],category + '_numpy.csv')
    if self.use_cash == True:
      index_numpys = load_index_numpys(cash_path)

      if index_numpys == None:
        index_numpys = make_index_numpys(image_lists, category)
        save_index_numpys(index_numpys, cash_path)
    else:
      index_numpys = make_index_numpys(image_lists, category)
      save_index_numpys(index_numpys, cash_path)
    class_keys = image_lists.keys()

    while(True):
      if is_shuffle:
        random.shuffle(index_numpys)
      image_datas = []
      ground_truths = []
      filenames = []
      for i, index_array in enumerate(index_numpys, 1):
        label_index = index_array[0]
        image_index = index_array[1]
        label_name = list(class_keys)[label_index]
        image_path = get_image_path(image_lists, label_name, image_index,
                                    image_dir,category)
        if not gfile.Exists(image_path):
          tf.logging.fatal('File does not exist %s', image_path)
        jpeg_data = gfile.FastGFile(image_path, 'rb').read()
        image_datas.append(jpeg_data)
        ground_truths.append(label_index)
        filenames.append(image_path)
        if i % batch_size == 0:
          yield image_datas, ground_truths, filenames
          image_datas = []
          ground_truths =[]
          filenames = []

  def create_sample(self, category, is_shuffle = True):
    """make the sample generator
    Args:
      category: Name string of which set of images to fetch - training, testing,
      or validation
      is_shuffle: Boolean whether to randomly shuffle images.
    yields:
      jpeg_data and their corresponding ground truths
    """
    image_lists = self.image_lists
    image_dir = self.image_dir
    cash_path = os.path.join(
                    self.dataset_info['cash_dic_dir'],category + '_numpy.csv')
    if self.use_cash == True:
      index_numpy = load_index_numpys(cash_path)

      if index_numpy == None:
        index_numpys = make_index_numpys(image_lists, category)
        save_index_numpys(index_numpys, cash_path)
    else:
      index_numpys = make_index_numpys(image_lists, category)
      save_index_numpys(index_numpys, cash_path)
    class_keys = image_lists.keys()
    while(True):
      if is_shuffle:
        random.shuffle(index_numpys)
      for i, index_array in enumerate(index_numpys, 1):
        label_index = index_array[0]
        image_index = index_array[1]
        label_name = list(class_keys)[label_index]
        image_path = get_image_path(image_lists, label_name, image_index,
                                    image_dir,category)
        if not gfile.Exists(image_path):
          tf.logging.fatal('File does not exist %s', image_path)
        jpeg_data = gfile.FastGFile(image_path, 'rb').read()
        yield jpeg_data, label_index, image_name
