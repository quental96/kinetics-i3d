"""Loads a sample video and classifies using a trained Kinetics checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os

import i3d

_IMAGE_SIZE = 224

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, rgb600, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')
tf.flags.DEFINE_string('final_endpoint', 'Logits', 'Mixed_4f, Logits, Predictions, etc.')
tf.flags.DEFINE_string('path', 'temp', '')
tf.flags.DEFINE_string('flow_path', 'temp', '')
tf.flags.DEFINE_string('save_path', 'temp', '')
tf.flags.DEFINE_integer('frames', 0, '')

_PATHS = {
    'rgb': FLAGS.path,
    'flow': FLAGS.flow_path,
}

_SAVE_PATH = FLAGS.save_path

_SAMPLE_VIDEO_FRAMES = FLAGS.frames

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  eval_type = FLAGS.eval_type
  imagenet_pretrained = FLAGS.imagenet_pretrained
  final_endpoint = FLAGS.final_endpoint

  NUM_CLASSES = 400
  if eval_type == 'rgb600':
    NUM_CLASSES = 600

  if eval_type not in ['rgb', 'rgb600', 'flow', 'joint']:
    raise ValueError('Bad `eval_type`, must be one of rgb, rgb600, flow, joint')

  if final_endpoint not in ['Mixed_4f', 'Logits', 'Predictions']:
    raise ValueError('Bad `final_endpoint`, must be one of Mixed_4f, Logits, Predictions')

  if eval_type == 'rgb600':
    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH_600)]
  else:
    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

  if eval_type in ['rgb', 'rgb600', 'joint']:
    # RGB input has 3 channels.
    rgb_input = tf.placeholder(
        tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))


    with tf.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint=final_endpoint)
      rgb_output, _ = rgb_model(
          rgb_input, is_training=False, dropout_keep_prob=1.0)


    rgb_variable_map = {}
    for variable in tf.global_variables():

      if variable.name.split('/')[0] == 'RGB':
        if eval_type == 'rgb600':
          rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
        else:
          rgb_variable_map[variable.name.replace(':0', '')] = variable

    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

  if eval_type in ['flow', 'joint']:
    # Flow input has only 2 channels.
    flow_input = tf.placeholder(
        tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))
    with tf.variable_scope('Flow'):
      flow_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint=final_endpoint)
      flow_output, _ = flow_model(
          flow_input, is_training=False, dropout_keep_prob=1.0)
    flow_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'Flow':
        flow_variable_map[variable.name.replace(':0', '')] = variable
    flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

  if eval_type == 'rgb' or eval_type == 'rgb600':
    model_output = rgb_output
  elif eval_type == 'flow':
    model_output = flow_output
  else:
    model_output = rgb_output + flow_output

  with tf.Session() as sess:
    feed_dict = {}
    if eval_type in ['rgb', 'rgb600', 'joint']:
      if imagenet_pretrained:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
      else:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
      tf.logging.info('RGB checkpoint restored')
      rgb_sample = np.load(_PATHS['rgb'])
      rgb_sample = rgb_sample[np.newaxis, ...]
      tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
      feed_dict[rgb_input] = rgb_sample

    if eval_type in ['flow', 'joint']:
      if imagenet_pretrained:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
      else:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
      tf.logging.info('Flow checkpoint restored')
      flow_sample = np.load(_PATHS['flow'])
      flow_sample = flow_sample[np.newaxis, ...]
      tf.logging.info('Flow data loaded, shape=%s', str(flow_sample.shape))
      feed_dict[flow_input] = flow_sample

    output = sess.run(model_output, feed_dict=feed_dict)
    np.save(_SAVE_PATH, output)


if __name__ == '__main__':
  tf.app.run(main)