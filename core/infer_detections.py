#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 19:08:18 2017

@author: avanetten

Lightly adapted from:
    https://github.com/tensorflow/models/blob/master/research/object_detection/inference/infer_detections.py
    and 
    https://github.com/tensorflow/models/blob/master/research/object_detection/inference/detection_inference.py
    Also see:
    https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

"""

from __future__ import print_function

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Infers detections on a TFRecord of TFExamples given an inference graph.
Example usage:
  ./infer_detections \
    --input_tfrecord_paths=/path/to/input/tfrecord1,/path/to/input/tfrecord2 \
    --output_tfrecord_path_prefix=/path/to/output/detections.tfrecord \
    --inference_graph=/path/to/frozen_weights_inference_graph.pb
The output is a TFRecord of TFExamples. Each TFExample from the input is first
augmented with detections from the inference graph and then copied to the
output.
The input and output nodes of the inference graph are expected to have the same
types, shapes, and semantics, as the input and output nodes of graphs produced
by export_inference_graph.py, when run with --input_type=image_tensor.
The script can also discard the image pixels in the output. This greatly
reduces the output size and can potentially accelerate reading data in
subsequent processing steps that don't require the images (e.g. computing
metrics).
"""


######################
# Not sure why the line below fails, so import manually...
#from object_detection.inference import detection_inference
import os
import sys
sys.path.append('/opt/tensorflow-models/research/object_detection/inference')
import detection_inference
######################

#from tensorflow.python.platform import gfile
import itertools
import tensorflow as tf
import cv2
import csv
import numpy as np
import time
#from PIL import Image


tf.flags.DEFINE_string('inference_graph', None,
                       'Path to the inference graph with embedded weights.')
tf.flags.DEFINE_boolean('verbose', False, 'Lots o print statements')
tf.flags.DEFINE_boolean('use_tfrecords', True, 'Switch to use tfrecords')


# tfrecords
tf.flags.DEFINE_string('input_tfrecord_paths', None,
                       'A comma separated list of paths to input TFRecords.')
tf.flags.DEFINE_string('output_tfrecord_path', None,
                       'Path to the output TFRecord.')
tf.flags.DEFINE_boolean('discard_image_pixels', True,
                        'Discards the images in the output TFExamples. This'
                        ' significantly reduces the output size and is useful'
                        ' if the subsequent tools don\'t need access to the'
                        ' images (e.g. when computing evaluation measures).')

 # if outputting a dataframe rather than tfrecord
tf.flags.DEFINE_string('input_file_list', None,
                       'A comma separated list of paths to sliced images')
tf.flags.DEFINE_string('output_csv_path', None,
                       'Path to the output dataframe'
                       "output_columns = ['Loc_Tmp', u'Prob', u'Xmin', u'Ymin', u'Xmax', u'Ymax', u'Category_Int']"
                       )
tf.flags.DEFINE_float('min_thresh', 0.05,
                        'Minumum score to retain')
tf.flags.DEFINE_integer('GPU', 0,
                        'Which GPU to use')
tf.flags.DEFINE_integer('BGR2RGB', 0,
                        'Sometimes we need to change cv2 images to BGR')



FLAGS = tf.flags.FLAGS



###############################################################################
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
  
###############################################################################
def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  required_flags = ['inference_graph']
  #required_flags = ['input_tfrecord_paths', 'output_tfrecord_path',
  #                  'inference_graph']
  for flag_name in required_flags:
    if not getattr(FLAGS, flag_name):
      raise ValueError('Flag --{} is required'.format(flag_name))

  t0 = time.time()
  #config = tf.ConfigProto(device_count = {'GPU': FLAGS.GPU})
  #with tf.device('/gpu:'+str(FLAGS.GPU))
  # original version, outputs a tfrecord

  if FLAGS.use_tfrecords:
  #if FLAGS.output_tfrecord_path:
  
    # INFO:tensorflow:Time to process records 62.7291879654 seconds
    #with tf.Session(config=config) as sess:
    with tf.Session() as sess:

        input_tfrecord_paths = [
            v for v in FLAGS.input_tfrecord_paths.split(',') if v]
        tf.logging.info('Reading input from %d files', len(input_tfrecord_paths))
        serialized_example_tensor, image_tensor = detection_inference.build_input(
            input_tfrecord_paths)
        
        tf.logging.info('Reading graph and building model...')
        (detected_boxes_tensor, detected_scores_tensor,
         detected_labels_tensor) = detection_inference.build_inference_graph(
             image_tensor, FLAGS.inference_graph)
    
        tf.logging.info('Running inference and writing output to {}'.format(
            FLAGS.output_tfrecord_path))
        sess.run(tf.local_variables_initializer())
        tf.train.start_queue_runners()
    
        with tf.python_io.TFRecordWriter(
            FLAGS.output_tfrecord_path) as tf_record_writer:
          try:
            for counter in itertools.count():
              tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 10,
                                     counter)
              
              tf_example = detection_inference.infer_detections_and_add_to_example(
                  serialized_example_tensor, detected_boxes_tensor,
                  detected_scores_tensor, detected_labels_tensor,
                  FLAGS.discard_image_pixels)
              tf_record_writer.write(tf_example.SerializeToString())
                     
          except tf.errors.OutOfRangeError:
            tf.logging.info('Finished processing records')
            t1 = time.time()
            tf.logging.info('Time to process records ' + str(t1 - t0) +  ' seconds')
  
  else:

    if FLAGS.verbose:
        print ("min_thresh:", FLAGS.min_thresh)
    t0 = time.time()


    # define inference graph
    inference_graph = tf.Graph()
    with inference_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.FastGFile(FLAGS.inference_graph, 'rb') as fid:
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name='')
    print ("Time to load graph:", time.time() - t0, "seconds")
    
    with open(FLAGS.output_csv_path, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        output_columns =  ['Loc_Tmp', u'Prob', u'Xmin', u'Ymin', u'Xmax', u'Ymax', u'Category']    
        csvwriter.writerow(output_columns)
        
        #df_data = []
        with inference_graph.as_default():
          with tf.Session(graph=inference_graph) as sess:
            #print('tf.sessionPassed')
            # get image paths
            with open(FLAGS.input_file_list, 'rb') as f:
                image_paths = f.readlines()
                
            line_count = 0
            for i,image_path in enumerate(image_paths):
                
              image_root = os.path.basename(image_path).strip()
              #print ("image_root:", image_root)
              #if image_root != '12TVK220980-CROP__0_0_416_416_0_3386_3386.png':
              #    continue
                                  
              #if (i % 10) == 0:
              #    print ("i:", i, "image_root:", image_root)
                  
              image_bgr = cv2.imread(image_path.strip(), 1)
              # invert colors, if required
              if FLAGS.BGR2RGB == 1:
                  image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
              else:
                  image = image_bgr
                  
                 
              height, width = image.shape[:2]
              # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
              image_expanded = np.expand_dims(image, axis=0)
              image_tensor = inference_graph.get_tensor_by_name('image_tensor:0')
              boxes = inference_graph.get_tensor_by_name('detection_boxes:0')
              scores = inference_graph.get_tensor_by_name('detection_scores:0')
              classes = inference_graph.get_tensor_by_name('detection_classes:0')
              # Perform detection
              (detected_boxes, detected_scores, detected_classes) = sess.run(
                  [boxes, scores, classes],
                      feed_dict={image_tensor: image_expanded})
    
              # box info
              detected_boxes = detected_boxes.T
              ymins = detected_boxes[0].T
              xmins = detected_boxes[1].T
              ymaxs = detected_boxes[2].T
              xmaxs = detected_boxes[3].T
              
              if FLAGS.verbose:
                  print ("\nimage path:", image_path.strip())
                  print ("all scores:", detected_scores)
                  print ("all scores shape:", detected_scores.shape)
                  print ("xmins:", xmins)
                  print ("xmins shape:", xmins.shape)

#              # flatten, if correct shape
#              for arr in [ymins, xmins, ymaxs, xmaxs, detected_scores, 
#                          detected_classes]:
#                  if arr.shape != (1, 100):
#                      print ("array shape is not (1,100), stopping!")
#                      print ("  arr.shape:", arr.shape)
#                      return
                  
              # flatten, convert positions from fractions to pixel coords
              ymins = (height * (ymins.flatten())).astype(int)
              xmins = (width  * (xmins.flatten())).astype(int)
              ymaxs = (height * (ymaxs.flatten())).astype(int)
              xmaxs = (width  * (xmaxs.flatten())).astype(int)
              detected_scores = detected_scores.flatten()
              detected_classes = detected_classes.flatten()
              locs = len(xmins) * [image_path.strip()]
  
              # filter out low scores?
              good_idxs = np.where(detected_scores >= float(FLAGS.min_thresh))
              detected_scores = detected_scores[good_idxs]
              detected_classes = detected_classes[good_idxs]
              ymins = ymins[good_idxs]
              xmins = xmins[good_idxs]
              ymaxs = ymaxs[good_idxs]
              xmaxs = xmaxs[good_idxs]
              locs = len(xmins) * [image_path.strip()]
              
              #if (len(xmins) > 0):
              #    print ("float(FLAGS.plot_thresh)", float(FLAGS.plot_thresh))
              #    print ("detected scores:", detected_scores)
              #    print ("good idxs:", good_idxs)
                  
              # print progress
              if (i % 10) == 0:
                 # print ("  i",  i, "line_count", line_count, image_path.strip())
                  print ("i:", i, "line_count:", line_count, 
                         "image_root:", image_root)
                  if (len(xmins) > 0) and  (2 < 1):
                      print ("    nlines:", line_count, "classes0, scores0", 
                             "xmins0, ymins0, xmaxs0, ymaxs0",
                             detected_classes[0], 
                             detected_scores[0], xmins[0], ymins[0], 
                             xmaxs[0], ymaxs[0])

              # write to file
              # output_columns =  ['Loc_Tmp', u'Prob', u'Xmin', u'Ymin', u'Xmax', u'Ymax', u'Category']    
              for row in zip(locs, detected_scores, xmins, ymins, xmaxs, ymaxs, detected_classes):
                   line_count += 1
                   csvwriter.writerow(row)

#              # update data by only keeping high scores (slow but very explicit)
#              # output_columns =  ['Loc_Tmp', u'Prob', u'Xmin', u'Ymin', u'Xmax', u'Ymax', u'Category']
#              for j in range(len(xmins)):
#                  score = detected_scores[j]
#                  if FLAGS.verbose:
#                      print ("score:", score)
#                  if score >= plot_thresh:
#                       out_row = [image_path,  #img_loc_string[j],
#                         score,
#                         xmins[j],
#                         ymins[j],
#                         xmaxs[j],
#                         ymaxs[j],
#                         detected_classes[j]
#                       ]
#                       #df_data.append(out_row)
#                       csvwriter.writerow(out_row)
  
    t1 = time.time()
    print ('Time to process', line_count,  'records', t1 - t0, 'seconds')


if __name__ == '__main__':
  tf.app.run()
