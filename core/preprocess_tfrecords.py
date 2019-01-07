#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:26:53 2017

@author: avanetten

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py

https://va-vsrv-github.a.internal/Cosmiq/spacenetTensorflow/blob/master/create_geojson_airplane_tf_record_updated.py

https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/examples/09_tfrecord_example.py

def get_image_binary(filename):
    #You can read in the image using tensorflow too, but it's a drag
    #since you have to create graphs. It's much easier using Pillow and NumPy
    image = Image.open(filename)
    image = np.asarray(image, np.uint8)
    shape = np.array(image.shape, np.int32)
    return shape.tobytes(), image.tobytes() # convert image to raw data bytes in the array.

"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import os
import cv2
import pandas as pd
import tensorflow as tf
import argparse
import random
import time

#import logging
#from lxml import etree
#import PIL.Image
#import io

###############################################################################
### Dataset utils
# https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py
#import sys
#path_to_utils = '/Users/avanetten/Documents/cosmiq/git/tensorflow_models-master/research/object_detection/utils'
#sys.path.append(path_to_utils)
#import dataset_util

## or if already in path
#from object_detection.utils import dataset_util
#from object_detection.utils import label_map_util

"""Utility functions for creating TFRecord data sets."""

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
###############################################################################

###############################################################################  
# https://github.com/tensorflow/models/blob/master/research/object_detection/utils/label_map_util.py
#   annoying to get up and running, just create a parser instead
def load_pbtxt(pbtxt_filename, verbose=False):
    '''Read pbtxt file of format: 
        item {
                id: 1
                name: 'airplane'
             }
    output a dictionary of {1: 'airplane', 2: 'car', ....}
    Class integers must start at 1 (not 0) !!!!
    '''
    if verbose:
        print ("\nRun load_pbtxt in preprocess_tfrecords.py")
    try:
        f = open(pbtxt_filename, "r")
    except:
        return {0:0}
    label_dict = {}
    prev_line = ''
    for i,line_raw in enumerate(f.readlines()):
        line = line_raw.strip()
        if (prev_line.startswith('id:')) and (line.startswith('name:')):
            idx = int(prev_line.split('id: ')[-1])
            label = line.split('name: ')[-1].replace("'", "")
            label_dict[idx] = label
            if verbose:
                print("line:", line)
                print (" prev_line:", prev_line)
                print (" label:", label)
        prev_line = line
        
    if verbose:
        print("label_dict:", label_dict)
    f.close()
    
    return label_dict
###############################################################################        


###############################################################################
# from convert.py
def convert_reverse(size, box):
    '''Back out pixel coords from yolt format
    input = image_size, [x,y,w,h]
    image_size = [width, height] '''
    x,y,w,h = box
    dw = 1./size[0]
    dh = 1./size[1]
    
    w0 = w/dw
    h0 = h/dh
    xmid = x/dw
    ymid = y/dh
    
    x0, x1 = xmid - w0/2., xmid + w0/2.
    y0, y1 = ymid - h0/2., ymid + h0/2.

    return [x0, x1, y0, y1]
    
###############################################################################
# from convert.py
def convert_bbox_yolt_to_tf(height, width, yolt_row):
    '''yolt stores in format: [category xmid_frac, ymid_frac, width_frac, 
                                height_frac]
    Tensorflow object detection api wants things in format:
        [x0_frac, y0_frac, x1_frac, y1_frac]'''
        
    cat_int, x_frac, y_frac, width_frac, height_frac = yolt_row
    box = [x_frac, y_frac, width_frac, height_frac]
    
    [x0, x1, y0, y1] = convert_reverse((width, height), box)
    
    xmin, xmax = 1.*x0 / width,  1.*x1 / width
    ymin, ymax = 1.*y0 / height, 1.*y1 / height
    
    xmin_out = max([0.0001, xmin])
    xmax_out = min([0.9999, xmax])
    ymin_out = max([0.0001, ymin])
    ymax_out = min([0.9999, ymax])   
    
    return [xmin_out, xmax_out, ymin_out, ymax_out]

###############################################################################
def yolt_to_tf_example(image_file, label_file, 
                       label_map_dict,
                       convert_dict={},
                       #cat_int_plus=1,
                       ignore_difficult_instances=False,
                       labelfile_columns=['cat_int', 'x_frac', 'y_frac', 
                                          'width_frac', 'height_frac'],
                       verbose=False):
  """
  Adapted from:
      https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py
    
  Create tfrecord from yolt image_flle and label_file
  
  Tensorflow object detection api requires jpeg format !
  
  convert_dict maps yolt internal labels to the integers for .pbtxt 
  
  """
  
  if verbose:
      print("image_file:", image_file)
  
  # read image file
  im = cv2.imread(image_file, 1)
  height, width = im.shape[:2]
  
  if verbose:
      print("image shape:", im.shape)
  
  
  with tf.gfile.GFile(image_file, 'rb') as fid:
    encoded_jpg = fid.read()
  key = hashlib.sha256(encoded_jpg).hexdigest()
  
  #img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
  #full_path = os.path.join(dataset_directory, img_path)
  #with tf.gfile.GFile(full_path, 'rb') as fid:
  #  encoded_jpg = fid.read()
  #encoded_jpg_io = io.BytesIO(encoded_jpg)
  #image = PIL.Image.open(encoded_jpg_io)
  #if image.format != 'JPEG':
  #  raise ValueError('Image format not JPEG')
  #key = hashlib.sha256(encoded_jpg).hexdigest()
  #width = int(data['size']['width'])
  #aheight = int(data['size']['height'])


  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  #truncated = []
  #poses = []
  #difficult_obj = []

  if len(label_file) > 0:
      # read label file
      df = pd.read_csv(label_file, sep=' ', names=labelfile_columns)
      #data = df.values
    
      #for obj in data['object']:
      for idx,row in df.iterrows():
        
        cat_int, x_frac, y_frac, width_frac, height_frac = row
        #box = [x_frac, y_frac, width_frac, height_frac]
        # get pixel coords
        [x0, x1, y0, y1] = convert_bbox_yolt_to_tf(height, width, row)
        
        ## difficult objects?
        #difficult = False #bool(int(obj['difficult']))
        #if ignore_difficult_instances and difficult:
        #  continue
        #difficult_obj.append(int(difficult))
    
        xmin.append(x0) #float(obj['bndbox']['xmin']) / width)
        ymin.append(y0) #float(obj['bndbox']['ymin']) / height)
        xmax.append(x1) #float(obj['bndbox']['xmax']) / width)
        ymax.append(y1) #float(obj['bndbox']['ymax']) / height)
        #cat_int_out = cat_int + cat_int_plus
        if len(convert_dict.keys()) > 0:
            cat_int_out = convert_dict[cat_int]
        else:
            cat_int_out = cat_int
        classes.append(int(cat_int_out)) #cd dlabel_map_dict[obj['name']])
        classes_text.append(label_map_dict[cat_int_out]) #obj['name'].encode('utf8'))
        #truncated.append(0) #int(obj['truncated']))
        #poses.append(0) #obj['pose'].encode('utf8'))

  if verbose:
      print("  len objects:", len(xmin))
      print("  classes:", classes)
      print("  classes_text:", classes_text)

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
      'image/filename': bytes_feature(
          image_file.encode('utf8')),
      'image/source_id': bytes_feature(
          image_file.encode('utf8')),
      'image/key/sha256': bytes_feature(key.encode('utf8')),
      'image/encoded': bytes_feature(encoded_jpg),
      'image/format': bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': float_list_feature(xmin),
      'image/object/bbox/xmax': float_list_feature(xmax),
      'image/object/bbox/ymin': float_list_feature(ymin),
      'image/object/bbox/ymax': float_list_feature(ymax),
      'image/object/class/text': bytes_list_feature(classes_text),
      'image/object/class/label': int64_list_feature(classes),
      #'image/object/difficult': int64_list_feature(difficult_obj),
      #'image/object/truncated': int64_list_feature(truncated),
      #'image/object/view': int64_list_feature(poses),

#  example = tf.train.Example(features=tf.train.Features(feature={
#      'image/height': dataset_util.int64_feature(height),
#      'image/width': dataset_util.int64_feature(width),
#      'image/filename': dataset_util.bytes_feature(
#          image_file.encode('utf8')),
#      'image/source_id': dataset_util.bytes_feature(
#          image_file.encode('utf8')),
#      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
#      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
#      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
#      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
#      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
#      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
#      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
#      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
#      'image/object/class/label': dataset_util.int64_list_feature(classes),
#      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
#      'image/object/truncated': dataset_util.int64_list_feature(truncated),
#      'image/object/view': dataset_util.int64_list_feature(poses),

  }))
      
  return example


###############################################################################
def test_image_tf_example(image_file, verbose=False):
  """
  Create tfrecord from test image_file
  Tensorflow object detection api requires jpeg or png format!  
  """
  if verbose:
      print("  image_file:", image_file)
    
  # read image file
  im = cv2.imread(image_file, 1)
  height, width = im.shape[:2]
  
  if verbose:
      print("image shape:", im.shape)
  
  # encode jpg
  with tf.gfile.GFile(image_file, 'rb') as fid:
    encoded_jpg = fid.read()
  key = hashlib.sha256(encoded_jpg).hexdigest()
  
  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
      'image/filename': bytes_feature(
          image_file.encode('utf8')),
      'image/source_id': bytes_feature(
          image_file.encode('utf8')),
      'image/key/sha256': bytes_feature(key.encode('utf8')),
      'image/encoded': bytes_feature(encoded_jpg),
      'image/format': bytes_feature('jpeg'.encode('utf8')),
      }))
      
  return example

###############################################################################
def yolt_dir_to_tf(image_dir, label_map_dict, TF_RecordPath, #cat_int_plus=1,
                   convert_dict={},
                   verbose=False):
    
    '''Create tfrecord from yolt labels/images directories'''
    
    writer = tf.python_io.TFRecordWriter(TF_RecordPath)

    image_files = [p for p in os.listdir(image_dir) if p.endswith('.tif')]
    for image_file in image_files:
        image_file_full = os.path.join(image_dir, image_file)
        label_file_full = image_file_full.split('.')[0].replace('images', 'labels') + '.txt'
        tf_example = yolt_to_tf_example(image_file_full, label_file_full, 
                       label_map_dict,
                       #cat_int_plus=cat_int_plus,
                       convert_dict=convert_dict,
                       ignore_difficult_instances=False,
                       verbose=verbose)
        if tf_example:
            writer.write(tf_example.SerializeToString())
            
    writer.close()
   
###############################################################################
def yolt_imlist_to_tf(image_list_file, label_map_dict, TF_RecordPath,
                      TF_PathVal='', val_frac=0.0, 
                      convert_dict={}, verbose=False):
    
    '''Create tfrecord from yolt labels/images directories'''
    
    if verbose:
        print("\nIngesting", image_list_file, "...\n")
        
    # shuffle    
    floc = open(image_list_file, 'rb')
    lines = floc.readlines()
    random.shuffle(lines)
    # take first few records as validation files
    n_val = int(val_frac * len(lines))
    
    writer = tf.python_io.TFRecordWriter(TF_RecordPath)
    if (val_frac > 0) and len(TF_PathVal)> 0:
        writer_val = tf.python_io.TFRecordWriter(TF_PathVal)
    for i,image_file in enumerate(lines):  #floc.readlines():
        image_file_full = image_file.strip()
        if verbose:
            print("image_file:", image_file_full)
            print("  os.path.exists(image_file?", os.path.exists(image_file_full))
        label_file_full = image_file_full.split('.')[0].replace('images', 'labels') + '.txt'
        if not os.path.exists(label_file_full):
            label_file_full = ''
        tf_example = yolt_to_tf_example(image_file_full, label_file_full, 
                       label_map_dict,                       
                       #cat_int_plus=cat_int_plus,
                       convert_dict=convert_dict,
                       ignore_difficult_instances=False,
                       verbose=verbose)
        if tf_example:
            if (len(TF_PathVal) > 0) and i < n_val:
                writer_val.write(tf_example.SerializeToString())
            else:   
                writer.write(tf_example.SerializeToString())
            
    floc.close()
    writer.close()
    if (val_frac > 0) and len(TF_PathVal)> 0:
        writer_val.close()
    
    print ("len records:", len(lines))
    return
    
###############################################################################
def testlist_to_tf(image_list_file,  TF_RecordPath, verbose=False):
    
    '''Create tfrecord from list of test images'''
    
    if verbose:
        print("\nIngesting", image_list_file, "...\n")
        
    # shuffle  
    t0 = time.time()
    floc = open(image_list_file, 'rb')
    lines = floc.readlines()
    #random.shuffle(lines)
    
    writer = tf.python_io.TFRecordWriter(TF_RecordPath)
    for i,image_file in enumerate(lines):  #floc.readlines():
        image_file_full = image_file.strip()
        if verbose:
            print("image_file_full:", image_file_full)
            print("  os.path.exists(image_file_full?", os.path.exists(image_file_full))
        tf_example =  test_image_tf_example(image_file_full, verbose=verbose)
        if (tf_example):
            writer.write(tf_example.SerializeToString())
            
    floc.close()
    writer.close()
    
    print ("Time to run testlist_to_tf():", time.time()  - t0, "seconds")
    
    return

###############################################################################

    
###############################################################################    
###############################################################################     
def main():

    verbose = True
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('--image_list_file', type=str, default='/Users/avanetten/Documents/cosmiq/simrdwn/test_list.txt',
    parser.add_argument('--image_list_file', type=str, default='/Users/avanetten/Documents/cosmiq/qgis_labels/parse_shapefile/outputs/qgis_labels_car_boat_plane_list.txt',  
                      help="File holding locations of image files")
    parser.add_argument('--outfile', type=str, default='/Users/avanetten/Documents/cosmiq/simrdwn/data/qgis_labels_car_boat_plane.tfrecord',
                        help="Output file location")
    parser.add_argument('--outfile_val', type=str, default='/Users/avanetten/Documents/cosmiq/simrdwn//data/qgis_labels_car_boat_plane_val.tfrecord',
                        help="Output validation file location")
    parser.add_argument('--pbtxt_filename', type=str, default='/Users/avanetten/Documents/cosmiq/simrdwn/data/class_labels_airplane_boat_car.pbtxt',
                        help="Class dictionary")
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help="Fraction of items to reserve for validation") 
    #parser.add_argument('--convert_dict', type=json.loads, default=)
    #  https://stackoverflow.com/questions/18608812/accepting-a-dictionary-as-an-argument-with-argparse-and-python
    #parser.add_argument('--cat_int_plus', type=int, default=0,
    #                    help="The value to add to the integers in label_file."\
    #                    + " This is necessary because the tensorflow object" \
    #                    + " detection api requires categories to be 1-indexed.")  

    parser.add_argument('--test_list', type=str, default='',
                        help="Testlist of images from which to create tfrecord") 
    parser.add_argument('--test_list_tfrecord_out', type=str, default='',
                        help="output tfrecord of test list") 

    args = parser.parse_args()

    if len(args.test_list) > 0:
        testlist_to_tf(args.test_list, args.test_list_tfrecord_out, 
                       verbose=False)
        return


    # set outdir
    outdir = os.path.dirname(args.outfile)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        
        
    # set label dictionary (
    # .pbtxt must start at 1, not 0!
    # for airplane, boat, car we want airplane: 1, boat: 2, car: 3
    label_map_dict = load_pbtxt(args.pbtxt_filename, verbose=verbose)
    
    # if data prep was done with different labels, set up a conversion dict
    # tf record labels must start with 1, not zero
    if args.pbtxt_filename.endswith('class_labels_airplane_boat_car.pbtxt'):
        convert_dict = {1:  1,  # airplane
                    3:  2,  # boat     
                    5:  3,  #' car'
                   }
    elif args.pbtxt_filename.endswith('class_labels_airport.pbtxt'):
        convert_dict = {0:  1,  # airport
                   }
    # usually we just want to increase the number by one
    else:
        convert_dict = {x:x+1 for x in range(100)}
        #convert_dict = {0:  1,  
        #                1:  2,     
        #                2:  3,     
        #                3:  4,     
        #                4:  5,     
        #                5:  6,     
        #                6:  7,     
        #                7:  8,     
        #                8:  9,     
        #                9:  10
        #           }
        

    #label_map_dict = {0:    'airplane',
    #                  1:    'airport',
    #                  2:    'boat',
    #                  3:    'boat_harbor',
    #                  4:    'car'}
    
    ##out_dir =   '/Users/avanetten/Documents/cosmiq/simrdwn/'    
    ## make temporary image list to test out yolt_imlist_to_tf()
    #image_dir = '/Users/avanetten/Documents/cosmiq/qgis_labels/parse_shapefile/outputs/WV03_03102015_R1C2/images'
    #image_list_file = os.path.join(outdir, 'test_list.txt')
    #file_out = open(image_list_file, "w")
    #im_list = os.listdir(image_dir)
    #for im in im_list:
    #    path_tot = os.path.join(image_dir, im)
    #    #line_out = path_tot.replace('|', '\|') + '\n'
    #    line_out = path_tot + '\n'
    #    file_out.write(line_out)
    #    #print("line_out:", line_out)
    #file_out.close()
    
        
    #yolt_dir_to_tf(image_dir, label_map_dict, TF_RecordPath, verbose=verbose)
    yolt_imlist_to_tf(args.image_list_file, label_map_dict, args.outfile,
                      TF_PathVal=args.outfile_val, val_frac=args.val_frac, 
                      convert_dict=convert_dict, verbose=verbose)  
    
    print ("Outputs:", args.outfile, args.outfile_val)
    
###############################################################################
###############################################################################
if __name__ == '__main__':
    main()


'''
python /raid/local/src/simrdwn/core/preprocess_tfrecords.py \
    --image_list_file /raid/local/src/simrdwn/data/qgis_labels_airplane_boat_car_devbox_list2.txt \
    --pbtxt_filename /raid/local/src/simrdwn/data/class_labels_airplane_boat_car.pbtxt \
    --outfile /raid/local/src/simrdwn/data/qgis_labels_airplane_boat_car_train2.tfrecord \
    --outfile_val /raid/local/src/simrdwn/data/qgis_labels_airplane_boat_car_val2.tfrecord \
    --val_frac 0.05
'''
    
