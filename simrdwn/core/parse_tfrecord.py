#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 19:23:34 2017

@author: avanetten


Adapted from:
https://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

"""


from __future__ import print_function
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import time
import os
import post_process
import preprocess_tfrecords


###############################################################################
def tf_to_df(tfrecords_filename, label_map_dict={},
             max_iter=50000, tf_type='test',
             output_columns=['Loc_Tmp', u'Prob', u'Xmin',
                             u'Ymin', u'Xmax', u'Ymax', u'Category'],
             ):
    """
    Convert inference tfrecords file to pandas dataframe.

    Arguments
    ---------
    tfrecords_filename : str
        Location of .tfrecord
    label_map_dict : dict
        Dictionary mapping category strings to int. Defaults to ``{}``.
    max_iter : int
        Maximum number of records to ingest. Defaults to ``50000``.
    tf_type : str
        Type of tfrecord.  If tf_type=test, assume this is an inference
        tfrecord. Else, it will be a training tfrecord. Defaults to ``'test'``.
    output_columns : list
        Column names for output dataframe, Defaults to
        ['Loc_Tmp', u'Prob', u'Xmin', u'Ymin', u'Xmax', u'Ymax', u'Category']

    Returns
    -------
    df : pandas dataframe
        Dataframe corresponding to the input tfrecord
    """

    t0 = time.time()

    print("\nTransforming tfrecord to dataframe...")
    df_data = []
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    for i, string_record in enumerate(record_iterator):

        example = tf.train.Example()
        example.ParseFromString(string_record)

        if i == 0:
            print("example.features.feature.keys()",
                  sorted(example.features.feature.keys()))

        if i > max_iter:
            break

        height = int(example.features.feature['image/height']
                                     .int64_list.value[0])
        width = int(example.features.feature['image/width']
                                    .int64_list.value[0])

        #a = example.features.feature['image/detection/score']
        if tf_type == 'test':
            xmins = np.array(example.features.feature['image/detection/bbox/xmin']
                             .float_list.value)
            ymins = np.array(example.features.feature['image/detection/bbox/ymin']
                             .float_list.value)
            xmaxs = np.array(example.features.feature['image/detection/bbox/xmax']
                             .float_list.value)
            ymaxs = np.array(example.features.feature['image/detection/bbox/ymax']
                             .float_list.value)
            classes_int = np.array(example.features.feature['image/detection/label']
                                   .int64_list.value)
            scores = np.array(example.features.feature['image/detection/score']
                              .float_list.value)

        else:
            xmins = np.array(example.features.feature['image/object/bbox/xmin']
                             .float_list.value)
            ymins = np.array(example.features.feature['image/object/bbox/ymin']
                             .float_list.value)
            xmaxs = np.array(example.features.feature['image/object/bbox/xmax']
                             .float_list.value)
            ymaxs = np.array(example.features.feature['image/object/bbox/ymax']
                             .float_list.value)
            classes_int = np.array(example.features.feature['image/object/class/label']
                                   .int64_list.value)
            scores = np.ones(len(classes_int))

        # convert from fractions to pixel coords
        xmins = (width * xmins).astype(int)
        xmaxs = (width * xmaxs).astype(int)
        ymins = (height * ymins).astype(int)
        ymaxs = (height * ymaxs).astype(int)
        #boxes = np.stack((ymins, xmins, ymaxs, xmaxs), axis=1)
        classes_int_str = classes_int.astype(str)
        # classes_text = np.array(example.features.feature['image/detection/class/text']
        #                             .bytes_list.value)

        classes_str, classes_legend_str = classes_int_str, classes_int_str
        if len(label_map_dict.keys()) > 0:
            classes_str = [label_map_dict[ztmp] for ztmp in classes_int]
            classes_legend_str = [
                str(ztmp) + ' = ' + label_map_dict[ztmp] for ztmp in classes_int]

        img_loc_string = (str(example.features.feature['image/filename']
                              .bytes_list.value[0]))
        # .bytes_list.value[0].encode())
        if (i % 100) == 0:
            print("\n", i, "Image Location:", img_loc_string)
            print("  xmins:", xmins)
            print("  classes_str:", classes_str)
        #print ("classes_int:", classes_int)

        # update data
        for j in range(len(xmins)):
            out_row = [img_loc_string,  # img_loc_string[j],
                       scores[j],
                       xmins[j],
                       ymins[j],
                       xmaxs[j],
                       ymaxs[j],
                       # classes_int_str[j]]
                       classes_str[j]]

            df_data.append(out_row)

#        img_string = (example.features.feature['image/encoded']
#                                      .bytes_list
#                                      .value[0])
#        img_1d = np.fromstring(img_string, dtype=np.uint8)
#        reconstructed_img = img_1d.reshape((height, width, 3))
#        annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)
#        # Annotations don't have depth (3rd dimension)
#        reconstructed_annotation = annotation_1d.reshape((height, width))
#        reconstructed_images.append((reconstructed_img, reconstructed_annotation))

    df = pd.DataFrame(df_data, columns=output_columns)
    #print("\ndf_init.columns:", df_init.columns)

    print("len dataframe:", len(df))
    print("Time to transform", len(df), "tfrecords to dataframe:",
          time.time() - t0, "seconds")
    return df


###############################################################################
###############################################################################
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--tfrecords_filename', type=str, default='/cosmiq/simrdwn/results/test/val_detections_ssd.tfrecord',
                        help="tfrecords file")
    parser.add_argument('--outdir', type=str, default='/cosmiq/simrdwn/results/test/images_ssd',
                        help="Output file location")
    parser.add_argument('--pbtxt_filename', type=str, default='/cosmiq/simrdwn/data/class_labels_airplane_boat_car.pbtxt',
                        help="Class dictionary")
    parser.add_argument('--tf_type', type=str, default='test',
                        help="weather the tfrecord is for test or train")
    parser.add_argument('--slice_val_images', type=int, default=0,
                        help="Switch for if validaion images are sliced")
    parser.add_argument('--verbose', type=int, default=0,
                        help="Print a lot o stuff?")

    # Plotting settings
    parser.add_argument('--plot_thresh', type=float, default=0.33,
                        help="Threshold for plotting boxes, set < 0 to skip plotting")
    parser.add_argument('--nms_overlap_thresh', type=float, default=0.5,
                        help="IOU threshold for non-max-suppresion, skip if < 0")
    parser.add_argument('--make_box_labels', type=int, default=1,
                        help="If 1, make print label above each box")
    parser.add_argument('--scale_alpha', type=int, default=1,
                        help="If 1, scale box opacity with confidence")
    parser.add_argument('--plot_line_thickness', type=int, default=1,
                        help="If 1, scale box opacity with confidence")

    args = parser.parse_args()
    print("args:", args)
    t0 = time.time()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    # make label_map_dic (key=int, value=str), and reverse
    label_map_dict = preprocess_tfrecords.load_pbtxt(
        args.pbtxt_filename, verbose=False)
    #label_map_dict_rev = {v: k for k,v in label_map_dict.iteritems()}

    # convert tfrecord to dataframe
    df_init0 = tf_to_df(tfrecords_filename=args.tfrecords_filename,
                        label_map_dict=label_map_dict,
                        tf_type=args.tf_type)
    # df_init = tf_to_df(tfrecords_filename=args.tfrecords_filename,
    #            outdir=args.outdir, plot_thresh=args.plot_thresh,
    #            label_map_dict=label_map_dict,
    #            show_labels = bool(args.make_box_labels),
    #            alpha_scaling = bool(args.scale_alpha),
    #            plot_line_thickness=args.plot_line_thickness)
    t1 = time.time()
    print("Time to run tf_to_df():", t1-t0, "seconds")
    print("df_init.columns:", df_init0.columns)

    # filter out low confidence detections
    df_init = df_init0.copy()[df_init0['Prob'] >= args.plot_thresh]

    # augment dataframe columns
    df = post_process.augment_df(df_init,
                                 valid_testims_dir_tot='',
                                 slice_sizes=[0],
                                 valid_slice_sep='__',
                                 edge_buffer_valid=0,
                                 max_edge_aspect_ratio=4,
                                 valid_box_rescale_frac=1.0,
                                 rotate_boxes=False,
                                 verbose=bool(args.verbose))
    print("len df:", len(df))
    print("df.columns:", df_init.columns)
    print("df.iloc[0[:", df.iloc[0])
    outfile_df = os.path.join(args.outdir, '00_dataframe.csv')
    df.to_csv(outfile_df)

    # plot
    if args.plot_thresh > 0:
        post_process.refine_and_plot_df(
            df, label_map_dict=label_map_dict,
            outdir=args.outdir,
            sliced=bool(args.slice_val_images),
            plot_thresh=args.plot_thresh,
            nms_overlap_thresh=args.nms_overlap_thresh,
            show_labels=args.make_box_labels,
            alpha_scaling=args.scale_alpha,
            plot_line_thickness=args.plot_line_thickness,
            verbose=bool(args.verbose))

    print("Plots output to:", args.outdir)
    print("Time to get and plot records:", time.time() - t0, "seconds")


###############################################################################
if __name__ == "__main__":
    main()
