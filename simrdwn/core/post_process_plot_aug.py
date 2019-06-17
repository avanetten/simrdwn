#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 21:00:35 2019

@author: ave

Plot the augmented dataframe
"""

import pandas as pd
import numpy as np
import argparse
import os
from . import preprocess_tfrecords
from . import post_process
from . import add_geo_coords


###############################################################################
def plot_df_aug(df_aug_loc, label_map_path, results_dir,
                plot_threshes=[0.2],
                groupby='Image_Path',
                groupby_cat='Category',
                cats_to_ignore=[],
                image_roots_to_keep=[],
                test_add_geo_coords=True,
                save_json=True,
                nms_overlap_thresh=0.5,
                n_test_output_plots=4,
                plot_line_thickness=2,
                show_labels=True,
                alpha_scaling=True,
                suffix='',
                verbose=True
                ):
    '''Plot augmented dataframe output of SIMRDWN'''

    # make label_map_dic
    if len(label_map_path) > 0:
        label_map_dict = preprocess_tfrecords.load_pbtxt(
            label_map_path, verbose=False)
    else:
        label_map_dict = {}
    print("label_map_dict:", label_map_dict)

    df_tot = pd.read_csv(df_aug_loc)

    # keep only ceratain image_roots
    if len(image_roots_to_keep) > 0:
        df_tot = df_tot.loc[df_tot['Image_Root'].isin(image_roots_to_keep)]

    # refine for each plot_thresh
    for plot_thresh_tmp in plot_threshes:
        print("Plotting at:", plot_thresh_tmp)
        df_refine = post_process.refine_df(df_tot,
                                           groupby=groupby,
                                           groupby_cat=groupby_cat,
                                           cats_to_ignore=cats_to_ignore,
                                           nms_overlap_thresh=nms_overlap_thresh,
                                           plot_thresh=plot_thresh_tmp,
                                           verbose=verbose)

        post_process.plot_refined_df(df_refine, groupby=groupby,
                                     label_map_dict=label_map_dict,
                                     outdir=results_dir,
                                     plot_thresh=plot_thresh_tmp,
                                     show_labels=show_labels,
                                     alpha_scaling=alpha_scaling,
                                     plot_line_thickness=plot_line_thickness,
                                     print_iter=5,
                                     n_plots=n_test_output_plots,
                                     building_csv_file='',
                                     shuffle_ims=False,
                                     verbose=verbose)

        # geo coords?
        if test_add_geo_coords:
            df_refine, json = add_geo_coords.add_geo_coords_to_df(
                df_refine,
                create_geojson=save_json,
                inProj_str='epsg:4326', outProj_str='epsg:3857',
                # inProj_str='epsg:32737', outProj_str='epsg:3857',
                verbose=False)

        # save df_refine
        outpath_tmp = os.path.join(
            results_dir,
            'test_predictions_refine' +
            '_thresh=' + str(plot_thresh_tmp) + suffix + '.csv')
        df_refine.to_csv(outpath_tmp)
        print("Total num objects at thresh:", plot_thresh_tmp, "=",
              len(df_refine))
        # save json
        if save_json:
            output_json_path = os.path.join(
                results_dir,
                'test_predictions_refine' +
                '_thresh=' + str(plot_thresh_tmp) + suffix + '.GeoJSON')
            json.to_file(output_json_path, driver="GeoJSON")

    return

###############################################################################


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--results_dir', type=str, default='',
                        help="Location to save images")
    parser.add_argument('--label_map_path', type=str, default='',
                        help="Location of class label file")
    parser.add_argument('--plot_thresh_str', type=str, default='0.3',
                        help="Proposed thresholds to try for test, will be split"
                        + " into array by commas (e.g.: '0.2,0.3' => [0.2,0.3])")
    parser.add_argument('--cats_to_ignore_str', type=str, default='',
                        help="Categories to ignore")
    parser.add_argument('--image_roots_to_keep_str', type=str, default='',
                        help="Image Roots to keep")
    parser.add_argument('--show_labels', type=int, default=0,
                        help="Switch to use show object labels")
    parser.add_argument('--alpha_scaling', type=int, default=0,
                        help="Switch to scale box alpha with probability")
    parser.add_argument('--suffix', type=str, default='',
                        help="Suffix for output files")

    parser.add_argument('--df_aug_file', type=str, 
                        default='test_predictions_aug.csv',
                        help="Name of augmented dataframe")
    parser.add_argument('--groupby', type=str, default='Image_Path',
                        help="Grouping per image")
    parser.add_argument('--groupby_cat', type=str, default='Category',
                        help="Grouping per category")
    parser.add_argument('--n_test_output_plots', type=int, default=10,
                        help="Switch to save test pngs")
    parser.add_argument('--nms_overlap_thresh', type=float, default=0.5,
                        help="Overlap threshold for non-max-suppresion in python"
                        + " (set to <0 to turn off)")
    parser.add_argument('--plot_line_thickness', type=int, default=2,
                        help="Thickness for test output bounding box lines")
    parser.add_argument('--test_add_geo_coords', type=int, default=1,
                        help="switch to add geo coords to test outputs")
    parser.add_argument('--save_json', type=int, default=1,
                        help="Switch to save a json in test")
    parser.add_argument('--str_delim', type=str, default=',',
                        help="Delimiter for string lists")
    # parser.add_argument('--show_test_plots', type=int, default=0,
    #                    help="Switch to show plots in real time in test")

    args = parser.parse_args()

    args.plot_threshes = np.array(
        args.plot_thresh_str.split(args.str_delim)).astype(float)
    if len(args.cats_to_ignore_str) > 0:
        args.cats_to_ignore = np.array(
            args.cats_to_ignore_str.split(args.str_delim)).astype(str)
    else:
        args.cats_to_ignore = []
    if len(args.image_roots_to_keep_str) > 0:
        args.image_roots_to_keep = np.array(
            args.image_roots_to_keep_str.split(args.str_delim)).astype(str)
    else:
        args.image_roots_to_keep = []
    args.df_aug_loc = os.path.join(args.results_dir, args.df_aug_file)

    plot_df_aug(args.df_aug_loc, args.label_map_path, args.results_dir,
                plot_threshes=args.plot_threshes,
                groupby=args.groupby,
                groupby_cat=args.groupby_cat,
                cats_to_ignore=args.cats_to_ignore,
                image_roots_to_keep=args.image_roots_to_keep,
                test_add_geo_coords=bool(args.test_add_geo_coords),
                save_json=bool(args.save_json),
                nms_overlap_thresh=args.nms_overlap_thresh,
                n_test_output_plots=args.n_test_output_plots,
                plot_line_thickness=args.plot_line_thickness,
                show_labels=bool(args.show_labels),
                alpha_scaling=bool(args.alpha_scaling),
                suffix=args.suffix,
                verbose=True)


###############################################################################
if __name__ == "__main__":

    print("Plotting SIMRDWN boxes...")
    main()
