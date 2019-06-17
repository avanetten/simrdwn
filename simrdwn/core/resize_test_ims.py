#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 15:42:18 2018

@author: avanetten

Resize test images, may be necessaary if training resolution differs
from native test resolution

"""

import os
import cv2
import skimage.io
import argparse


###############################################################################
def resize_dir(input_dir, output_dir, resize_factor=2, compression_level=3):
    '''Resize images in input dir
    compression goes from 0-9 (9 being most compressed), default is 3'''

    im_list = [z for z in os.listdir(input_dir) if z.endswith('.tif')]
    for i, im_name in enumerate(im_list):

        print(i, "/", len(im_list), ":", im_name)

        im_path = os.path.join(input_dir, im_name)
        out_path = os.path.join(output_dir, im_name)

        # load in file, cv2 can't load very large files
        use_skimage = False
        try:
            im = cv2.imread(im_path)
        except:
            # load with skimage, (reversed order of bands)
            im = skimage.io.imread(im_path)    # [::-1]
            use_skimage = True

        height, width = im.shape[:2]
        im_out = cv2.resize(im, (int(width/resize_factor),
                            int(height/resize_factor)),
                            interpolation=cv2.INTER_CUBIC)
        print("  input shape:", im.shape)
        print("  output shape:", im_out.shape)
        # write file
        if not use_skimage:
            cv2.imwrite(out_path, im_out,
                        [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
        else:
            # if we read in with skimage, need to reverse colors
            cv2.imwrite(out_path, cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
    return


###############################################################################
def main():

    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='',
                        help="images location")
    parser.add_argument('--output_dir', type=str, default='',
                        help="output_images location")
    parser.add_argument('--resize_factor', type=int, default=2,
                        help="inverse of fraction to resize the image ")
    parser.add_argument('--compression_level', type=int, default=3,
                        help="compression level of image"
                        + " (0-9, 9 being most compressed and 3 is default")
    # parser.add_argument('--n_bands', type=int, default=3,
    #                     help="number of image bands")
    args = parser.parse_args()

    # make output dir
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # execute
    resize_dir(args.input_dir, args.output_dir,
               resize_factor=args.resize_factor,
               compression_level=args.compression_level)


###############################################################################
if __name__ == "__main__":
    main()
