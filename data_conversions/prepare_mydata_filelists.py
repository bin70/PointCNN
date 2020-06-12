#!/usr/bin/python3
'''Prepare Filelists for S3DIS Segmentation Task.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', help='Path to data folder')
    parser.add_argument('--h5_num', '-d', help='Number of h5 files to be loaded each time', type=int, default=8)
    parser.add_argument('--repeat_num', '-r', help='Number of repeatly using each loaded h5 list', type=int, default=2)

    args = parser.parse_args()
    print(args)

    root = args.folder if args.folder else '../data/mydata'
    filename_h5s = ['./%s\n' % (filename) for filename in os.listdir(root) 
                                            if filename.endswith('.h5')]
    filelist_txt = os.path.join(root, 'my_test_data.txt')
    print('{}-Saving {}...'.format(datetime.now(), filelist_txt))
    with open(filelist_txt, 'w') as filelist:
        for filename_h5 in filename_h5s:
            filelist.write(filename_h5)

if __name__ == '__main__':
    main()
