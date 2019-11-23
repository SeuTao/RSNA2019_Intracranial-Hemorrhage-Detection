import numpy as np # linear algebra
import pandas as pd
import os
import cv2
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from PIL import Image
import math
import seaborn as sns
from collections import defaultdict
from pathlib import Path
import cv2
from tqdm import tqdm
import re
import sys
import pydicom
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-root_path", "--root_path", type=str)
    parser.add_argument("-train_dcm_path", "--train_dcm_path", type=str)
    parser.add_argument("-test_dcm_path", "--test_dcm_path", type=str)
    parser.add_argument("-save_path", "--save_path", type=str)


    args = parser.parse_args()
    ROOT_DIR = args.root_path
    train_dcm_path = args.train_dcm_path
    test_dcm_path = args.test_dcm_path
    save_path = args.save_path

    sub_df = pd.read_csv(ROOT_DIR + 'stage_2_sample_submission.csv')

    sub_df['filename'] = sub_df['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".png")
    sub_df['type'] = sub_df['ID'].apply(lambda st: st.split('_')[2])

    test_df = pd.DataFrame(sub_df.filename.unique(), columns=['filename'])
    patient_id = []
    study_instance_uid = []
    series_instance_uid = []
    image_position = []
    samples_per_pixel = []
    pixel_spacing = []
    pixel_representation = []
    window_center = []
    window_width = []
    rescale_intercept = []
    rescale_slope = []

    for i in tqdm(range(test_df.shape[0])):
        dcm_path = test_dcm_path + test_df['filename'][i].replace('png', 'dcm')

        data = pydicom.dcmread(dcm_path)
        patient_id.append(data[('0010', '0020')].value)
        study_instance_uid.append(data[('0020', '000d')].value)
        series_instance_uid.append(data[('0020', '000e')].value)
        image_position.append(data[('0020', '0032')].value)
        samples_per_pixel.append(data[('0028', '0002')].value)
        pixel_spacing.append(data[('0028', '0030')].value)
        pixel_representation.append(data[('0028', '0103')].value)
        window_center.append(data[('0028', '1050')].value)
        window_width.append(data[('0028', '1051')].value)
        rescale_intercept.append(data[('0028', '1052')].value)
        rescale_slope.append(data[('0028', '1053')].value)

    test_df['patient_id'] = patient_id
    test_df['study_instance_uid'] = study_instance_uid
    test_df['series_instance_uid'] = series_instance_uid
    test_df['image_position'] = image_position
    test_df['samples_per_pixel'] = samples_per_pixel
    test_df['pixel_spacing'] = pixel_spacing
    test_df['pixel_representation'] = pixel_representation
    test_df['window_center'] = window_center
    test_df['window_width'] = window_width
    test_df['rescale_intercept'] = rescale_intercept
    test_df['rescale_slope'] = rescale_slope

    test_df.to_csv(save_path + 'stage2_test_cls.csv')