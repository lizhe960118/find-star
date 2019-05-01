import os
import pandas as pd
import numpy as np
from PIL import Image
import sys


# classes_list = ['noise','ghost', 'pity', 'newtarget', 'isstar', 'asteroid', 'isnova', 'known']
# count_list = [0 for i in range(8)]

# def convert(size, box):
#     dw = 1./(size[0])
#     dh = 1./(size[1])
#     x = (box[0] + box[1])/2.0 - 1
#     y = (box[2] + box[3])/2.0 - 1
#     w = box[1] - box[0]
#     h = box[3] - box[2]
#     x = x*dw
#     w = w*dw
#     y = y*dh
#     h = h*dh
#     return (x,y,w,h)


def convert_csv2test_txt(test_data_path):
    
    # print('**** CSV data to TXT ****', file=sys.stderr)
    # fin : str = sys.argv[1]             # filepath: input data as csv file
    # print('Input file: %s' % fin, file=sys.stderr)

#     test_data_path = './data/af2019-cv-testA-20190318/test.csv'
    df = pd.read_csv(test_data_path + '/test.csv')
    print('Origin shape: ',df.shape)

    df = df.dropna(axis=0)

    num_of_df = df.shape[0]
    # print('Preprocessing: ', df.shape)

    # image_ids = []

    for index in range(num_of_df):
        image_path = str(df.ix[index,0])

        # prefix_path = image_path[:2]
        # img = Image.open('./af2019-cv-testA-20190318/' + prefix_path + '/' + image_path + '_a.jpg')    
        # img_w = img.size[0]
        # img_h = img.size[1]

        # cx = float(df.ix[index, 1])
        # cy = float(df.ix[index, 2])

        # if cx > img_w or cy > img_h:
        #     continue

        # label = str(df.ix[index, 3])

        # if label not in classes_list:
        #     continue

        # for i in range(len(classes_list)):
        #     if classes_list[i] == label:
        #         label_index = i
        #         count_list[i] += 1

        # if label_index <= 2:
        #     w, h = 5, 5
        #     # box = (cx, cy, w, h, label_index)
        # else:
        #     w, h = 15, 15
        #     # box = (cx, cy, w, h, label_index)

        # xmin = cx - w / 2
        # ymin = cy - h / 2
        # xmax = cx + w / 2
        # ymax = cy + h / 2

        # box = (float(xmin), float(ymin), float(xmax), float(ymax), label_index)
        # bb = convert((w,h), b)

        # cls_id = 0

        # image_id = labels[index,0][:-4]

        with open('./data/find_star_txt_folder/find_star_test_bbx_gt.txt', 'a+') as out_file:
            out_file.write(image_path + '\n')
            # out_file.write(" ".join([str(a) for a in box]) + '\n')

    # for i in range(8):
    #     print("this number of {} is:{}".format(classes_list[i], count_list[i]))

    print('finish convert csv to test_bbx_gt.txt')