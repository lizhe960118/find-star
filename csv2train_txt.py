import os
import pandas as pd
import numpy as np
from PIL import Image
import sys
import random

classes_list = ['noise','ghost', 'pity', 'newtarget', 'isstar', 'asteroid', 'isnova', 'known']
# repetition_rate = [0.5, 10, 0.5, 10, 2, 5, 10, 5]
# repetition_rate = [0.5, 20, 0.5, 40, 2, 5, 40, 5]
repetition_rate = [1, 1, 1, 1, 1, 1, 1, 1]
# repetition_rate = [0.5, 39, 0.5, 234, 2, 10, 280, 8]

def modify_text(txt_name):
    with open(txt_name, "r+") as f:
        f.seek(0)
        f.truncate()   #清空文件

def convert_csv2train_txt(train_data_path):    
    # print('**** CSV data to TXT ****', file=sys.stderr)
    # fin : str = sys.argv[1]             # filepath: input data as csv file
    # print('Input file: %s' % fin, file=sys.stderr)
    count_list = [0 for i in range(8)]
    
    df = pd.read_csv(train_data_path + '/list.csv')
    print('Origin shape: ',df.shape)

    df = df.dropna(axis=0)

    num_of_df = df.shape[0]
    print('Preprocessing: ', df.shape)

    image_loc_labels = []

    for index in range(num_of_df):
        image_path = str(df.ix[index,0])

        prefix_path = image_path[:2]
        img = Image.open(train_data_path + '/' + prefix_path + '/' + image_path + '_a.jpg')    
        img_w = img.size[0]
        img_h = img.size[1]

        cx = float(df.ix[index, 1])
        cy = float(df.ix[index, 2])

        # 统一设置
        w, h = 15, 15

        if cx > img_w or cy > img_h:
            continue

        label = str(df.ix[index, 3])

        if label not in classes_list:
            continue

        for i in range(len(classes_list)):
            if classes_list[i] == label:
                label_index = i
        
        xmin = cx - w / 2
        ymin = cy - h / 2
        xmax = cx + w / 2
        ymax = cy + h / 2

        # box = (float(xmin), float(ymin), float(xmax), float(ymax), label_index)

        image_loc_label = (image_path, float(xmin), float(ymin), float(xmax), float(ymax), label_index)
#         image_loc_labels.append(image_loc_label)
#         count_list[label_index] += 1

        rept_num = repetition_rate[label_index]
        
        if rept_num >= 1:            
            for i in range(rept_num):
                image_loc_labels.append(image_loc_label)
                count_list[label_index] += 1
        else:
            if np.random.rand() > 0.5:
                image_loc_labels.append(image_loc_label)
                count_list[label_index] += 1

    num_data = len(image_loc_labels)
    
    # clean the file
    txt_name = './data/find_star_txt_folder/find_star_train_bbx_gt.txt'
    modify_text(txt_name)
#     test_txt_name = './data/find_star_txt_folder/find_star_test_bbx_gt.txt'
#     modify_text(test_txt_name)

    # shuffle the input data
    index_shuffle = [i for i in range(num_data)]
    random.shuffle(index_shuffle)

    for i in range(num_data):
        image_loc_label = image_loc_labels[index_shuffle[i]]
#         if i < 5000:
        with open(txt_name, 'a+') as out_file:   
            # 覆盖写入
            out_file.write(" ".join([str(item) for item in image_loc_label]) + '\n')
#         else:
#             with open(test_txt_name, 'a+') as out_file:
#                 out_file.write(" ".join([str(item) for item in image_loc_label]) + '\n')

    for i in range(8):
        print("this number of {} is:{}".format(classes_list[i], count_list[i]))

    print('finish convert csv to train_bbx_gt.txt')

def test():
    convert_csv2train_txt('./data/af2019-cv-training-20190312')

# test()