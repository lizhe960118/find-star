import os
import pandas as pd
import numpy as np
from PIL import Image
import sys
import random

classes_list = ['noise','ghost', 'pity', 'newtarget', 'isstar', 'asteroid', 'isnova', 'known']
# repetition_rate = [0.5, 20, 0.5, 40, 2, 5, 40, 5]
# repetition_rate = [0.5, 20, 0.5, 10, 2, 5, 10, 5]


def modify_text(txt_name):
    with open(txt_name, "r+") as f:
        f.seek(0)
        f.truncate()   #清空文件

def choose(train_data_path):    
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
        image_loc_labels.append(image_loc_label)

#         rept_num = repetition_rate[label_index]
        
#         if rept_num > 1:            
#             for i in range(rept_num):
#                 image_loc_labels.append(image_loc_label)
                
#         else:
#             if np.random.rand() > 0.8:
#                 image_loc_labels.append(image_loc_label)

    num_data = len(image_loc_labels)
    
    # clean the file
    txt_name = './data/find_star_txt_folder/find_star_test_bbx_gt.txt'
    modify_text(txt_name)

    index_shuffle = [i for i in range(num_data)]
    random.shuffle(index_shuffle)

    for i in range(num_data):
        image_loc_label = image_loc_labels[index_shuffle[i]]
        if np.random.rand() < 0.1:
            count_list[image_loc_label[-1]] += 1
            with open(txt_name, 'a+') as out_file:
                out_file.write(" ".join([str(item) for item in image_loc_label]) + '\n')

    for i in range(8):
        print("this number of {} is:{}".format(classes_list[i], count_list[i]))

    print('finish choose data to test_bbx_gt.txt')

def test():
    choose('./data/af2019-cv-training-20190312')

# if __name__ == "__main__":
#     test()