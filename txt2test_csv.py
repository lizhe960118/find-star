import csv

def modify_csv(target_csv_name):
    out = open(target_csv_name, 'w')
    out.close()

def convert_txt2csv(list_file_name, target_csv_name):
    with open(list_file_name) as f:
        lines = f.readlines()
        num_images = len(lines)

    modify_csv(target_csv_name) # 清除csv中的内容

    head = ['id', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3' ,'havestar']

    out = open(target_csv_name, 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')

    csv_write.writerow(head)    

    for line in lines:
        splited = line.strip().split()
        csv_write.writerow(splited)

    print("write over!")

def test():
    convert_txt2csv('./data/find_star_txt_folder/test_result.txt', 'submit.csv')

# test()