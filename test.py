import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw

import pandas as pd
import txt2test_csv
import argparse

def modify_text(txt_name):
    with open(txt_name, "r+") as f:
        f.seek(0)
        f.truncate()   #清空文件
        
def de_resize_boxes(boxes, w, h, old_w, old_h):
    sw = float(old_w) / w
    sh = float(old_h) / h
    return boxes * torch.Tensor([sw, sh, sw, sh]).cuda()
#     return boxes * torch.Tensor([sw, sh, sw, sh])
        
parser = argparse.ArgumentParser(description='PyTorch RetinaNet Testing')
# parser.add_argument('--test_dataset', default= './data/af2019-cv-testA-20190318',help="the path of test data")
parser.add_argument('--test_dataset', default= './data/af2019-cv-testB-20190408',help="the path of test data")
parser.add_argument('--model', default = 'checkpoint', help='model_dir')
parser.add_argument('--prediction_file', default= 'submit.csv',help="the path to save result")
args = parser.parse_args()

with torch.no_grad(): 
    print('Loading model..')
    net = RetinaNet()

    load_model_path = args.model # 这里需要手动设定
    load_model_epoch = args.load_model_epoch
    checkpoint = torch.load('{}/ckpt.pth'.format(load_model_path))
    # net.load_state_dict(torch.load('./checkpoint/params.pth'))
    
    net.load_state_dict(checkpoint['net'])
    
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.cuda()
    net.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])

    print('Loading image..')
    test_data_path = args.test_dataset
#     df = pd.read_csv(test_data_path + '/test.csv')
    df = pd.read_csv(test_data_path + '/list.csv')
    print('Origin shape: ',df.shape)

    df = df.dropna(axis=0)
    num_of_test_image = df.shape[0]

    result_file_name = './data/find_star_txt_folder/test_result.txt'
    modify_text(result_file_name)

    for index in range(num_of_test_image):
        fname = str(df.ix[index,0])
        prefix_name = fname[:2]
        image_path = test_data_path + '/' + prefix_name + '/' + fname

        img_a = Image.open(image_path + '_a.jpg')
        img_b = Image.open(image_path + '_b.jpg')
        img_c = Image.open(image_path + '_c.jpg')

        img = Image.merge('RGB', (img_a, img_b, img_c))
        old_w, old_h = img.size
    #     img = Image.open('./image/000001.jpg')
        w = h = 600
        img = img.resize((w,h))

        print('Predicting..{}'.format(fname))

        x = transform(img)
        x = x.unsqueeze(0)
        x = Variable(x.cuda())
    #     x = Variable(x)
        loc_preds, cls_preds = net(x)

        print('Decoding..')
        encoder = DataEncoder()
        boxes, labels = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w,h))

        boxes = de_resize_boxes(boxes, w, h, old_w, old_h)

        n_boxes = boxes.shape[0]
        if n_boxes == 1:
            box_1 = boxes[0]
            boxes = []
            for i in range(3):
                boxes.append(box_1)
            boxes = torch.stack(boxes)
        elif n_boxes == 2:
            box_1 = boxes[0]
            box_2 = boxes[1]
            boxes = []
            boxes.append(box_1)
            boxes.append(box_1)
            boxes.append(box_2)
            boxes = torch.stack(boxes)        
        else: 
            boxes = boxes[:3,:] # box(x1, x2, y1, y2)

    #     print(boxes)
        cxcy = (boxes[:,:2] + boxes[:,2:]) / 2
    #     print(cxcy.shape)
        str_cxcy = []
        for i in range(3):
            cx,cy = cxcy[i].cpu().numpy().astype(int).tolist()
    #         cx,cy = cxcy[i].numpy().astype(int).tolist()
            str_cxcy.append(str(cx))
            str_cxcy.append(str(cy))

        most_important_label = labels[0].cpu().data.item()
    #     most_important_label = labels[0].data.item()

        haveStar = 0
        if most_important_label >= 3:
            haveStar = 1

        with open(result_file_name, 'a+') as out_file:
            out_file.write(fname + " ")
            out_file.write(" ".join([item for item in str_cxcy]) + ' ')
            out_file.write(str(haveStar) + '\n')

# write to result.csv
target_csv_name = args.prediction_file
txt2test_csv.convert_txt2csv(result_file_name, target_csv_name)
    
# draw = ImageDraw.Draw(img)
# for box in boxes:
#     draw.rectangle(list(box), outline='red')
# img.show()
