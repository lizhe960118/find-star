'''Encode object boxes and labels.'''
import math
import torch
import numpy as np

from utils import meshgrid, box_iou, box_nms, change_box_order
from scripts.nms.nms_wrapper import nms


class DataEncoder:
    def __init__(self):
        # self.anchor_areas = [32*32., 64 * 64., 128*128., 256*256., 512*512.]  # p3 -> p7
        self.anchor_areas = [4 * 4, 8 * 8., 16 * 16., 32 * 32.]  # p2 -> p5
#         self.anchor_areas = [2 * 2, 4* 4, 8 * 8., 16 * 16.]  # p2 -> p5
        # self.aspect_ratios = [1/2., 1/1., 2/1.]
        self.aspect_ratios = [1/1.] # 只选用一种形状的框
        self.scale_ratios = [1., pow(2,1/3.), pow(2,2/3.)]
#         self.scale_ratios = [1., pow(2,2/3.), 2]
        self.anchor_wh = self._get_anchor_wh()

    def _get_anchor_wh(self):
        '''Compute anchor width and height for each feature map.

        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        '''
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # w/h = ar
                h = math.sqrt(s/ar)
                w = ar * h
                for sr in self.scale_ratios:  # scale
                    anchor_h = h*sr
                    anchor_w = w*sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_wh).view(num_fms, -1, 2)

    def _get_anchor_boxes(self, input_size):
        '''Compute anchor boxes for each feature map.

        Args:
          input_size: (tensor) model input size of (w,h).

        Returns:
          boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                        where #anchors = fmw * fmh * #anchors_per_cell
        '''
        num_fms = len(self.anchor_areas)
        # fm_sizes = [(input_size/pow(2.,i+3)).ceil() for i in range(num_fms)]  # p3 -> p7 feature map sizes
        fm_sizes = [(input_size/pow(2.,i+1)).ceil() for i in range(num_fms)] # # p2 -> p5 feature map sizes

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fm_w,fm_h) + 0.5  # [fm_h*fm_w, 2]
            # xy = (xy*grid_size).view(fm_h,fm_w,1,2).expand(fm_h,fm_w,9,2)
            # wh = self.anchor_wh[i].view(1,1,9,2).expand(fm_h,fm_w,9,2)
            xy = (xy*grid_size).view(fm_h,fm_w,1,2).expand(fm_h,fm_w,3,2)
            wh = self.anchor_wh[i].view(1,1,3,2).expand(fm_h,fm_w,3,2)
            box = torch.cat([xy,wh], 3)  # [x,y,w,h]
            boxes.append(box.view(-1,4))
        return torch.cat(boxes, 0)

    def encode(self, boxes, labels, input_size):
        '''Encode target bounding boxes and class labels.

        We obey the Faster RCNN box coder:
          tx = (x - anchor_x) / anchor_w
          ty = (y - anchor_y) / anchor_h
          tw = log(w / anchor_w)
          th = log(h / anchor_h)

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        '''
        NEG = 10
        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
                     else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)
        boxes = change_box_order(boxes, 'xyxy2xywh')

        ious = box_iou(anchor_boxes, boxes, order='xywh')
        max_ious, max_ids = ious.max(1)
        
        boxes = boxes[max_ids]

        loc_xy = (boxes[:,:2] - anchor_boxes[:,:2]) / anchor_boxes[:,2:]
        loc_wh = torch.log(boxes[:,2:] / anchor_boxes[:,2:])
        loc_targets = torch.cat([loc_xy,loc_wh], 1)
        
        # 这里，我们设置正负采样比例为 1:3
        cls_targets = 1 + labels[max_ids] # 类别等于label加1, 最开始初始化为正类
#         print(cls_targets)
        
        cls_targets[max_ious < 0.1] = 0
        ignore = (max_ious > 0.05) & (max_ious < 0.1)
        cls_targets[ignore] = -1  # for now just mark ignored to -1
        '''
        cls_targets[max_ious < 0.1] = 0
#         print("cls_targets shape:", cls_targets.shape)
        pos = cls_targets > 0 
        n_pos = pos.data.float().sum().item()
#         print(n_pos)
        n_neg = NEG * n_pos if n_pos != 0 else NEG
        n_neg = int(n_neg)
#         print('n_neg',n_neg)
        
#         print(max_ious.shape)
        max_ious = max_ious.numpy().astype(np.float)
        neg_index = np.where(max_ious < 0.1)[0]
#         print("neg_index shape", neg_index.size)
#         print("neg_index", neg_index)
#         neg_index = neg_index.squeeze(1)
#         neg_index = neg_index.numpy().astype(np.int)
#         print("neg_index numpy shape", neg_index.shape)
        
        if neg_index.shape[0] > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
#             disable_index = disable_index.unsqueeze(1)
#             print("disable_index",disable_index.shape)
            disabel_index = torch.from_numpy(disable_index).float()
            cls_targets[disable_index] = -1
#         print("cls_targets",cls_targets)
#         pos_neg = cls_targets > -1  # exclude ignored anchors
#         print("pos_neg", pos_neg.data.float().sum().item())
# #         ignore = (max_ious > 0.05) & (max_ious<0.01）
# #         cls_targets[ignore] = -1  # for now just mark ignored to -1
        '''
        return loc_targets, cls_targets

    """
    def decode(self, loc_preds, cls_preds, input_size):
        '''Decode outputs back to bouding box locations and class labels.
        Args:
          loc_preds: (tensor) predicted locations, sized [#anchors, 4].
          cls_preds: (tensor) predicted class labels, sized [#anchors, #classes].
          input_size: (int/tuple) model input size of (w,h).
        Returns:
          boxes: (tensor) decode box locations, sized [#obj,4].
          labels: (tensor) class labels for each box, sized [#obj,].
        '''
        CLS_THRESH = 0.015
        NMS_THRESH = 0.5

        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
                     else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size).cuda()

        loc_xy = loc_preds[:,:2]
        loc_wh = loc_preds[:,2:]

        xy = loc_xy * anchor_boxes[:,2:] + anchor_boxes[:,:2]
        wh = loc_wh.exp() * anchor_boxes[:,2:]
        boxes = torch.cat([xy-wh/2, xy+wh/2], 1)  # [#anchors,4]

        score, labels = cls_preds.sigmoid().max(1)          # [#anchors,]
        ids = score > CLS_THRESH
        ids = ids.nonzero().squeeze()             # [#obj,]
        print(ids)
        print(ids.shape)
        keep = box_nms(boxes[ids], score[ids], threshold=NMS_THRESH)
        return boxes[ids][keep], labels[ids][keep]

        """
    def decode(self, loc_preds, cls_preds, input_size):
        '''Decode outputs back to bouding box locations and class labels.

        Args:
          loc_preds: (tensor) predicted locations, sized [#anchors, 4].
          cls_preds: (tensor) predicted class labels, sized [#anchors, #classes].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          boxes: (tensor) decode box locations, sized [#obj,4].
          labels: (tensor) class labels for each box, sized [#obj,].
        '''
#         CLS_THRESH = 0.08
#         NMS_THRESH = 0.5
        NMS_THRESH = 0.2
        N_BBOXES = 200

        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
                     else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size).cuda()
#         anchor_boxes = self._get_anchor_boxes(input_size)

        loc_xy = loc_preds[:,:2]
        loc_wh = loc_preds[:,2:]

        xy = loc_xy * anchor_boxes[:,2:] + anchor_boxes[:,:2]
        wh = loc_wh.exp() * anchor_boxes[:,2:]
        boxes = torch.cat([xy-wh/2, xy+wh/2], 1)  # [#anchors,4] (x1, y1, x2, y2)

        score, labels = cls_preds.sigmoid().max(1)          # [#anchors,]
#         ids = score > CLS_THRESH
#         ids = ids.nonzero().squeeze()             # [#obj,]
        
        numpy_score = score.cpu().numpy().astype(np.float) # 如果取前200个最得分最大的框的话
#         numpy_score = score.numpy().astype(np.float)
        rank_ids = np.argsort(numpy_score)[::-1]
#         print(rank_ids)

        if len(rank_ids) > N_BBOXES:
            choose_ids = rank_ids[:N_BBOXES].astype(np.int)
            choose_ids = torch.from_numpy(choose_ids).cuda()
#             choose_ids = torch.from_numpy(choose_ids)
            ids = choose_ids
#         print(ids)
#         print(ids.shape)
#         print(boxes[ids])
#         print(score[ids])
#         keep = nms(torch.cat((boxes[ids].cuda(), score[ids].view(-1, 1).cuda()), 1), NMS_THRESH)
#         keep = keep.long().squeeze(1)
#         print(keep.size())
        
        keep = box_nms(boxes[ids], score[ids], threshold=NMS_THRESH)
    
        return boxes[ids][keep], labels[ids][keep]