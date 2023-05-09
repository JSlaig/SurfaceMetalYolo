from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import time
import datetime
import argparse
import tqdm

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_final.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)

    print("Compute mAP...")

    precision_list = []
    recall_list = []
    AP_list = []
    f1_list = []
      
    for i in range(16):
        # Load the checkpoint
        checkpoint_path = f"checkpoints/yolov3_ckpt_{i+1}.pth"
        model.load_state_dict(torch.load(checkpoint_path))
    
        # Evaluate the model and get the recall for each class
        precision, recall, AP, f1, ap_class = evaluate(
            model,
            path=valid_path,
            iou_thres=opt.iou_thres,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            img_size=opt.img_size,
            batch_size=8,
        )
        
        precision_list.append(precision)
        recall_list.append(recall)
        AP_list.append(AP)
        f1_list.append(f1)
    
    print(precision, recall, AP, f1, ap_class)

    c1_precision_list = []
    c1_recall_list = []
    c1_AP_list = []
    c1_f1_list = []
    
    c2_precision_list = []
    c2_recall_list = []
    c2_AP_list = []
    c2_f1_list = []
    
    c3_precision_list = []
    c3_recall_list = []
    c3_AP_list = []
    c3_f1_list = []
    
    c4_precision_list = []
    c4_recall_list = []
    c4_AP_list = []
    c4_f1_list = []
    
    c5_precision_list = []
    c5_recall_list = []
    c5_AP_list = []
    c5_f1_list = []
    
    c6_precision_list = []
    c6_recall_list = []
    c6_AP_list = []
    c6_f1_list = []
    
    for j in precision_list:
        c1_precision_list.append(precision_list[j][0])
        c1_recall_list.append(recall_list[j][0])
        c1_AP_list.append(AP_list[j][0])
        c1_f1_list.append(f1_list[j][0])   
        
        c2_precision_list.append(precision_list[j][1])
        c2_recall_list.append(recall_list[j][1])
        c2_AP_list.append(AP_list[j][1])
        c2_f1_list.append(f1_list[j][1])
        
        c3_precision_list.append(precision_list[j][2])
        c3_recall_list.append(recall_list[j][2])
        c3_AP_list.append(AP_list[j][2])
        c3_f1_list.append(f1_list[j][2])
        
        c4_precision_list.append(precision_list[j][3])
        c4_recall_list.append(recall_list[j][3])
        c4_AP_list.append(AP_list[j][3])
        c4_f1_list.append(f1_list[j][3])
        
        c5_precision_list.append(precision_list[j][4])
        c5_recall_list.append(recall_list[j][4])
        c5_AP_list.append(AP_list[j][4])
        c5_f1_list.append(f1_list[j][4])
        
        c6_precision_list.append(precision_list[j][5])
        c6_recall_list.append(recall_list[j][5])
        c6_AP_list.append(AP_list[j][5])
        c6_f1_list.append(f1_list[j][5])
        
    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        
        if c == 0:
            precision = c1_precision_list
            recall = c1_recall_list
        elif c == 1:
            precision = c2_precision_list
            recall = c2_recall_list
        elif c == 2:
            precision = c3_precision_list
            recall = c3_recall_list
        elif c == 1:
            precision = c3_precision_list
            recall = c3_recall_list
        elif c == 1:
            precision = c4_precision_list
            recall = c4_recall_list
        elif c == 1:
            precision = c5_precision_list
            recall = c5_recall_list 
        
            

            # Plot precision-recall curve
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.set_title(f'Precision-Recall Curve for {class_names[c]}')
            plt.show()
    print(f"mAP: {AP.mean()}")
