import os
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from detect import predict_gpu
from nets.resnet_yolo import resnet18
from nets.squeezenet_yolo import squeezenet
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_ap(rec, prec):
    """Calculate the average precision for QR code detection."""
    # Concatenate for interpolation
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
   
    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
   
    # Integrate the area under the curve to get AP
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate_detections(predictions, ground_truths, threshold=0.2):
    """Evaluate detection performance and compute mAP."""
    aps = []
    pred_boxes = predictions['qr_code']
    image_ids = [x[0] for x in pred_boxes]
    confidences = np.array([x[1] for x in pred_boxes])
    BB = np.array([x[2:] for x in pred_boxes])
   
    # Sort by confidence
    sorted_indices = np.argsort(-confidences).tolist()  # Ensure it's a list for indexing
    BB = BB[sorted_indices, :]
    image_ids = [image_ids[x] for x in sorted_indices]  # Corrected
   
    tp = np.zeros(len(image_ids))
    fp = np.zeros(len(image_ids))
    npos = sum(len(boxes) for key, boxes in ground_truths.items() if key[1] == 'qr_code')
   
    for d, image_id in enumerate(image_ids):
        pred_box = BB[d]
        if (image_id, 'qr_code') in ground_truths:
            gt_boxes = ground_truths[(image_id, 'qr_code')]
            for gt_box in gt_boxes:
                overlap = compute_overlap(pred_box, gt_box)
                if overlap > threshold:
                    tp[d] = 1
                    gt_boxes.remove(gt_box)
                    if not gt_boxes:
                        del ground_truths[(image_id, 'qr_code')]
                    break
            fp[d] = 1 - tp[d]
        else:
            fp[d] = 1
   
    # Compute precision, recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = calculate_ap(rec, prec)
    print(f'---class qr_code AP {ap}---')

def compute_overlap(boxA, boxB):
    """Compute the overlap between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
   
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def load_and_prepare_data(label_file):
    """Load image paths and ground truth boxes from the label file."""
    ground_truths = defaultdict(list)
    with open(label_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        image_id = parts[0]
        boxes = [list(map(int, parts[i:i+4])) for i in range(1, len(parts), 5)]
        for box in boxes:
            ground_truths[(image_id, 'qr_code')].append(box)
    return ground_truths

if __name__ == '__main__':
    label_file = 'test_labels-QRcode.txt'
    model_path = './WEIGHT/yolo-float-separable.pth'
    image_folder = '/home/mywang53/QRimages/'

    ground_truths = load_and_prepare_data(label_file)
    model = resnet18().to(device)

    # import linger
    # # 定义dummy输入数据
    # dummy_input=torch.randn(1,3,64,64,requires_grad=True).cuda()
    # # 定义replace_tuple，表示需要替换的模块类型，用于模型量化
    # replace_tuple =(nn.Conv2d,nn.Linear,nn.BatchNorm2d,nn.AvgPool2d)
    # # 使用linger.trace_layers获取模型的BN信息，并进行BN融合
    # linger.trace_layers(model,model,dummy_input,fuse_bn=True)
    # # 使用linger.init进行模型量化
    # model=linger.init(model,quant_modules=replace_tuple,mode=linger.QuantMode.QValue)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    predictions = defaultdict(list)
    for image_id in tqdm(list(set(key[0] for key in ground_truths.keys()))):
        image_path = os.path.join(image_folder, image_id)
        detected_boxes = predict_gpu(model, image_id, image_folder)
        for box in detected_boxes:
            predictions[box[2]].append([box[3], box[4], box[0][0], box[0][1], box[1][0], box[1][1]])
   
    evaluate_detections(predictions, ground_truths)
