import cv2
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision.transforms import transforms
import pyzbar.pyzbar as pyzbar
from nets.resnet_yolo import resnet18
from nets.squeezenet_yolo import squeezenet

# Constants
VOC_CLASSES = ('qr_code',)
COLOR = [[0, 0, 0]]  # Black for QR codes

# Helper function to perform non-maximum suppression
def nms(bboxes, scores, threshold=0.1):
    """
    Perform non-maximum suppression to avoid overlapping bounding boxes.
    """
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0] if order.numel() > 1 else order.item()
        keep.append(i)

        if order.numel() == 1:
            break

        inter = torch.zeros(order.numel() - 1)
        for idx in range(1, order.numel()):
            xx1 = max(x1[i], x1[order[idx]])
            yy1 = max(y1[i], y1[order[idx]])
            xx2 = min(x2[i], x2[order[idx]])
            yy2 = min(y2[i], y2[order[idx]])

            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            inter[idx-1] = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= threshold).nonzero().squeeze()
        order = order[ids + 1]

    return torch.LongTensor(keep)

# Decoder function to interpret YOLO outputs
def decoder(pred):
    """
    Decode the predictions of the network.
    """
    grid_num = 8
    boxes = []
    # cls_indexs = []
    probs = []
    cell_size = 1.0 / grid_num
    pred = pred.squeeze(0)

    contain1 = pred[:, :, 4].unsqueeze(2)
    contain2 = pred[:, :, 9].unsqueeze(2)
    contain = torch.cat((contain1, contain2), 2)
    mask = (contain > 0.1) | (contain == contain.max())

    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                if mask[i, j, b]:
                    box = pred[i, j, b*5:b*5+4]
                    contain_prob = torch.FloatTensor([pred[i, j, b*5+4]])
                    xy = torch.FloatTensor([j, i]) * cell_size
                    box[:2] = box[:2] * cell_size + xy
                    box_xy = torch.FloatTensor(box.size())
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]
                    if float(contain_prob[0]) > 0.1:
                        boxes.append(box_xy.view(1, 4))
                        probs.append(contain_prob)

    if not boxes:
        return torch.zeros(1, 4), torch.zeros(1)

    boxes = torch.cat(boxes)
    probs = torch.cat(probs)
    keep = nms(boxes, probs)
    return boxes[keep], probs[keep]

def predict_gpu(model, image_name, root_path=''):
    """
    Predict function to process a single image with QR code decoding.
    """
    result = []
    image = cv2.imread(root_path + image_name)
    h, w, _ = image.shape
    img = cv2.cvtColor(cv2.resize(image, (64, 64)), cv2.COLOR_BGR2RGB)
    img = (img - np.array([128, 128, 128], dtype=np.float32)) / 2

    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    with torch.no_grad():
        img = img[None, :, :, :]
        img = img.cuda()

        pred = model(img)
        pred = pred.cpu()
        boxes, probs = decoder(pred)
        for i, box in enumerate(boxes):
            x1 = int(box[0] * w)
            x2 = int(box[2] * w)
            y1 = int(box[1] * h)
            y2 = int(box[3] * h)

            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            prob = float(probs[i])

            # Draw bounding box regardless of decoding success
            color = COLOR[0]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Crop the QR code region
            pad_height = int(0.001 * (y2 - y1))
            pad_width = int(0.001 * (x2 - x1))
            cropped_img = image[max(0, y1 - pad_height):min(h, y2 + pad_height), max(0, x1 - pad_width):min(w, x2 + pad_width)]

            # Check if the cropped image is valid
            if cropped_img.size > 0 and cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
                gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

                # Decode the grayscale image
                barcodes = pyzbar.decode(gray_img)
                if barcodes:
                    barcode_data = barcodes[0].data.decode("utf-8")
                    label = f"QR Code: {barcode_data} ({round(prob, 2)})"
                    print(f"QR Code decoded successfully: {barcode_data}")
                else:
                    label = "QR Code: Decoding failed"
                    print("QR Code decoding failed")
            else:
                label = "Invalid cropped QR image, skipping decoding"
                print("Invalid cropped QR image, skipping decoding")

            # Draw label
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            p1 = (x1, y1 - text_size[1])
            cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline),
                          (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
            cv2.putText(image, label, (p1[0], p1[1] + baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

    cv2.imwrite('result.jpg', image)


if __name__ == '__main__':
    model = resnet18().cuda()
    model.load_state_dict(torch.load('./WEIGHT/yolo-float-separable.pth'))
    model.eval()

    image_name = '0000964.jpg'
    results = predict_gpu(model, image_name)
    image = cv2.imread(image_name)