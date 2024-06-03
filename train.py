import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models  
from torch.autograd import Variable

from yoloLoss import yoloLoss
from dataset import yoloDataset
from nets.resnet_yolo import resnet18
from nets.squeezenet_yolo import squeezenet

import linger
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


file_root = '/home/mywang53/QRimages/'
learning_rate = 0.001
batch_size = 4
use_resnet = True
best_test_loss = np.inf
 

logfile = open('log.txt', 'w')

def train(net, train_loader, optimizer, criterion, mode="float", load_model_path=None, num_epochs=None):
    global best_test_loss
    dummy_input = torch.randn((1, 3, 64, 64)).to(device)
    if mode == "float":
        print("Original float train...")
    elif mode == "clamp" or mode == "quant":
        print(f"{mode} train...")              
        linger.trace_layers(net, net, dummy_input, fuse_bn=True)
        type_modules = (nn.Conv2d)
        normalize_modules = (nn.Conv2d, nn.Linear)
        linger.normalize_module(net, type_modules=type_modules, normalize_weight_value=8, normalize_bias_value=8, normalize_output_value=8)
        net = linger.normalize_layers(net, normalize_modules=normalize_modules, normalize_weight_value=8, normalize_bias_value=8, normalize_output_value=8)        
        if mode == "quant":
            quant_modules = (nn.Conv2d, nn.Linear)
            net = linger.init(net, quant_modules=quant_modules)

    if load_model_path is not None:
        net.load_state_dict(torch.load(load_model_path), strict=True)
   
    for epoch in range(num_epochs):
        net.train()
        total_loss = 0.
        adjust_learning_rate(optimizer, epoch)
        print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
       
        for i, (images, target) in enumerate(train_loader):
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad()
            pred = net(images)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i+1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Iter [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}, average_loss: {total_loss/(i+1):.4f}')
       
        # Save model after every epoch
        torch.save(net.state_dict(), f'./WEIGHT/yolo-{mode}-separable.pth')

        net.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for images, target in test_loader:
                images, target = images.to(device), target.to(device)
                pred = net(images)
                loss = criterion(pred, target)
                validation_loss += loss.item()
        validation_loss /= len(test_loader)
        print(f'Epoch {epoch+1}, Validation Loss: {validation_loss:.5f}')
       


    net.eval()    
    with torch.no_grad():
        save_path = "tmp.ignore/YOLO." + mode + ".onnx"
        torch.onnx.export(net,
                          dummy_input.to(device),
                          save_path,
                          input_names=["input"], # 输入命名
                          output_names=["output"], # 输出命名
                          #dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}},  # 动态轴
                          export_params=True,
                          opset_version=12,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
                          )

    return net

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch == 50:
        lr = 0.0001
    if epoch == 100:
        lr = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Learning Rate for this epoch: {}'.format(lr))

def load_pretrained_weights(model, pretrained_model):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in model_dict and model_dict[k].size() == v.size()}
   
    # 打印将要更新的权重名称
    for k, v in pretrained_dict.items():
        print(f"Loading weights for {k} from pretrained model")
   
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

if __name__ == "__main__":
    
    # 可以选择使用预训练模型
    # my_model = resnet18()
    # pretrained_resnet18 = models.resnet18(pretrained=True)
    # net = load_pretrained_weights(my_model, pretrained_resnet18)

    # my_model = squeezenet()
    # pretrained_squeezenet = models.squeezenet1_0(pretrained=True)
    # net = load_pretrained_weights(my_model, pretrained_squeezenet)

    # net = squeezenet()
    net = resnet18()

    if torch.cuda.is_available():
        net.cuda()
    print("Pretrained weights loaded successfully.")

    criterion = yoloLoss(8, 2, 5, 0.5).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    train_dataset = yoloDataset(root=file_root, list_file='train_labels-QRcode.txt', train=True, transform=[transforms.ToTensor()])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataset = yoloDataset(root=file_root, list_file='test_labels-QRcode.txt', train=False, transform=[transforms.ToTensor()])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f'The dataset has {len(train_dataset)} images')
    print(f'The batch_size is {batch_size}')

    # # Train and validate model
    train(net, train_loader, optimizer, criterion, mode="float", load_model_path=None,num_epochs = 110)

    # Train and validate model
    train(net, train_loader, optimizer, criterion, mode="clamp", load_model_path="./WEIGHT/yolo-float-separable.pth",num_epochs = 25)

    # # Train and validate model
    train(net, train_loader, optimizer, criterion, mode="quant", load_model_path="./WEIGHT/yolo-clamp-separable.pth",num_epochs = 15)

    logfile.close()

