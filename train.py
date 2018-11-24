import os
import argparse
import logging
from pathlib import Path
import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms

from datasets import Voc2012
from networks import Classifier, PAN, ResNet50, Mask_Classifier
from utils import save_model, get_each_cls_iu, PolyLR
from sklearn.metrics import average_precision_score

parser = argparse.ArgumentParser(description='PAN')
parser.add_argument('--batch_size', type=int, default=4,
                    help='input batch size for training (default: 4)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=4e-3, help='learning rate (default:4e-3)')
parser.add_argument('--alpha', type=float, default=1,
                    help='Cls Loss')
parser.add_argument('--beta', type=float, default=1,
                    help='Semantic Segmentation loss')

args = parser.parse_args()

experiment_name = 'batch_size{:}a{:}b{:}_deeplabv3_cls_detach_div2_debug'.format(args.batch_size, args.alpha, args.beta)
path_log = Path('./log/' + experiment_name + '.log')
try:
    if path_log.exists():
        raise FileExistsError
except FileExistsError:
    print("Already exist log file: {}".format(path_log))
    raise
else:
    logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                                datefmt='%a, %d %b %Y %H:%M:%S',
                                filename=path_log.__str__(),
                                filemode='w'
                                )
    print('Create log file: {}'.format(path_log))

train_transforms = transforms.Compose([
            transforms.Resize((384, 384)),  # Let the smallest edge match
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

mask_transforms = transforms.Compose([
            transforms.Resize((384, 384), interpolation=0),  # Nearest interpolation
        ])

test_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
])

training_data = Voc2012('/home/tom/DISK/DISK2/jian/PASCAL/VOC2012',
                        'train_aug',transform=train_transforms, mask_transform=mask_transforms)
test_data = Voc2012('/home/tom/DISK/DISK2/jian/PASCAL/VOC2012',
                    'val',transform=test_transforms, mask_transform=mask_transforms)
training_loader = torch.utils.data.DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

length_training_dataset = len(training_data)
length_test_dataset = len(test_data)

NUM_CLASS = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

convnet = ResNet50(pretrained=True)
classifier = Classifier(in_features=2048, num_class=NUM_CLASS)
pan = PAN(convnet.blocks[::-1])
mask_classifier = Mask_Classifier(in_features=64, num_class=(NUM_CLASS+1))

convnet.to(device)
classifier.to(device)
pan.to(device)
mask_classifier.to(device)

def train(epoch, optimizer, data_loader):
    convnet.train()
    classifier.train()
    pan.train()
    y_true = []
    y_pred = []
    for batch_idx, (imgs, cls_labels, mask_labels) in enumerate(data_loader):
        imgs, cls_labels, mask_labels = imgs.to(device), cls_labels.to(device), mask_labels.to(device)
        fms_blob, z = convnet(imgs)
        out_cls = classifier(z.detach())

        out_ss = pan(fms_blob[::-1])
        mask_pred = mask_classifier(out_ss)
        mask_pred = F.interpolate(mask_pred, scale_factor=2, mode='bilinear', align_corners=True)
        # Classification Loss
        loss_cls = F.binary_cross_entropy_with_logits(out_cls, cls_labels)

        # Semantic Segmentation Loss
        loss_ss = F.cross_entropy(mask_pred, mask_labels)

        # results
        y_true.append(cls_labels.data.cpu().numpy())
        y_pred.append(torch.sigmoid(out_cls).data.cpu().numpy())

        # Update model
        model_name = [convnet, classifier, pan, mask_classifier]
        for m in model_name:
            m.zero_grad()
        (args.alpha*loss_cls + args.beta*loss_ss).backward()
        model_name = ['convnet', 'classifier', 'pan', 'mask_classifier']
        for m in model_name:
            optimizer[m].step()

        if (batch_idx+1) % 64 == 0:
            acc = average_precision_score(np.concatenate(y_true, 0), np.concatenate(y_pred, 0))
            pixel_acc = mask_pred.max(dim=1)[1].data.cpu().eq(mask_labels.cpu()).float().mean()
            logging.info(
                "Train Epoch:{:}, {:}/{:}, loss_cls:{:.4f}, cls_acc:{:.4f}%, pixel_acc:{:.4f}%".format(
                    epoch, args.batch_size*batch_idx, length_training_dataset, loss_cls, acc * 100, pixel_acc*100))

def test(data_loader):
    global best_acc
    convnet.eval()
    classifier.eval()
    pan.eval()
    all_i_count = []
    all_u_count = []
    y_true = []
    y_pred = []
    for batch_idx, (imgs, cls_labels, mask_labels) in enumerate(data_loader):
        with torch.no_grad():
            imgs, cls_labels = imgs.to(device), cls_labels.to(device)
            fms_blob, z = convnet(imgs)
            out_cls = classifier(z)
            out_ss = pan(fms_blob[::-1])
            mask_pred = mask_classifier(out_ss)
            mask_pred = F.interpolate(mask_pred, scale_factor=2, mode='bilinear', align_corners=True)

        # results
        y_pred.append(torch.sigmoid(out_cls).data.cpu().numpy())
        y_true.append(cls_labels.data.cpu().numpy())

        i_count, u_count = get_each_cls_iu(mask_pred.max(1)[1].cpu().data.numpy(), mask_labels.numpy())
        all_i_count.append(i_count)
        all_u_count.append(u_count)

    acc = average_precision_score(np.concatenate(y_true, 0), np.concatenate(y_pred, 0))
    each_cls_IOU = (np.array(all_i_count).sum(0) / np.array(all_u_count).sum(0))
    mIOU = each_cls_IOU.mean()
    pixel_acc = mask_pred.max(dim=1)[1].data.cpu().eq(mask_labels.cpu()).float().mean()

    logging.info("Length of test set:{:} Test Cls Acc:{:.4f}% Each_cls_IOU:{:} mIOU:{:.4f} PA:{:.4f}".format(length_test_dataset, acc*100, dict(zip(test_data.classes, (100*each_cls_IOU).tolist())), mIOU*100, pixel_acc))
    if mIOU > best_acc:
        logging.info('==>Save model, best mIOU:{:.3f}%'.format(mIOU*100))
        best_acc = mIOU
        state = {'epoch': epoch,
                 'best_acc': best_acc,
                 'convnet': convnet.state_dict(),
                 'pan': pan.state_dict(),
                 'classifier': classifier.state_dict(),
                 'mask_classifier': mask_classifier.state_dict(),
                 'optimizer': optimizer,
                 }
        save_model(state, directory='./checkpoints', filename=experiment_name+'.pkl')

model_name = ['convnet', 'classifier', 'pan', 'mask_classifier']
optimizer = {'convnet': optim.Adam(convnet.parameters(), lr=args.lr, weight_decay=1e-4),
             'classifier': optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=1e-4),
             'pan': optim.Adam(pan.parameters(), lr=args.lr, weight_decay=1e-4),
             'mask_classifier': optim.Adam(mask_classifier.parameters(), lr=args.lr, weight_decay=1e-4)}
optimizer_lr_scheduler = {'convnet': PolyLR(optimizer['convnet'], max_iter=args.epochs, power=0.9),
                          'classifier': PolyLR(optimizer['classifier'], max_iter=args.epochs, power=0.9),
                          'pan': PolyLR(optimizer['pan'], max_iter=args.epochs, power=0.9),
                          'mask_classifier': PolyLR(optimizer['mask_classifier'], max_iter=args.epochs, power=0.9)}

best_acc = 0

for epoch in range(args.epochs+1):
    for m in model_name:
        optimizer_lr_scheduler[m].step(epoch)
    logging.info('Epoch:{:}'.format(epoch))
    train(epoch, optimizer, training_loader)
    if epoch % 1 == 0:
        test(test_loader)