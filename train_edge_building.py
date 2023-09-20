import torch
import torch.nn as nn
from torch.utils import data

from torch.autograd import Variable
import torch.optim as optim

import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import argparse
import numpy as np
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random
from PIL import Image
import csv

import segmentation_models_pytorch as smp

# from utils.loss import CrossEntropy2d
from util.LoadDataOCD import *
import torch.nn as nn

from util.losses import ComboLoss

""""=============================================================================="""
"""                             GPU device 설정                                   """
"""==============================================================================="""
gpu_device = '0' # GPU 번호가 1번인 GPU 사용

"""=============================================================================="""
"""                                                                              """
"""                             주요 파라미터 셋팅                                 """
"""                                                                              """
"""=============================================================================="""
BATCH_SIZE = 4 # 18  
NUM_EPOCH = 30 # epoch
model_mode = 'DeeplabV3p_res101' # Unet_res101 / Unet_res50 / Unetpp_res101 / Unetpp_res50 / FPN_res101 / FPN_res50 / DeeplabV3p_res101 / DeeplabV3p_res50
"""------------------------ LOAD ---------------------------"""
IMG_FORMAT = '.png'
GT_FORMAT = '.png'
INPUT_SIZE = 1024 # 학습으로 들어가는 영상 크기
ORIGINAL_SIZE = 1024 # 데이터셋 영상 크기
DATA_DIRECTORY = 'C:/Users/yonghyun/Desktop/buildingdetection2020-main'
csv_data = DATA_DIRECTORY + '/train_list.csv'
img_folder_name = '/buildings/Training/train_buildings_data' # DATA_DIRECTORY 내 image 가 들어있는 폴더 이름
label_folder_name = '/label_color_contour_pub4' # DATA_DIRECTORY 내 GT image 가 들어있는 폴더 이름
edge_folder_name = '/output/pred/edge_buildings'
"""------------------------------- SAVE -----------------------------------"""
SAVE_PRED_EVERY = 300 # 해당 iter 마다 한번씩 저장
SNAPSHOT_DIR = 'C:/Users/yonghyun/Desktop/smp_building/weight/addedge_fourchannels_DeeplabV3p_res101_resize512_AdamW' # 저장 경로
"""------------------------ function parameters ---------------------------"""
WEIGHT_DECAY = 0.0005 # weight 감소량
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
NUM_CLASSES = 7  # 0:NO building, 1:building
POWER = 0.9
re_size = 512

"""=============================================================================="""
"""                                                                              """
"""                             Parser                                           """
"""                                                                              """
"""=============================================================================="""
def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description='BUILDING DETECTION')  

    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-epoch", type=int, default=NUM_EPOCH,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-scale", default=False,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots(weight files) of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter")
    parser.add_argument("--gpu", type=str, default=gpu_device,
                        help="choose gpu device.")
    parser.add_argument("--img-format", type=str, default=IMG_FORMAT,
                        help="Input image format")
    parser.add_argument("--gt-format", type=str, default=GT_FORMAT,
                        help="Input ground truth format")
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                        help="The size of input image for training")
    parser.add_argument("--imgdata-size", type=int, default=ORIGINAL_SIZE,
                        help="The size of image in the dataset")
    parser.add_argument("--csv-data", type=str, default=csv_data,
                        help="Dataset list for training phase")
    parser.add_argument("--model-mode", type=str, default=model_mode,
                        help="Models : DeepRes101 / DeepRes50 / DeepFCN101")
    parser.add_argument("--img-folder-name", type=str, default=img_folder_name,
                        help="folder name including image data")
    parser.add_argument("--label-folder-name", type=str, default=label_folder_name,
                        help="folder name including GT data")



    return parser.parse_args()

args = get_arguments()
os.environ['CUDA_VISIBLE_DEVICES']= args.gpu

automated_log_path = SNAPSHOT_DIR + "_log.txt" # log 저장 이름
INPUT_SIZE_m = [args.input_size, args.input_size] 
original_size = [args.imgdata_size, args.imgdata_size]  # width, height
IMG_MEAN = np.array((128, 128, 128), dtype=np.float32) # 학습 속도를 위해 RGB 128을 영점으로 둔다. [-128~127], Load code에서 128로 나눔 [-1~0.999]

"""=============================================================================="""
"""                                                                              """
"""                         Learning rate options                                """
"""                                                                              """
"""=============================================================================="""
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_epoch, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


"""=============================================================================="""
"""                                                                              """
"""                                     MAIN                                     """
"""                                                                              """
"""=============================================================================="""
def main():
    cudnn.enabled = True

    """-------------------------- 개발 MODEL ROAD --------------------------"""
    # UNet / Encoders
    # UNet++ / Encoders
    # FPN / Encoders
    # DeepLabV3+ / Encoders
    if args.model_mode == 'Unet_res101':
        model = smp.Unet(encoder_name="resnet101", in_channels=4, classes=7)
    elif args.model_mode == 'Unet_res50':
        model = smp.Unet(encoder_name="resnet50", in_channels=4, classes=7)
    elif args.model_mode == 'Unetpp_res101':
        model = smp.UnetPlusPlus(encoder_name="resnet101", in_channels=4, classes=7)
    elif args.model_mode == 'Unetpp_res50':
        model = smp.UnetPlusPlus(encoder_name="resnet50", in_channels=4, classes=7)
    elif args.model_mode == "FPN_res101":
        model = smp.FPN(encoder_name="resnet101", in_channels=4, classes=7)
    elif args.model_mode == "FPN_res50":
        model = smp.FPN(encoder_name="resnet50", in_channels=4, classes=7)
    elif args.model_mode == "DeeplabV3p_res101":
        model = smp.DeepLabV3Plus(encoder_name="resnet101", in_channels=4, classes=7)
    elif args.model_mode == "DeeplabV3p_res50":
        model = smp.DeepLabV3Plus(encoder_name="resnet50", in_channels=4, classes=7)
    else:
        raise Exception("Please select a model")
    
    model.cuda(0)
    model.train()
    # 쉽게 true로 두면 비용(memory 등) 이 더 들지만 성능이 향상됨.
    cudnn.benchmark = False  # cudnn.benchmark = true -- uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
                            # -- If this is set to false, uses some in-built heuristics that might not always be fastest.

    """-------------------------- FILE SAVE --------------------------"""
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    # log 값을 기록
    with open(automated_log_path, "a") as myfile:
        myfile.write("Epoch\t\titer\t\tloss") 


    """-------------------------- FILE LOAD --------------------------"""

    name_list = []
#     f = open(args.data_dir + args.csv_data, 'r')
    f = open(args.csv_data, 'r')
    names = csv.reader(f)
    for name in names:
        name[0] = name[0] + args.gt_format  # csv 파일 list 에 확장자가 빠진 이름들의 list 이므로
        name_list.append(name[0])
    f.close()

   
    filename_list_imgs = os.listdir(args.data_dir + '/' + args.img_folder_name)
    filename_list_labels = os.listdir(args.data_dir + '/' + args.label_folder_name)
 
    # Thumbs.db 파일 때문에 정확하게 해당 확장자만 불러옴
    image_list_imgs_whole = [file_i for file_i in filename_list_imgs if file_i.endswith(args.img_format)]

    image_list_imgs = name_list

    ITER_SIZE = int(image_list_imgs.__len__() / args.batch_size)  # training dataset 갯수 / batch_size

    trainloader = data.DataLoader(
        LoadAddEdgeOrigin(args.data_dir, args.img_folder_name, image_list_imgs, edge_folder_name, args.label_folder_name, 
                        mean=IMG_MEAN,
                        scale=args.random_scale,
                        re_size=(re_size, re_size)),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)

    trainloader_iter = iter(trainloader)

    # implement model.optim_parameters(args) to handle different models' lr setting
    optimizer = optim.AdamW(model.parameters(),
                          lr=args.learning_rate,
                        #   momentum=args.momentum,
                          weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # bce_loss = torch.nn.BCEWithLogitsLoss()

    loss_function = ComboLoss({'dice': 1, 'focal': 1}, per_image=True).cuda(0)
    


    """----------------------- TRAINING START ------------------------"""
    for i_iter in range(args.num_epoch): 
        trainloader_iter = iter(trainloader)
        # 파라미터를 학습 하겠다.
        for param in model.parameters():
                param.requires_grad = True  

        adjust_learning_rate(optimizer, i_iter)

        for sub_i in range(ITER_SIZE):
            """----------------------- LOAD DATA ------------------------"""
            batch = next(trainloader_iter)
            train_img, train_label, _, _ = batch
            image_input = train_img
            print(image_input.shape)
            image_input = image_input.transpose(1, 3)
            image_input = image_input.transpose(2, 3)
            image_input = Variable(image_input).cuda(0)
            
            # pair image 와 label load
            labels = train_label
            print(labels.shape)
            labels = torch.tensor(labels)
            labels = labels.transpose(1, 3)
            labels = labels.transpose(2, 3)
            labels = Variable(labels).cuda(0)

            """----------------------- RESULTS ------------------------"""  
            pred_comb = model(image_input)


            # pred_comb = torch.squeeze(pred_comb) # 불필요한 1차원 제거
            # labels = labels.unsqueeze(1) # 필요한 1번째 차원 증가
            # print(pred_comb)
            # print("pred_comb['out'] 모양:", pred_comb.shape)
            # print("pred_comb['out'] 타입:", type(pred_comb))
            # print("labels 모양:", labels.shape)
            # print("labels 타입:", type(labels))

            """----------------------- BACKWARD ------------------------""" 
            
            loss_target = loss_function(pred_comb, labels)
            print(f'{model_mode}')

            # proper normalization
            loss = loss_target  
            """ source  loss, backward for differention """
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # weight update
            # torch.cuda.empty_cache() ########### segment fault 해결을 위한 방법

            print('sub_i = {0:3d}/{1:3d},   epoch = {2:3d}/{3:3d},   loss = {4:.3f}'.format(sub_i, ITER_SIZE, i_iter, args.num_epoch, loss))
            """----------------------- SAVE WEIGHT FILE ------------------------""" 
            
            with open(automated_log_path, "a") as myfile: # 원래 있던 값에 덮어쓰기
                    myfile.write("\n%d\t\t%d\t\t%.3f" % (i_iter, sub_i, loss))

            if sub_i % args.save_pred_every == 0 and sub_i != 0:
                print('taking snapshot ...')
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, f'{model_mode}_' + str(sub_i) + '_ep_' + str(i_iter) + '.pth'))

        print('exp = {}'.format(args.snapshot_dir))
        # print('iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f} loss_adv1 = {4:.3f}, loss_adv2 = {5:.3f} loss_D1 = {6:.3f} loss_D2 = {7:.3f}'.format(
        #     i_iter, args.num_steps, loss_seg_value1, loss_seg_value2, loss_adv_target_value1, loss_adv_target_value2, loss_D_value1, loss_D_value2))


        if i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, f'{model_mode}_' + str(i_iter) + '.pth'))


if __name__ == '__main__':
    main()
