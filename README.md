# Building_seg_with_smp
Satellite Building data Augmentation for segmentation.
- segmentation models pytorch의 pretrained 모델을 활용하여 기존의 위성영상 데이터를 augmentation하여 성능 향상이 목적

# Satellite Building DATA
- data source from
https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=73
- 1238장의 train_data, 159장의 test_data로 이루어져있음.
- 1024 X 1024 크기의 이미지 데이터와 Label 데이터로 이루어짐.
- train set, test set의 Label 데이터는 0:배경, 1:소형, 2:아파트, 3:공장, 4:중형단독시설, 5:대형시설, 6:contour로 따로 **전처리**를 하여 총 7개의 class를 가짐.(draw_polygons.py)
- **Data Image**

|PNG FIle(RGB)|PNG FIle(Label)|
|:---------:|:----:|
|![rgb_img](https://github.com/hyp3252/Building_seg_with_smp/blob/main/images/rgb_image.png?raw=true)|![label](https://github.com/hyp3252/Building_seg_with_smp/blob/main/images/label.png?raw=true)|


# Satellite Building Data Segmentation add SAM Edge data Augmentation
### 1. Edge image 추출하여 RGB image와 같은 이름으로 지정된 경로로 저장
<pre>
  $ cd segment-anything-edge-detection
  $ python save_edge.py
</pre>

### 2. Edge image + RGB image (4 channels)
|PNG FIle(RGB img)|PNG FIle(Edge img)|
|:---------:|:----:|
|![rgb_img](https://github.com/hyp3252/Building_seg_with_smp/blob/main/images/rgb_image.png?raw=true)|![Edge_img](https://github.com/hyp3252/Building_seg_with_smp/blob/main/images/SAM_edge_detection.png?raw=true)|

### 3. Segmentation Models Pytorch(SMP)
- 사용한 모델 : UNet, FPN, DeepLabV3+
- 사용한 인코더 : Resnet-50, Resnet-101

<pre>
  import segmentation_models_pytorch as smp

  model = smp.Unet(
            encoder_name="resnet50",        # choose encoder
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=4,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=7,)                      # model output channels (number of classes in your dataset)
</pre>


# 연구방향
### 1. RGB image를 학습하여 test image로 mIoU 산출
### 2. RGB + Edge image를 np.concatenate으로 4채널 input image를 생성하여 학습하여 test image로 mIoU 산출
- test 코드에 사용할 모듈은 edge_utils에서
### 3. 사용한 모델은 UNet-resnet50, UNet-resnet101, FPN-resnet50, FPN-resnet101, DeepLabV3+-resnet50, DeepLabV3+-resnet101
