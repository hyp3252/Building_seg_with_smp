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
- 
