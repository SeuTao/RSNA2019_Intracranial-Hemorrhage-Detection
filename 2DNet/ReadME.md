# 2D CNN Classifier

## Pretrained models
- seresnext101_256*256 [\[seresnext101\]](https://drive.google.com/open?id=18Py5eW1E4hSbTT6658IAjQjJGS28grdx)
- densenet169_256*256 [\[densenet169\]](https://drive.google.com/open?id=1vCsX12pMZxBmuGGNVnjFFiZ-5u5vD-h6)
- densenet121_512*512 [\[densenet121\]](https://drive.google.com/open?id=1o0ok-6I2hY1ygSWdZOKmSD84FsEpgDaa)

## Preprocessing
![image](https://github.com/SeuTao/RSNA2019_1st_place_solution/blob/master/docs/preprocessing.png)

Prepare csv file:

download [\[data.zip\]](https://drive.google.com/open?id=1buISR_b3HQDU4KeNc_DmvKTYJ1gvj5-3)


1. convert dcm to png
```
python3 prepare_data.py -dcm_path stage_1_train_images -png_path train_png
python3 prepare_data.py -dcm_path stage_1_test_images -png_path train_png
python3 prepare_data.py -dcm_path stage_2_test_images -png_path test_png
```

2. train

```
python3 train_model.py -backbone DenseNet121_change_avg -img_size 256 -tbs 256 -vbs 128 -save_path DenseNet121_change_avg_256
python3 train_model.py -backbone DenseNet169_change_avg -img_size 256 -tbs 256 -vbs 128 -save_path DenseNet169_change_avg_256
python3 train_model.py -backbone se_resnext101_32x4d -img_size 256 -tbs 80 -vbs 40 -save_path se_resnext101_32x4d_256
```

3. predict
```
python3 predict.py -backbone DenseNet121_change_avg -img_size 256 -tbs 4 -vbs 4 -spth DenseNet121_change_avg_256
python3 predict.py -backbone DenseNet169_change_avg -img_size 256 -tbs 4 -vbs 4 -spth DenseNet169_change_avg_256
python3 predict.py -backbone se_resnext101_32x4d -img_size 256 -tbs 4 -vbs 4 -spth se_resnext101_32x4d_256
```