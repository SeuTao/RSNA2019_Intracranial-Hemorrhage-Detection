# Final Sequence Model

After single model training, we use this sequence model to do stacking.

#### Dependencies
- imgaug == 0.2.8
- opencv-python==3.4.2
- scikit-image==0.14.0
- scikit-learn==0.19.1
- scipy==1.1.0
- torch==1.1.0.
- torchvision==0.2.2

#### Path Setup
Set data path to your own in ./setting.py

#### Sequence Stacking Model Training
sequence stacking verison1：
```
CUDA_VISIBLE_DEVICES=0 python stacking_study_feature_cnn_lstm_version1_max_FeaDim.py
```
sequence stacking verison2：
```
CUDA_VISIBLE_DEVICES=0 python stacking_study_feature_cnn_lstm_version2_max_FeaDim.py
```
sequence stacking verison3：
```
CUDA_VISIBLE_DEVICES=0 python stacking_study_feature_cnn_lstm_version3_max_FeaDim.py
```
The final submissions are in the folder ../FinalSubmission/version2/submission_tta.csv


















