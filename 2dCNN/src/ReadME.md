1. prepare csv file
python3 prepare_csv.py -root_path -train_dcm_path -test_dcm_path -save_path
root_path         kaggle data path
train_dcm_path    train dicom data path
test_dcm_path     test dicom data path
save_path         output path

2. convert dcm to png
python3 prepare_data.py -dcm_path stage_1_train_images -png_path train_png
python3 prepare_data.py -dcm_path stage_1_test_images -png_path test_png

3. train
python3 train_model.py -backbone DenseNet121_change_avg -img_size 256 -tbs 256 -vbs 128 -save_path DenseNet121_change_avg_256
python3 train_model.py -backbone DenseNet169_change_avg -img_size 256 -tbs 256 -vbs 128 -save_path DenseNet169_change_avg_256
python3 train_model.py -backbone se_resnext101_32x4d -img_size 256 -tbs 80 -vbs 40 -save_path se_resnext101_32x4d_256

4. predict
python3 predict.py -backbone DenseNet121_change_avg -img_size 256 -tbs 4 -vbs 4 -spth DenseNet121_change_avg_256
python3 predict.py -backbone DenseNet169_change_avg -img_size 256 -tbs 4 -vbs 4 -spth DenseNet169_change_avg_256
python3 predict.py -backbone se_resnext101_32x4d -img_size 256 -tbs 4 -vbs 4 -spth se_resnext101_32x4d_256