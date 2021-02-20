import os
import pandas as pd
import numpy as np
import gc
from settings import *
import random

def save_study():
    if not os.path.exists(csv_root):
        os.makedirs(csv_root)

    all_df = pd.read_csv(rf'{csv_root}/train_meta_id_seriser.csv')
    StudyInstance = list(all_df['StudyInstance'].unique())
    random.shuffle(StudyInstance)

    for study in StudyInstance:
        save_path = os.path.join(csv_root, 'study_csv', study + '.csv')
        if not os.path.exists(save_path):
            df = all_df[all_df['StudyInstance'] == study]
            if not os.path.exists(os.path.join(csv_root, 'study_csv')):
                os.makedirs(os.path.join(csv_root, 'study_csv'))
            df.to_csv(save_path)
            print(study)

    all_df = pd.read_csv(rf'{csv_root}/test_meta_id_seriser_stage2.csv')
    StudyInstance = list(all_df['StudyInstance'].unique())
    random.shuffle(StudyInstance)

    for study in StudyInstance:
        save_path = os.path.join(csv_root, 'study_csv', study + '.csv')
        if not os.path.exists(save_path):
            df = all_df[all_df['StudyInstance'] == study]
            if not os.path.exists(os.path.join(csv_root, 'study_csv')):
                os.makedirs(os.path.join(csv_root, 'study_csv'))
            df.to_csv(save_path)
            print(study)

feature_dim = 2048
def get_train_test_feature(dir):
    model_name = os.path.split(dir)[1]

    fea_name = os.path.join(dir, model_name+'_val_oof_feature_TTA_stage2_finetune.npy')
    if not os.path.exists(fea_name):
        fea_name = os.path.join(dir, model_name + '_val_oof_feature_TTA.npy')

    print(' '+os.path.split(fea_name)[1])
    val_feature = np.load(fea_name).astype(np.float16)
    val_fea = np.zeros([val_feature.shape[0], feature_dim, 1], dtype=np.float16)
    val_fea[:,0:val_feature.shape[1],0] = val_feature
    del val_feature

    fea_name = os.path.join(dir, model_name+'_test_feature_TTA_stage2_finetune.npy')
    if not os.path.exists(fea_name):
        fea_name = os.path.join(dir, model_name + '_test_feature_TTA_stage2.npy')

    if os.path.exists(fea_name):
        print(' '+ os.path.split(fea_name)[1])
        test_feature = np.load(fea_name).astype(np.float16)
        test_fea = np.zeros([test_feature.shape[0], feature_dim, 1], dtype=np.float16)
        test_fea[:,0:test_feature.shape[1],0] = test_feature
        del test_feature
    else:
        print(' test fea is None')
        test_fea = None

    return val_fea, test_fea

train_features = []
test_features = []

#################################################################################################################
for model_name in os.listdir(os.path.join(feature_path, r'stage2_finetune')):
    print(model_name)
    val_fea,test_fea = get_train_test_feature(dir = os.path.join(feature_path, r'stage2_finetune', model_name))
    train_features.append(val_fea)
    if test_fea is not None:
        test_features.append(test_fea)
#################################################################################################################

train_fea = np.concatenate(train_features,axis=2)
print(train_fea.shape)

if len(test_features) > 0:
    test_fea = np.concatenate(test_features,axis=2)
    print(test_fea.shape)

feature_num = train_fea.shape[2]
gc.collect()

v0 = list(pd.read_csv(f'{csv_root}/val_fold0.csv')['filename'])
v1 = list(pd.read_csv(f'{csv_root}/val_fold1.csv')['filename'])
v2 = list(pd.read_csv(f'{csv_root}/val_fold2.csv')['filename'])
v3 = list(pd.read_csv(f'{csv_root}/val_fold3.csv')['filename'])
v4 = list(pd.read_csv(f'{csv_root}/val_fold4.csv')['filename'])
fea_ids = v0+v1+v2+v3+v4
fea_ids = [tmp.replace('.dcm','') for tmp in fea_ids]
fea_id_dict = {}

i = 0
for id in fea_ids:
    fea_id_dict[id] = i
    i += 1

csv = f'{csv_root}/stage_2_sample_submission.csv'
df = pd.read_csv(csv)
df['filename'] = df['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".dcm")
df['type'] = df['ID'].apply(lambda st: st.split('_')[2])
df = pd.DataFrame(df.filename.unique(), columns=['filename'])

df["filename"] = [tmp.replace('.dcm', '') for tmp in df["filename"]]
df["filename"] = [tmp.replace('.png', '') for tmp in df["filename"]]
test_fea_ids = list(df['filename'])

i = 0
for id in test_fea_ids:
    fea_id_dict[id] = i
    i += 1



