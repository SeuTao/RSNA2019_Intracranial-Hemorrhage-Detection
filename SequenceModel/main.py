import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from seq_dataset import *
import warnings
import os
warnings.filterwarnings('ignore')

fold_index = -1
fold_num = 5
Add_position = True
lstm_layers = 2
seq_len = 24
hidden = 96
drop_out = 0.5
train_epoch = 40

class_num = 6
model_save_dir = os.path.join(final_output_path, 'version3')
if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

def bce_loss(input, target, OHEM_percent=None, class_num = None):
    if OHEM_percent is None:
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='mean')
        return loss
    else:
        loss = F.binary_cross_entropy_with_logits(input, target, reduce=False)
        value, index= loss.topk(int(class_num * OHEM_percent), dim=1, largest=True, sorted=True)
        return value.mean()

def mutl_log_loss(y_truth,y_pre,index):
    y_tre_tmp=y_truth[:,index]
    y_pre_tmp=y_pre[:,index]
    tmp_loss=log_loss(y_tre_tmp, y_pre_tmp,labels=[0,1])
    return tmp_loss

def criterion(logit, labels):
    w = [2.0,1.0,1.0,1.0,1.0,1.0]
    loss = [bce_loss(logit[:, 0, :, i:i+1], labels[:,:, i:i+1])*w[i] for i in range(6)]
    loss = sum(loss) / sum(w)
    return loss

class SequenceModel(nn.Module):
    def __init__(self, model_num):
        super(SequenceModel, self).__init__()

        # version1
        self.fea_conv = nn.Sequential(nn.Dropout2d(drop_out),
                                      nn.Conv2d(feature_dim, 512, kernel_size=(1, 1), stride=(1,1),padding=(0,0), bias=False),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(),
                                      nn.Dropout2d(drop_out),
                                      nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(),
                                      nn.Dropout2d(drop_out),
                                      )

        self.fea_first_final = nn.Sequential(nn.Conv2d(128*feature_num, 6, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True))

        # # bidirectional GRU
        self.hidden_fea = hidden
        self.fea_lstm = nn.GRU(128*feature_num, self.hidden_fea, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.fea_lstm_final = nn.Sequential(nn.Conv2d(1, 6, kernel_size=(1, self.hidden_fea*2), stride=(1, 1), padding=(0, 0), dilation=1, bias=True))

        ratio = 4
        if Add_position:
            model_num += 2
        else:
            model_num += 1

        self.conv_first = nn.Sequential(nn.Conv2d(model_num, 128*ratio, kernel_size=(5, 1), stride=(1,1),padding=(2,0),dilation=1, bias=False),
                                        nn.BatchNorm2d(128*ratio),
                                        nn.ReLU(),
                                        nn.Conv2d(128*ratio, 64*ratio, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0),dilation=2, bias=False),
                                        nn.BatchNorm2d(64*ratio),
                                        nn.ReLU())

        self.conv_res = nn.Sequential(nn.Conv2d(64 * ratio, 64 * ratio, kernel_size=(3, 1), stride=(1, 1),padding=(4, 0),dilation=4, bias=False),
                                      nn.BatchNorm2d(64 * ratio),
                                      nn.ReLU(),
                                      nn.Conv2d(64 * ratio, 64 * ratio, kernel_size=(3, 1), stride=(1, 1),padding=(2, 0),dilation=2, bias=False),
                                      nn.BatchNorm2d(64 * ratio),
                                      nn.ReLU(),)

        self.conv_final = nn.Sequential(nn.Conv2d(64*ratio, 1, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), dilation=1,bias=False))

        # bidirectional GRU
        self.hidden = hidden
        self.lstm = nn.GRU(64*ratio*6, self.hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.final = nn.Sequential(nn.Conv2d(1, 6, kernel_size=(1, self.hidden*2), stride=(1, 1), padding=(0, 0), dilation=1, bias=True))


    def forward(self, fea, x):
        batch_size, _, _, _ = x.shape

        fea = self.fea_conv(fea)
        fea = fea.permute(0, 1, 3, 2).contiguous()
        fea = fea.view(batch_size, 128 * feature_num, -1).contiguous()
        fea = fea.view(batch_size, 128 * feature_num, -1, 1).contiguous()
        fea_first_final = self.fea_first_final(fea)
        #################################################
        out0 = fea_first_final.permute(0, 3, 2, 1)
        #################################################

        # bidirectional GRU
        fea = fea.view(batch_size, 128 * feature_num, -1).contiguous()
        fea = fea.permute(0, 2, 1).contiguous()
        fea, _ = self.fea_lstm(fea)
        fea = fea.view(batch_size, 1, -1, self.hidden_fea * 2)
        fea_lstm_final = self.fea_lstm_final(fea)
        fea_lstm_final = fea_lstm_final.permute(0, 3, 2, 1)
        #################################################
        out0 += fea_lstm_final
        #################################################

        out0_sigmoid = torch.sigmoid(out0)
        x = torch.cat([x, out0_sigmoid], dim = 1)
        x = self.conv_first(x)
        x = self.conv_res(x)
        x_cnn = self.conv_final(x)
        #################################################
        out = x_cnn
        #################################################

        # bidirectional GRU
        x = x.view(batch_size, 256, -1, 6)
        x = x.permute(0,2,1,3).contiguous()
        x = x.view(batch_size, x.size()[1], -1).contiguous()
        x, _= self.lstm(x)
        x = x.view(batch_size, 1, -1, self.hidden*2)
        x = self.final(x)
        x = x.permute(0,3,2,1)
        #################################################
        out += x
        #################################################
        #res
        return out, out0

log = open(os.path.join(model_save_dir,'log.txt'),'a')
if 1:
        kf = KFold(n_splits=fold_num, shuffle=True, random_state=48)
        all_df = pd.read_csv(r'./csv/train_meta_id_seriser.csv')
        StudyInstance = list(all_df['StudyInstance'].unique())
        print(len(StudyInstance))
        dict_ = get_train_dict()
        for s_fold, (train_idx, valid_idx) in enumerate(kf.split(StudyInstance)):
            print('fold ' + str(s_fold))

            if s_fold != fold_index:
                continue

            batch_size = 128
            train_data = StackingDataset_study( dict_, X,y, train_idx, seq_len = seq_len, mode='train', Add_position=Add_position)
            train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True, shuffle=True)
            val_data = StackingDataset_study( dict_, X,y, valid_idx, seq_len = -1, mode='valid', Add_position=Add_position)
            val_loader = DataLoader(val_data, batch_size=1, num_workers=8, pin_memory=True, drop_last=False, shuffle=False)

            model = SequenceModel(model_num = model_num).cuda()
            print(model)

            optimizer = optim.Adam(model.parameters(), lr=3e-4)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 30, 40], gamma=0.1)

            best_score = 100
            for epoch in range(train_epoch):
                running_loss = 0.0
                model.train()
                for fea, data, labels in tqdm(train_loader, position=0):
                    fea, data, labels = fea.float().cuda(), data.float().cuda(), labels.float().cuda()

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(True):
                        fea, data, labels = fea.cuda(), data.cuda(), labels.cuda()
                        logit, logit_help = model(fea, data)
                        loss0 = criterion(logit, labels)
                        loss1 = criterion(logit_help, labels)

                        loss = loss0 + loss1
                        loss.backward()
                        optimizer.step()

                    running_loss += loss * data.shape[0]

                train_loss = running_loss / train_data.__len__()
                scheduler.step()

                running_loss = 0
                count = 0
                model.eval()
                num_sample = 0
                for fea, data, labels in tqdm(val_loader, position=0):
                    fea, data, labels = fea.float().cuda(), data.float().cuda(), labels.float().cuda()

                    with torch.set_grad_enabled(False):
                        logit,_ = model(fea, data)
                        loss = criterion(logit, labels)

                    count += 1
                    running_loss += loss.item() * data.size(2)
                    num_sample += data.size(2)

                val_loss = running_loss / num_sample
                print(model_save_dir)
                print(str(epoch), 'train_loss:{} val_loss:{} score:{}'.format(train_loss, val_loss, val_loss))
                log.write('fold: '+str(s_fold)+' '+str(epoch)+' train_loss:{} val_loss:{} score:{}'.format(train_loss, val_loss, val_loss))
                log.write('\n')
                if best_score > val_loss:
                    best_score = val_loss
                    print('save max score!!!!!!!!!!!!')
                    log.write('save max score!!!!!!!!!!!!')
                    log.write('\n')
                    torch.save(model.state_dict(), os.path.join(model_save_dir,'fold_' + str(s_fold) + '.pt'))
                gc.collect()

if 1:
    kf = KFold(n_splits=fold_num, shuffle=True, random_state=48)
    all_df = pd.read_csv(r'./csv/train_meta_id_seriser.csv')
    StudyInstance = list(all_df['StudyInstance'].unique())
    print(len(StudyInstance))
    dict_ = get_train_dict()

    logit_list = []
    label_list = []
    for s_fold, (train_idx, valid_idx) in enumerate(kf.split(StudyInstance)):

        batch_size = 128
        val_data = StackingDataset_study(dict_, X, y, valid_idx, seq_len=-1, mode='valid', reverse=True, Add_position=Add_position)
        val_loader = DataLoader(val_data, batch_size=1, num_workers=8, pin_memory=True, drop_last=False, shuffle=False)
        model = SequenceModel(model_num=model_num).cuda()

        print('fold ' + str(s_fold))
        model.load_state_dict(torch.load(os.path.join(model_save_dir,'fold_'+str(s_fold) + '.pt')))
        model.eval()
        for fea, data, labels in tqdm(val_loader, position=0):
            fea, data, labels = fea.float().cuda(), data.float().cuda(), labels.float().cuda()

            with torch.set_grad_enabled(False):
                logit,_ = model(fea, data)
                logit_list.append(logit)
                label_list.append(labels)
        print('===============================================================================================')

    running_loss = 0
    num_sample =0

    for logit, labels in zip(logit_list, label_list):
        loss = criterion(logit, labels)
        running_loss += loss.item() * logit.size(2)
        num_sample += logit.size(2)
    val_loss = running_loss / num_sample
    print(val_loss)
    log.write(str(val_loss))
    log.write('\n')

    logit_list_flip = []
    label_list_flip = []
    for s_fold, (train_idx, valid_idx) in enumerate(kf.split(StudyInstance)):
        batch_size = 128

        val_data = StackingDataset_study(dict_, X, y, valid_idx, seq_len=-1, mode='valid', reverse=False, Add_position=Add_position)
        val_loader = DataLoader(val_data, batch_size=1, num_workers=8, pin_memory=True, drop_last=False, shuffle=False)
        model = SequenceModel(model_num=model_num).cuda()

        print('fold ' + str(s_fold))
        model.load_state_dict(torch.load(os.path.join(model_save_dir, 'fold_' + str(s_fold) + '.pt')))
        model.eval()
        for fea, data, labels in tqdm(val_loader, position=0):
            fea, data, labels = fea.float().cuda(), data.float().cuda(), labels.float().cuda()

            with torch.set_grad_enabled(False):
                logit,_ = model(fea, data)
                logit_list_flip.append(logit)
                label_list_flip.append(labels)

        print('===============================================================================================')

    running_loss = 0
    num_sample =0
    for logit, labels in zip(logit_list_flip, label_list_flip):
        loss = criterion(logit, labels)
        running_loss += loss.item() * logit.size(2)
        num_sample += logit.size(2)
    val_loss = running_loss / num_sample
    print(val_loss)
    log.write(str(val_loss))
    log.write('\n')

    running_loss = 0
    num_sample =0
    for logit,logit_flip, labels, labels_flip in zip(logit_list,logit_list_flip, label_list, label_list_flip):
        logit = logit.cpu().numpy()
        logit_flip = logit_flip.cpu().numpy()
        logit = (logit + logit_flip[:,:,::-1,:]) / 2.0
        logit = torch.from_numpy(logit)
        logit = logit.float().cuda()

        loss = criterion(logit, labels)
        running_loss += loss.item() * logit.size(2)
        num_sample += logit.size(2)

    val_loss = running_loss / num_sample
    print(val_loss)
    log.write('final!!!!!!!!!!!!')
    log.write('\n')
    log.write(str(val_loss))
    log.write('\n')


if 1:
    predicts_list = []
    for s_fold in range(fold_num):
        running_loss = 0
        num_sample =0

        batch_size = 128
        test_id_dict = get_test_dict()
        dataset = StackingDataset_study(test_id_dict, X_test, None, None, seq_len=-1, mode='test', reverse=False, Add_position=Add_position)
        val_loader = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True, drop_last=False, shuffle=False)
        model = SequenceModel(model_num=model_num).cuda()

        print('fold ' + str(s_fold))
        model.load_state_dict(torch.load(os.path.join(model_save_dir,'fold_'+str(s_fold) + '.pt')))
        model.eval()

        filenames_list = []
        for filenames, inputs_fea, inputs in tqdm(val_loader, position=0):
            filenames_list.extend(filenames)

            inputs = inputs.float().cuda()
            inputs_fea= inputs_fea.float().cuda()

            with torch.set_grad_enabled(False):
                logit,_ = model(inputs_fea, inputs)
                logit = torch.sigmoid(logit)

            logit = logit.cpu().numpy()
            logit = logit.reshape([-1, 6])

            if num_sample != 0:
                predicts = np.vstack((predicts, logit))
            else:
                predicts = logit
            num_sample += inputs.size(2)

        print(predicts.shape)
        print(num_sample)
        predicts_list.append(predicts)

    final = np.mean(predicts_list,axis=0)
    predicts_list = []
    for s_fold in range(5):
        running_loss = 0
        num_sample =0

        batch_size = 128
        test_id_dict = get_test_dict()
        dataset = StackingDataset_study(test_id_dict, X_test, None, None, seq_len=-1, mode='test', reverse=True, Add_position=Add_position)
        val_loader = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True, drop_last=False, shuffle=False)
        model = SequenceModel(model_num=model_num).cuda()

        print('fold ' + str(s_fold))
        model.load_state_dict(torch.load(os.path.join(model_save_dir,'fold_'+str(s_fold) + '.pt')))
        model.eval()

        filenames_list = []
        for filenames, inputs_fea, inputs in tqdm(val_loader, position=0):
            filenames_list.extend(filenames)

            inputs = inputs.float().cuda()
            inputs_fea= inputs_fea.float().cuda()

            with torch.set_grad_enabled(False):
                logit,_ = model(inputs_fea, inputs)
                logit = torch.sigmoid(logit)

            logit = logit.cpu().numpy()
            logit = logit.reshape([-1, 6])
            logit = logit[::-1, :]

            if num_sample != 0:
                predicts = np.vstack((predicts, logit))
            else:
                predicts = logit
            num_sample += inputs.size(2)

        print(predicts.shape)
        print(num_sample)
        predicts_list.append(predicts)

    final_flip = np.mean(predicts_list,axis=0)
    final = (final + final_flip)/2.0

    filenames_list = list(np.asarray(filenames_list).reshape([-1]))
    test_df = pd.DataFrame()
    test_df['filename'] = filenames_list

    test_df = test_df.join(pd.DataFrame(final, columns=[
        'any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'
    ]))
    # Unpivot table, i.e. wide (N x 6) to long format (6N x 1)
    test_df = test_df.melt(id_vars=['filename'])
    # Combine the filename column with the variable column
    test_df['ID'] = test_df.filename.apply(lambda x: x.replace('.dcm', '')) + '_' + test_df.variable
    test_df['Label'] = test_df['value']
    test_df[['ID', 'Label']].to_csv(os.path.join(model_save_dir,'submission_tta.csv'), index=False)

