import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SequenceModel(nn.Module):
    def __init__(self, model_num, feature_dim, feature_num,
                 lstm_layers, hidden, drop_out, Add_position):
        super(SequenceModel, self).__init__()

        self.feature_num=feature_num

        # seq model 1
        self.fea_conv = nn.Sequential(
                                      nn.Dropout2d(drop_out),
                                      nn.Conv2d(feature_dim, 512, kernel_size=(1, 1), stride=(1,1), padding=(0,0), bias=False),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(),
                                      nn.Dropout2d(drop_out),
                                      nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(),
                                      nn.Dropout2d(drop_out),
                                      )

        self.fea_first_final = nn.Sequential(nn.Conv2d(128 * feature_num, 6, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True))

        # # bidirectional GRU
        self.hidden_fea = hidden
        self.fea_lstm = nn.GRU(128 * feature_num, self.hidden_fea, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.fea_lstm_final = nn.Sequential(nn.Conv2d(1, 6, kernel_size=(1, self.hidden_fea*2), stride=(1, 1), padding=(0, 0), dilation=1, bias=True))

        ratio = 4
        if Add_position:
            model_num += 2
        else:
            model_num += 1

        # seq model 2
        self.conv_first = nn.Sequential(nn.Conv2d(model_num, 128*ratio, kernel_size=(5, 1), stride=(1,1), padding=(2,0), dilation=1, bias=False),
                                        nn.BatchNorm2d(128*ratio),
                                        nn.ReLU(),
                                        nn.Conv2d(128*ratio, 64*ratio, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0), dilation=2, bias=False),
                                        nn.BatchNorm2d(64*ratio),
                                        nn.ReLU())

        self.conv_res = nn.Sequential(nn.Conv2d(64 * ratio, 64 * ratio, kernel_size=(3, 1), stride=(1, 1), padding=(4, 0), dilation=4, bias=False),
                                      nn.BatchNorm2d(64 * ratio),
                                      nn.ReLU(),
                                      nn.Conv2d(64 * ratio, 64 * ratio, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0), dilation=2, bias=False),
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
        fea = fea.view(batch_size, 128 * self.feature_num, -1).contiguous()
        fea = fea.view(batch_size, 128 * self.feature_num, -1, 1).contiguous()
        fea_first_final = self.fea_first_final(fea)
        #################################################
        out0 = fea_first_final.permute(0, 3, 2, 1)
        #################################################

        # bidirectional GRU
        fea = fea.view(batch_size, 128 * self.feature_num, -1).contiguous()
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

if __name__ == '__main__':
    model = SequenceModel(model_num=15, feature_dim = 128, feature_num=16,
                          lstm_layers = 2, hidden=128, drop_out=0.5,
                          Add_position = True)
    print(model)