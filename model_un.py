import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_LSTM(nn.Module):
    def __init__(self, cnn_model, input_size, hidden_size, num_layers, num_classes, dropout, bidirectional, device, window=40):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.window = window

        # self.cnn = cnn_model(num_classes=input_size).to(device)
        self.std_prediction_fc = nn.Linear(input_size, 1)
        self.cnn = nn.Sequential(cnn_model(num_classes=input_size).to(device),
                                nn.Dropout(p=dropout))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * (bidirectional + 1), num_classes)
    
    def forward(self, x):
        batch_size = x.shape[0] # (bs, 40, 5, 8, 8)
        x = x.flatten(0, 1) # (bs*40, 5, 8, 8)
        x = self.cnn(x) # (bs*40, input_size)
        x = x.reshape((batch_size, self.window, -1)) # (bs, 40, input_size)
        x = F.dropout(x, self.dropout)
        x, _ = self.lstm(x)
        if self.bidirectional:
            x = torch.cat((x[:, -1, :self.hidden_size], x[:, 0, self.hidden_size:]), dim=1)
        else:
            x = x[:, -1, :]
        x = F.dropout(x, self.dropout)
        x = self.fc(x)
        return x


# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)


# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()

#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1

#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)


class MARS(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(MARS, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3, stride=1, padding='same')
        # self.ca1 = ChannelAttention(16)
        # self.sa1 = SpatialAttention()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same')
        # self.ca2 = ChannelAttention(32)
        # self.sa2 = SpatialAttention()

        self.bn1 = nn.BatchNorm2d(32, momentum=0.95)
        self.l1 = nn.Linear(in_features=2048, out_features=512)

        self.bn2 = nn.BatchNorm1d(512, momentum=0.95)
        self.l2 = nn.Linear(in_features=512, out_features=39)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = self.ca1(x) * x
        # x = self.sa1(x) * x
        x = F.dropout(x, 0.3)
        x = F.relu(self.conv2(x))
        # x = self.ca2(x) * x
        # x = self.sa2(x) * x
        x = F.dropout(x, 0.3)

        x = self.bn1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.l1(x))

        x = self.bn2(x)
        x = F.dropout(x, 0.4)
        x = self.l2(x)

        return x
