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

        self.cnn = cnn_model(num_classes=input_size).to(device)
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

class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8):
        super(SelfAttention, self).__init__()
        self.in_dim = in_dim
        self.heads = heads
        self.head_dim = in_dim // heads
        
        assert (
            self.head_dim * heads == in_dim
        ), "input dim need divide by heads"
        
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.value = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        self.fc_out = nn.Linear(heads * self.head_dim, in_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        
        query = self.query(x).view(batch_size, -1, self.heads, self.head_dim)
        key = self.key(x).view(batch_size, -1, self.heads, self.head_dim)
        value = self.value(x).view(batch_size, -1, self.heads, self.head_dim)
        
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        
        attention_scores = torch.matmul(query, key.permute(0, 1, 3, 2))
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        attention_output = torch.matmul(attention_probs, value)
        
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.heads * self.head_dim)
        
        output = self.fc_out(attention_output)
        
        return output

class SelfAttentionWithEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, heads=8):
        super(SelfAttentionWithEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = SelfAttention(embed_dim, heads=heads)

    def forward(self, x):
        x_embedded = self.embedding(x)
        
        x_attention = self.attention(x_embedded)
        
        return x_attention
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
