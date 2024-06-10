import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNN(nn.Module):
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
        # self.fc = nn.Linear(hidden_size * (bidirectional + 1), num_classes)
    
    def forward(self, x):
        batch_size = x.shape[0] # (bs, 40, 5, 8, 8)
        x = x.flatten(0, 1) # (bs*40, 5, 8, 8)
        x = F.dropout(x, self.dropout)
        x = self.cnn(x) # (bs*40, input_size) x.shape([576,39])
        x = x.reshape((batch_size, self.window, -1)) # (bs, 40, input_size) x.shape([64, 9, 39])
        x = F.dropout(x, self.dropout)
        # x = self.fc(x)

        return x

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
        x = self.cnn(x) # (bs*40, input_size) x.shape([576,39])
        x = x.reshape((batch_size, self.window, -1)) # (bs, 40, input_size) x.shape([64, 9, 39])
        x = F.dropout(x, self.dropout)
        x, _ = self.lstm(x) # x.shape ([64, 9, 256])
        if self.bidirectional:
            x = torch.cat((x[:, -1, :self.hidden_size], x[:, 0, self.hidden_size:]), dim=1)
        else:
            x = x[:, -1, :]
        x = F.dropout(x, self.dropout)
        x = self.fc(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, device, heads=6):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        # print(f'd_model: {d_model},  heads: {heads},  head_dim: {self.head_dim}')
        
        assert (
            self.head_dim * heads == d_model
        ), "input dim need divide by heads"
        
        self.query = nn.Linear(d_model, d_model, bias=False).to(device)
        self.key = nn.Linear(d_model, d_model, bias=False).to(device)
        self.value = nn.Linear(d_model, d_model, bias=False).to(device)
        
        self.out = nn.Linear(d_model, d_model).to(device)

    def scaled_dot_product_attention(self, Q, K, V, uncertainty=None):
        # if uncertainty:
        if uncertainty is not None:
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.head_dim) * uncertainty.sum())
        else:
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.head_dim))
        # print(f'K scaled_dot_product_attention size: {K.size()}')
        # else:
        #     attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.head_dim))

        attention_probs = torch.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_probs, V)
        return output
    
    def split_heads(self, x):
        # print(f'x.size(): {x.size()}')
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.heads, self.head_dim).transpose(-2, -1)

    def combine_heads(self, x):
        batch_size, seq_length, _, _ = x.size()
        return x.transpose(-2, -1).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, uncertainty=None):
        # print(f'K forward size: {K.size()}')
        query = self.split_heads(self.query(Q))
        key = self.split_heads(self.key(K))
        value = self.split_heads(self.value(V))
        
        attention_output = self.scaled_dot_product_attention(query, key, value, uncertainty)

        output = self.out(self.combine_heads(attention_output))
        return output

class FFN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        # print(f'pe: {pe.size()}')
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # print(f'pe siez: {self.pe[:, :x.size(1)].size()}')
        batch_size, seq_length, _ = x.size()
        return x + self.pe[:, :seq_length].expand(batch_size, -1, -1)

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        # x = x.view((-1, 40))
        min_val = torch.min(x)
        max_val = torch.max(x)

        max_int = len(x) - 1
        normalized_array = ((x - min_val) / (max_val - min_val)) * max_int
        # normalized_array.long()
        x = self.embedding(normalized_array.long())

        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, device):

        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, device, num_heads)
        self.feed_forward = FFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, uncertainty):
        tmp = self.norm1(x)
        # print(f'encoder x size: {x.size()}')
        attn_output = self.self_attn(tmp, tmp, tmp, uncertainty)
        x = x + self.dropout(attn_output)
        # print(f'x.shape: {x.shape}')
        tmp = self.norm2(x)
        # print(f'tmp.shape: {tmp.shape}')
        ff_output = self.feed_forward(tmp)
        x = x + self.dropout(ff_output)
        # print(f'after encoder x size: {x.size()}')
        return x
    
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, device):
        super().__init__()
        self.encoder_embedding = Embedding(src_vocab_size, d_model).to(device)
        self.positional_encoding = PositionEncoding(d_model, max_seq_length).to(device)

        #* encoder
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout, device) for 
                                            _ in range(num_layers)])
        #* decoder
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, uncertainty=None):
        enc_output = self.dropout(self.positional_encoding(self.encoder_embedding(src)))

        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, uncertainty)
        # print(f'enc out size: {enc_output.size()}')

        output = self.fc(enc_output)
        # print(f'enc out size: {output.size()}')
        return output.squeeze()
    
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
