import torch
import torch.nn as nn
import torch.nn.functional as F


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias

class conv_blocks(nn.Module):
    def __init__(self, dilation=1):
        super(conv_blocks, self).__init__()
        self.kernel_size = 7
        self.dilation = dilation
        # print(dila)

        # SAME padding formula → ensures output length = input length
        padding = dilation * (self.kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel_size,
            dilation=dilation,
            padding=padding,
            padding_mode='circular',
            bias=False
        )

        # fixed moving-average weights
        w = torch.ones(1, 1, self.kernel_size) / self.kernel_size
        self.conv.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        """
        x: [B, L, C]
        Applies Conv1D separately to each feature dimension.
        """
        B, L, C = x.shape
        outs = []

        for c in range(C):
            xc = x[:, :, c].unsqueeze(1)        # [B,1,L]
            yc = self.conv(xc).squeeze(1)       # [B,L]
            outs.append(yc)

        return torch.stack(outs, dim=-1)        # [B,L,C]

class MultiConvBlocks(nn.Module):
    def __init__(self, dilations=[1, 3, 5]):
        super(MultiConvBlocks, self).__init__()
        self.branches = nn.ModuleList([conv_blocks(d) for d in dilations])
        self.fuse_conv = nn.Conv1d(
            in_channels=len(dilations),
            out_channels=1,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            bias=False
        )

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        outputs = []

        # Run separate dilation convs
        for branch in self.branches:
            y = branch(x)                # [B, L, C]
            outputs.append(y)

        # Stack: [B, L, C, K]
        stacked = torch.stack(outputs, dim=-1)

        # fuse along branch dimension
        # reshape to conv1d format: (B*C, K, L)
        stacked = stacked.permute(0, 2, 3, 1).reshape(B*C, len(self.branches), L)

        fused = self.fuse_conv(stacked).reshape(B, C, L).permute(0, 2, 1)
        # print('yes')
        return fused

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k
        # print(top_k)

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        # print('sucess')
        return x_season, x_trend
    
class DFT_series_decomp_multi(nn.Module):
    """
    Multi DFT-based series decomposition block
    Each top_k produces one seasonal-trend pair
    The model learns how to combine them
    """
    def __init__(self, top_k_list):
        super(DFT_series_decomp_multi, self).__init__()
        self.top_k_list = top_k_list
        self.num_k = len(top_k_list)

        # linear layer to learn combination weights
        self.weight_layer = nn.Linear(1, self.num_k)

    def single_dft(self, x, top_k):
        # print('success')
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[..., 0] = 0  # remove DC
        top_k_freq, _ = torch.topk(freq, top_k)
        xf_filtered = xf.clone()
        xf_filtered[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf_filtered)
        x_trend = x - x_season
        return x_season, x_trend

    def forward(self, x):
        # print('success')
        season_list = []
        trend_list  = []

        for top_k in self.top_k_list:
            seasonal, trend = self.single_dft(x, top_k)
            season_list.append(seasonal.unsqueeze(-1))
            trend_list.append(trend.unsqueeze(-1))

        # stack along last dimension: (B,L,C,K)
        season_stack = torch.cat(season_list, dim=-1)
        trend_stack  = torch.cat(trend_list,  dim=-1)

        # softmax weights
        weights = torch.softmax(self.weight_layer(x.unsqueeze(-1)), dim=-1)

        # weighted combination
        seasonal = torch.sum(season_stack * weights, dim=-1)
        trend = torch.sum(trend_stack * weights, dim=-1)

        return seasonal, trend

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        # print('sucess')
        return res, moving_mean 

class conv_series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, dilation=1):
        super(conv_series_decomp, self).__init__()
        self.moving_avg = conv_blocks(dilation)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        # print('success')
        return res, moving_mean
    
class conv_series_decomp_multi(nn.Module):
    def __init__(self, dilation=1):
        super(conv_series_decomp_multi, self).__init__()
        # print('dilation', dilation)
        self.moving_avg = MultiConvBlocks(dilation)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        
        if isinstance(moving_avg, list):
            self.decomp1 = conv_series_decomp_multi(moving_avg)
            self.decomp2 = conv_series_decomp_multi(moving_avg)
        else:
            self.decomp1 = conv_series_decomp(moving_avg)
            self.decomp2 = conv_series_decomp(moving_avg)
            
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # print('encoder-layer:', x.shape)
        # print('------------------')
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        # print('self-att-1:', new_x.shape)
        # print('------------------')
        x = x + self.dropout(new_x)
        # print(x.shape)
        # print('------------------')
        x, _ = self.decomp1(x)
        # print('decomp1:', x.shape)
        # print('------------------')
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # print('feed-forward:', y.shape)
        # print('------------------')
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # print(y.shape)
        # print('------------------')
        res, _ = self.decomp2(x + y)
        # print('decomp2:', res.shape)
        # print('------------------')
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        # print('encoder:', x.shape)
        # print('------------------')
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
            # print(x.shape)
            # print('------------------')
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        
        if isinstance(moving_avg, list):
            self.decomp1 = conv_series_decomp_multi(moving_avg)
            self.decomp2 = conv_series_decomp_multi(moving_avg)
            self.decomp3 = conv_series_decomp_multi(moving_avg)
        else:
            self.decomp1 = conv_series_decomp(moving_avg)
            self.decomp2 = conv_series_decomp(moving_avg)
            self.decomp3 = conv_series_decomp(moving_avg)
        
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        # print('self-att-2:', x.shape)
        # print('-------------------------')
        x, trend1 = self.decomp1(x)
        # print('decomp1:', x.shape)
        # print('-------------------------')
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        # print('cross-att:', x.shape)
        # print('-------------------------')
        x, trend2 = self.decomp2(x)
        # print('decomp2:', x.shape)
        # print('-------------------------')
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # print('feed-forward:', y.shape)
        # print('-------------------------')
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # print(y.shape)
        # print('-------------------------')
        x, trend3 = self.decomp3(x + y)
        # print('decomp3:', x.shape)
        # print('-------------------------')

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        # print('decoder_layer', residual_trend.shape, x.shape)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            # print('trend', trend.shape)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        
        # print('decoder-end', trend.shape, x.shape)
        return x, trend