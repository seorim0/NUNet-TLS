"""
Baseline model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import ConvSTFT, ConviSTFT
from config import WIN_LEN, HOP_LEN, FFT_LEN


# causal convolution
class causalConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=1, dilation=1, groups=1):
        super(causalConv2d, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=(padding[0], 0),
                              dilation=dilation, groups=groups)
        self.padding = padding[1]

    def forward(self, x):
        x = F.pad(x, [self.padding, 0, 0, 0])
        out = self.conv(x)
        return out


# convolution block
class CONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CONV, self).__init__()
        self.conv = causalConv2d(in_ch, out_ch, kernel_size=(3, 2), stride=(2, 1), padding=(1, 1))
        self.ln = nn.GroupNorm(1, out_ch, eps=1e-8)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.ln(self.conv(x)))


# convolution block for input layer
class INCONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(INCONV, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.ln = nn.GroupNorm(1, out_ch, eps=1e-8)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.ln(self.conv(x)))


# sub-pixel convolution block
class SPCONV(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(SPCONV, self).__init__()
        self.conv = causalConv2d(in_ch, out_ch * scale_factor, kernel_size=(3, 2), padding=(1, 1))
        self.ln = nn.GroupNorm(1, out_ch, eps=1e-8)
        self.prelu = nn.PReLU()

        self.n = scale_factor

    def forward(self, x):
        x = self.conv(x)  # [B, C, F, T]

        x = x.permute(0, 3, 2, 1)  # [B, T, F, C]
        r = torch.reshape(x, (x.size(0), x.size(1), x.size(2), x.size(3) // self.n, self.n))  # [B, T, F, C//2 , 2]
        r = r.permute(0, 1, 2, 4, 3)  # [B, T, F, 2, C//2]
        r = torch.reshape(r, (x.size(0), x.size(1), x.size(2) * self.n, x.size(3) // self.n))  # [B, T, F*2, C//2]
        r = r.permute(0, 3, 2, 1)  # [B, C, F, T]

        out = self.ln(r)
        out = self.prelu(out)
        return out


# 1x1 conv for down-sampling
class down_sampling(nn.Module):
    def __init__(self, in_ch):
        super(down_sampling, self).__init__()
        self.down_sampling = nn.Conv2d(in_ch, in_ch, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))

    def forward(self, x):
        return self.down_sampling(x)


# 1x1 conv for up-sampling
class upsampling(nn.Module):
    def __init__(self, in_ch):
        super(upsampling, self).__init__()
        self.upsampling = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=(3, 1), stride=(2, 1),
                                             padding=(1, 0), output_padding=(1, 0))

    def forward(self, x):
        out = self.upsampling(x)
        return out


# dilated dense block
class dilatedDenseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_layers):
        super(dilatedDenseBlock, self).__init__()

        self.input_layer = causalConv2d(in_ch, in_ch // 2, kernel_size=(3, 2), padding=(1, 1))  # channel half
        self.prelu1 = nn.PReLU()

        # dilated dense layer
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.caus_padd = ((2 ** i) // 2) * 2
            if i == 0: self.caus_padd = 1

            self.layers.append(nn.Sequential(
                # depth-wise separable conv
                causalConv2d(in_ch // 2 + i * in_ch // 2, in_ch // 2, kernel_size=(3, 2),
                             padding=(2 ** i, self.caus_padd), dilation=2 ** i, groups=in_ch // 2),
                # depth-wise
                nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=1),  # pointwise
                nn.GroupNorm(1, in_ch // 2, eps=1e-8),
                nn.PReLU()
            ))

        self.out_layer = causalConv2d(in_ch // 2, out_ch, kernel_size=(3, 2), padding=(1, 1))  # channel revert
        self.prelu2 = nn.PReLU()

    def forward(self, x):
        x = self.input_layer(x)  # C: in_ch//2
        x = self.prelu1(x)

        out1 = self.layers[0](x)

        # out2 = self.layers[1](torch.cat([out1, x], dim=1))
        out2 = torch.cat([out1, x], dim=1)  # C: in_ch//2 * 2
        out2 = self.layers[1](out2)

        # out3 = self.layers[2](torch.cat([out2, out1, x], dim=1))
        out3 = torch.cat([out2, out1], dim=1)
        out3 = torch.cat([out3, x], dim=1)  # C: in_ch//2 * 3
        out3 = self.layers[2](out3)

        # out4 = self.layers[3](torch.cat([out3, out2, out1, x], dim=1))
        out4 = torch.cat([out3, out2], dim=1)  # C: in_ch//2 * 4
        out4 = torch.cat([out4, out1], dim=1)
        out4 = torch.cat([out4, x], dim=1)
        out4 = self.layers[3](out4)

        # out5 = self.layers[4](torch.cat([out4, out3, out2, out1, x], dim=1))
        out5 = torch.cat([out4, out3], dim=1)  # C: in_ch//2 * 5
        out5 = torch.cat([out5, out2], dim=1)
        out5 = torch.cat([out5, out1], dim=1)
        out5 = torch.cat([out5, x], dim=1)
        out5 = self.layers[4](out5)

        # out = self.layers[5](torch.cat([out5, out4, out3, out2, out1, x], dim=1))
        out = torch.cat([out5, out4], dim=1)  # C: in_ch//2 * 6
        out = torch.cat([out, out3], dim=1)
        out = torch.cat([out, out2], dim=1)
        out = torch.cat([out, out1], dim=1)
        out = torch.cat([out, x], dim=1)
        out = self.layers[5](out)

        out = self.out_layer(out)
        out = self.prelu2(out)

        return out


# Multi-Scale Feature Extraction (MSFE) - 6
class MSFE6(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(MSFE6, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)
        self.en2 = CONV(mid_ch, mid_ch)
        self.en3 = CONV(mid_ch, mid_ch)
        self.en4 = CONV(mid_ch, mid_ch)
        self.en5 = CONV(mid_ch, mid_ch)
        self.en6 = CONV(mid_ch, mid_ch)

        # bottleneck
        self.ddense = dilatedDenseBlock(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, mid_ch)
        self.de4 = SPCONV(mid_ch * 2, mid_ch)
        self.de5 = SPCONV(mid_ch * 2, mid_ch)
        self.de6 = SPCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)
        out2 = self.en2(out1)
        out3 = self.en3(out2)
        out4 = self.en4(out3)
        out5 = self.en5(out4)
        out6 = self.en6(out5)

        # bottleneck
        out = self.ddense(out6)

        # decoder
        out = self.de1(torch.cat([out, out6], dim=1))
        out = self.de2(torch.cat([out, out5], dim=1))
        out = self.de3(torch.cat([out, out4], dim=1))
        out = self.de4(torch.cat([out, out3], dim=1))
        out = self.de5(torch.cat([out, out2], dim=1))
        out = self.de6(torch.cat([out, out1], dim=1))

        out += x
        return out


# Multi-Scale Feature Extraction (MSFE) - 5
class MSFE5(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(MSFE5, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)
        self.en2 = CONV(mid_ch, mid_ch)
        self.en3 = CONV(mid_ch, mid_ch)
        self.en4 = CONV(mid_ch, mid_ch)
        self.en5 = CONV(mid_ch, mid_ch)

        # bottleneck
        self.ddense = dilatedDenseBlock(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, mid_ch)
        self.de4 = SPCONV(mid_ch * 2, mid_ch)
        self.de5 = SPCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)
        out2 = self.en2(out1)
        out3 = self.en3(out2)
        out4 = self.en4(out3)
        out5 = self.en5(out4)

        # bottleneck
        out = self.ddense(out5)

        # decoder
        out = self.de1(torch.cat([out, out5], dim=1))
        out = self.de2(torch.cat([out, out4], dim=1))
        out = self.de3(torch.cat([out, out3], dim=1))
        out = self.de4(torch.cat([out, out2], dim=1))
        out = self.de5(torch.cat([out, out1], dim=1))

        out += x
        return out


# Multi-Scale Feature Extraction (MSFE) - 4
class MSFE4(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(MSFE4, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)
        self.en2 = CONV(mid_ch, mid_ch)
        self.en3 = CONV(mid_ch, mid_ch)
        self.en4 = CONV(mid_ch, mid_ch)

        # bottleneck
        self.ddense = dilatedDenseBlock(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, mid_ch)
        self.de4 = SPCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)
        out2 = self.en2(out1)
        out3 = self.en3(out2)
        out4 = self.en4(out3)

        # bottleneck
        out = self.ddense(out4)

        # decoder
        out = self.de1(torch.cat([out, out4], dim=1))
        out = self.de2(torch.cat([out, out3], dim=1))
        out = self.de3(torch.cat([out, out2], dim=1))
        out = self.de4(torch.cat([out, out1], dim=1))

        out += x
        return out


# Multi-Scale Feature Extraction (MSFE) - 3
class MSFE3(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(MSFE3, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)
        self.en2 = CONV(mid_ch, mid_ch)
        self.en3 = CONV(mid_ch, mid_ch)

        # bottleneck
        self.ddense = dilatedDenseBlock(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)
        out2 = self.en2(out1)
        out3 = self.en3(out2)

        # bottleneck
        out = self.ddense(out3)

        # decoder
        out = self.de1(torch.cat([out, out3], dim=1))
        out = self.de2(torch.cat([out, out2], dim=1))
        out = self.de3(torch.cat([out, out1], dim=1))

        out += x
        return out


# Baseline network for NUNet-TLS
class Baseline(nn.Module):

    def __init__(self, in_ch=1, mid_ch=32, out_ch=64):
        super(Baseline, self).__init__()

        # input layer
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.encoder_stage1 = nn.Sequential(
            MSFE6(out_ch, mid_ch, out_ch),
            down_sampling(out_ch)
        )

        self.encoder_stage2 = nn.Sequential(
            MSFE5(out_ch, mid_ch, out_ch),
            down_sampling(out_ch)
        )

        self.encoder_stage3 = nn.Sequential(
            MSFE4(out_ch, mid_ch, out_ch),
            down_sampling(out_ch)
        )

        self.encoder_stage4 = nn.Sequential(
            MSFE4(out_ch, mid_ch, out_ch),
            down_sampling(out_ch)
        )

        self.encoder_stage5 = nn.Sequential(
            MSFE4(out_ch, mid_ch, out_ch),
            down_sampling(out_ch)
        )

        self.encoder_stage6 = nn.Sequential(
            MSFE3(out_ch, mid_ch, out_ch),
            down_sampling(out_ch)
        )

        # Bottleneck block
        self.DDense = nn.Sequential(
            dilatedDenseBlock(out_ch, out_ch, 6)
        )

        # decoder
        self.decoder_stage1 = nn.Sequential(
            upsampling(out_ch * 2),
            MSFE3(out_ch * 2, mid_ch, out_ch)
        )

        self.decoder_stage2 = nn.Sequential(
            upsampling(out_ch * 2),
            MSFE4(out_ch * 2, mid_ch, out_ch)
        )

        self.decoder_stage3 = nn.Sequential(
            upsampling(out_ch * 2),
            MSFE4(out_ch * 2, mid_ch, out_ch)
        )

        self.decoder_stage4 = nn.Sequential(
            upsampling(out_ch * 2),
            MSFE4(out_ch * 2, mid_ch, out_ch)
        )

        self.decoder_stage5 = nn.Sequential(
            upsampling(out_ch * 2),
            MSFE5(out_ch * 2, mid_ch, out_ch)
        )

        self.decoder_stage6 = nn.Sequential(
            upsampling(out_ch * 2),
            MSFE6(out_ch * 2, mid_ch, out_ch)
        )

        # output layer
        self.output_layer = nn.Conv2d(out_ch, in_ch, kernel_size=1)

        # for feature extract
        self.stft = ConvSTFT(WIN_LEN, HOP_LEN, FFT_LEN, feature_type='real')
        self.istft = ConviSTFT(WIN_LEN, HOP_LEN, FFT_LEN, feature_type='real')

    def forward(self, x):
        # STFT
        mags, phase = self.stft(x)  # [B, F, T]
        hx = mags.unsqueeze(1)  # [B, 1, F, T]
        hx = hx[:, :, 1:]

        # input layer
        hx = self.input_layer(hx)

        # encoder stage 1
        hx1 = self.encoder_stage1(hx)

        # encoder stage 2
        hx2 = self.encoder_stage2(hx1)

        # encoder stage 3
        hx3 = self.encoder_stage3(hx2)

        # encoder stage 4
        hx4 = self.encoder_stage4(hx3)

        # encoder stage 5
        hx5 = self.encoder_stage5(hx4)

        # encoder stage 6
        hx6 = self.encoder_stage6(hx5)

        # dilated dense block
        out = self.DDense(hx6)

        # decoder stage 1
        out = self.decoder_stage1(torch.cat([out, hx6], dim=1))

        # decoder stage 2
        out = self.decoder_stage2(torch.cat([out, hx5], dim=1))

        # decoder stage 3
        out = self.decoder_stage3(torch.cat([out, hx4], dim=1))

        # decoder stage 4
        out = self.decoder_stage4(torch.cat([out, hx3], dim=1))

        # decoder stage 5
        out = self.decoder_stage5(torch.cat([out, hx2], dim=1))

        # decoder stage 6
        out = self.decoder_stage6(torch.cat([out, hx1], dim=1))

        # output layer
        out = self.output_layer(out)

        out = F.pad(out, [0, 0, 1, 0])

        # ISTFT
        out_wav = self.istft(out.squeeze(1), phase).squeeze(1)
        out_wav = torch.clamp_(out_wav, -1, 1)  # clipping [-1, 1]
        return out_wav

    def loss(self, enhanced, target):
        return F.mse_loss(enhanced, target, reduction='mean')


