import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, angRes, upscale_factor):
        super(Net, self).__init__()
        n_groups, n_blocks, channels = 4, 4, 64
        self.angRes = angRes
        # Feature Extraction
        self.angFE = nn.Conv2d(1, channels, kernel_size=int(angRes), stride=int(angRes), padding=0, bias=False)
        self.spaFE = nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False)
        # Spatial-Angular Interaction
        self.interaction = Cascaded_InterGroups(angRes, n_groups, n_blocks, channels)
        # Fusion and Reconstruction
        self.bottleneck = BottleNeck(angRes, n_groups, channels)
        self.reconstruction = ReconBlock(angRes, channels, upscale_factor)

    def forward(self, x):
        x = SAI2MacPI(x, self.angRes)
        xa, xs = self.angFE(x), self.spaFE(x)
        buffer_a, buffer_s = self.interaction(xa, xs)
        buffer_out = self.bottleneck(buffer_a, buffer_s) + xs
        out = self.reconstruction(buffer_out)
        return out


class InterBlock(nn.Module):
    def __init__(self, angRes, channels):
        super(InterBlock, self).__init__()

        self.Spa2Ang = nn.Conv2d(channels, channels, kernel_size=int(angRes), stride=int(angRes), padding=0, bias=False)
        self.Ang2Spa = nn.Sequential(
            nn.Conv2d(channels, int(angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes),
        )
        self.AngConv = nn.Conv2d(2*channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.SpaConv = nn.Conv2d(2*channels, channels, kernel_size=3, stride=1, dilation=int(angRes),
                                            padding=int(angRes), bias=False)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, xa, xs):
        buffer_ang1 = xa
        buffer_ang2 = self.ReLU(self.Spa2Ang(xs))
        buffer_spa1 = xs
        buffer_spa2 = self.Ang2Spa(xa)
        buffer_a = torch.cat((buffer_ang1, buffer_ang2), 1)
        buffer_s = torch.cat((buffer_spa1, buffer_spa2), 1)
        out_a = self.ReLU(self.AngConv(buffer_a)) + xa
        out_s = self.ReLU(self.SpaConv(buffer_s)) + xs
        return out_a, out_s


class InterGroup(nn.Module):
    def __init__(self, angRes, n_block, channels):
        super(InterGroup, self).__init__()
        modules = []
        self.n_block = n_block
        for i in range(n_block):
            modules.append(InterBlock(angRes, channels))
        self.chained_blocks = nn.Sequential(*modules)

    def forward(self, xa, xs):
        buffer_a = xa
        buffer_s = xs
        for i in range(self.n_block):
            buffer_a, buffer_s = self.chained_blocks[i](buffer_a, buffer_s)
        out_a = buffer_a
        out_s = buffer_s
        return out_a, out_s


class Cascaded_InterGroups(nn.Module):
    def __init__(self, angRes, n_group, n_block, channels):
        super(Cascaded_InterGroups, self).__init__()
        self.n_group = n_group
        body = []
        for i in range(n_group):
            body.append(InterGroup(angRes, n_block, channels))
        self.body = nn.Sequential(*body)

    def forward(self, buffer_a, buffer_s):
        out_a = []
        out_s = []
        for i in range(self.n_group):
            buffer_a, buffer_s = self.body[i](buffer_a, buffer_s)
            out_a.append(buffer_a)
            out_s.append(buffer_s)
        return torch.cat(out_a, 1), torch.cat(out_s, 1)


class BottleNeck(nn.Module):
    def __init__(self, angRes, n_blocks, channels):
        super(BottleNeck, self).__init__()

        self.AngBottle = nn.Conv2d(n_blocks*channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.Ang2Spa = nn.Sequential(
            nn.Conv2d(channels, int(angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes),
        )
        self.SpaBottle = nn.Conv2d((n_blocks+1)*channels, channels, kernel_size=3, stride=1, dilation=int(angRes),
                                    padding=int(angRes), bias=False)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, xa, xs):
        xa = self.ReLU(self.AngBottle(xa))
        xs = torch.cat((xs, self.Ang2Spa(xa)), 1)
        out = self.ReLU(self.SpaBottle(xs))
        return out


class ReconBlock(nn.Module):
    def __init__(self, angRes, channels, upscale_factor):
        super(ReconBlock, self).__init__()
        self.PreConv = nn.Conv2d(channels, channels * upscale_factor ** 2, kernel_size=3, stride=1,
                                 dilation=int(angRes), padding=int(angRes), bias=False)
        self.PixelShuffle = nn.PixelShuffle(upscale_factor)
        self.FinalConv = nn.Conv2d(int(channels), 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.angRes = angRes

    def forward(self, x):
        buffer = self.PreConv(x)
        bufferSAI_LR = MacPI2SAI(buffer, self.angRes)
        bufferSAI_HR = self.PixelShuffle(bufferSAI_LR)
        out = self.FinalConv(bufferSAI_HR)
        return out


def MacPI2SAI(x, angRes):
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(x[:, :, i::angRes, j::angRes])
        out.append(torch.cat(out_h, 3))
    out = torch.cat(out, 2)
    return out


def SAI2MacPI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    out = torch.cat(tempU, dim=2)
    return out


if __name__ == "__main__":
    net = Net(5, 4).cuda()
    from thop import profile
    input = torch.randn(1, 1, 160, 160).cuda()
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (total / 1024**2))
    print('   Number of FLOPs: %.2fG' % (flops / 1024**3))