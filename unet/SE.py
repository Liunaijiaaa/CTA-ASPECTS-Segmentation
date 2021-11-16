import torch.nn as nn
import torch

class SEModule(nn.Module):

    def __init__(self, channels, reduction=16, act_layer=nn.ReLU, min_channels=8, reduction_channels=None,
                 gate_layer='sigmoid'):
        super(SEModule, self).__init__()
        reduction_channels = reduction_channels or max(channels // reduction, min_channels)
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, bias=True)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, bias=True)
        #self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        x_se = x_se.sigmoid()
        return x * x_se


'''
scSE模块与BAM模块类似，不过scSE模块只在语义分割中进行应用和测试，对语义分割准确率带来的提升比较大，可以让分割边界更加平滑。

cSE模块类似BAM模块里的Channel attention模块，具体流程如下:
1.将feature map通过global average pooling方法从[C, H, W]变为[C, 1, 1] 
2.然后使用两个1×1×1卷积进行信息的处理，最终得到C维的向量 
3.然后使用sigmoid函数进行归一化，得到对应的mask 
4.最后通过channel-wise相乘，得到经过信息校准过的feature map

sSE模块，空间注意力机制的实现，与BAM中的实现确实有很大不同，实现过程变得很简单，具体分析如下：
1.直接对feature map使用1×1×1卷积, 从[C, H, W]变为[1, H, W]的features
2.然后使用sigmoid进行激活得到spatial attention map
3.然后直接施加到原始feature map中，完成空间的信息校准

链接：https://zhuanlan.zhihu.com/p/102036086
'''
class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels,
                                      in_channels // 2,
                                      kernel_size=1,
                                      bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels // 2,
                                         in_channels,
                                         kernel_size=1,
                                         bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)  # shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z)  # shape: [bs, c/2, 1, 1]
        z = self.Conv_Excitation(z)  # shape: [bs, c, 1, 1]
        z = self.norm(z)
        return U * z.expand_as(U)

class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U) # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q # 广播机制


class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse


if __name__ == "__main__":
    bs, c, h, w = 10, 3, 64, 64
    in_tensor = torch.ones(bs, c, h, w)

    c_se = cSE(c)
    s_se = sSE(c)
    se = SEModule(c)
    sc_se = scSE(c)
    print("in shape:", in_tensor.shape)

    out_tensor1 = c_se(in_tensor)
    out_tensor2 = s_se(in_tensor)
    out_tensor3 = se(in_tensor)
    out_tensor4 = sc_se(in_tensor)

    print("out shape:", out_tensor1.shape)
    print("out shape:", out_tensor2.shape)
    print("out shape:", out_tensor3.shape)
    print("out shape:", out_tensor4.shape)
