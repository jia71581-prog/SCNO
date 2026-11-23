"""Upstream architecture; spectral buffer dtype follows input for FP64 FWI audit."""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class SpectralConv3d_UNO(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            D1,
            D2,
            D3,
            modes1=None,
            modes2=None,
            modes3=None):
        super(SpectralConv3d_UNO, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        D1, D2, D3 are output dimensions (x,y,t)
        modes1,modes2,modes3 = Number of fourier coefficinets to consider along each spectral dimesion
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d1 = D1
        self.d2 = D2
        self.d3 = D3
        if modes1 is not None:
            # Number of Fourier modes to multiply, at most floor(N/2) + 1
            self.modes1 = modes1
            self.modes2 = modes2
            self.modes3 = modes3
        else:
            self.modes1 = D1  # Will take the maximum number of possiblel modes for given output dimension
            self.modes2 = D2
            self.modes3 = D3 // 2 + 1

        self.scale = (1 / (2 * in_channels))**(1.0 / 2.0)
        self.weights1 = nn.Parameter(
            self.scale *
            torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale *
            torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale *
            torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale *
            torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x, D1=None, D2=None, D3=None):
        """
        D1,D2,D3 are the output dimensions (x,y,t)
        """
        if D1 is not None:
            self.d1 = D1
            self.d2 = D2
            self.d3 = D3

        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm = 'forward')

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            self.d1,
            self.d2,
            self.d3 // 2 + 1,
            dtype=x_ft.dtype,
            device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(self.d1, self.d2, self.d3) , norm = 'forward')
        return x

class pointwise_op_3D(nn.Module):
    def __init__(self, in_channel, out_channel, dim1, dim2, dim3):
        super(pointwise_op_3D, self).__init__()
        self.conv = nn.Conv3d(int(in_channel), int(out_channel), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.dim3 = int(dim3)

    def forward(self, x, dim1=None, dim2=None, dim3=None):
        """
        dim1,dim2,dim3 are the output dimensions (x,y,t)
        """
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
            dim3 = self.dim3
        x_out = self.conv(x)
        
        ft = torch.fft.rfftn(x_out,dim=[-3,-2,-1])
        ft_u = torch.zeros_like(ft)
        ft_u[:, :, :(dim1//2), :(dim2//2), :(dim3//2)] = ft[:, :, :(dim1//2), :(dim2//2), :(dim3//2)]
        ft_u[:, :, -(dim1//2):, :(dim2//2), :(dim3//2)] = ft[:, :, -(dim1//2):, :(dim2//2), :(dim3//2)]
        ft_u[:, :, :(dim1//2), -(dim2//2):, :(dim3//2)] = ft[:, :, :(dim1//2), -(dim2//2):, :(dim3//2)]
        ft_u[:, :, -(dim1//2):, -(dim2//2):, :(dim3//2)] = ft[:, :, -(dim1//2):, -(dim2//2):, :(dim3//2)]

        x_out = torch.fft.irfftn(ft_u, s=(dim1, dim2, dim3))

        
        x_out = torch.nn.functional.interpolate(x_out, size=(
            dim1, dim2, dim3), mode='trilinear', align_corners=True)
        return x_out

class OperatorBlock_3D(nn.Module,):
    """
    To turn to normalization set Normalize = True
    To have linear operator set Non_Lin = False
    """
    def __init__(self, in_channel, out_channel,res1, res2,res3,modes1,modes2,modes3, Normalize = True, Non_Lin = True, spectral_factory=SpectralConv3d_UNO):
        super(OperatorBlock_3D,self).__init__()
        self.conv = spectral_factory(in_channel, out_channel, res1,res2,res3,modes1,modes2,modes3)
        self.w = pointwise_op_3D(in_channel, out_channel, res1,res2,res3)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm3d(out_channel,affine=True)


    def forward(self,x, res1 = None, res2 = None, res3 = None):

        x1_out = self.conv(x,res1,res2,res3)
        x2_out = self.w(x,res1,res2,res3)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out

class Uno3D(nn.Module):
    def __init__(self, in_width, width,pad = 0, factor = 1, pad_both = False, spectral_factory=SpectralConv3d_UNO):
        super(Uno3D, self).__init__()

        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  # pad the domain if input is non-periodic
        self.pad_both = pad_both
        self.fc_n1 = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) # input channel is 3: (a(x, y), x, y)
        
        self.conv0 = OperatorBlock_3D(self.width, 2*factor*self.width, 48, 48, 96, 24, 24, 24, spectral_factory=spectral_factory)
        
        self.conv1 = OperatorBlock_3D(2*factor*self.width, 4*factor*self.width, 32, 32, 64, 16, 16, 16, spectral_factory=spectral_factory)
        
        self.conv2 = OperatorBlock_3D(4*factor*self.width, 8*factor*self.width, 16, 16, 32, 8, 8, 8, spectral_factory=spectral_factory)
        
        self.conv3 = OperatorBlock_3D(8*factor*self.width, 16*factor*self.width, 8, 8, 16, 4, 4, 4, spectral_factory=spectral_factory)
        
        self.conv4 = OperatorBlock_3D(16*factor*self.width, 16*factor*self.width, 8, 8, 16, 4, 4, 4, spectral_factory=spectral_factory)
        
        self.conv5 = OperatorBlock_3D(16*factor*self.width, 8*factor*self.width, 16, 16, 32, 4, 4, 4, spectral_factory=spectral_factory) 
        
        self.conv6 = OperatorBlock_3D(8*factor*self.width, 4*factor*self.width, 32, 32, 64, 8, 8, 8, spectral_factory=spectral_factory)
        
        self.conv7 = OperatorBlock_3D(8*factor*self.width, 2*factor*self.width, 48, 48, 96, 16, 16, 16, spectral_factory=spectral_factory)
        
        self.conv8 = OperatorBlock_3D(4*factor*self.width, 2*self.width, 64, 64, 128, 24, 24, 32, spectral_factory=spectral_factory) # will be reshaped

        self.fc1 = nn.Linear(3*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 2)
        
        #self.bn_fc_1 = torch.nn.BatchNorm3d(self.width)
        #self.bn_fc0 = torch.nn.InstanceNorm3d(self.width)
        #self.bn_fc1 = torch.nn.InstanceNorm3d(4*self.width)

    def forward(self, x, time_grid = 128):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x_fc = self.fc_n1(x)
        x_fc = F.gelu(x_fc)
        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)
        
        #x_fc0 = F.pad(x_fc0, [0,self.padding,0,0,0,0],mode ='constant')
        
        D1,D2,D3 = x_fc0.shape[-3],x_fc0.shape[-2],x_fc0.shape[-1]

        x_c0 = self.conv0(x_fc0)
        x_c1 = self.conv1(x_c0)
        x_c2 = self.conv2(x_c1)
        
        x_c3 = self.conv3(x_c2)
        x_c4 = self.conv4(x_c3)
        x_c5 = self.conv5(x_c4)
        
        x_c6 = self.conv6(x_c5)
        x_c6 = torch.cat([x_c6, torch.nn.functional.interpolate(x_c1, size = (x_c6.shape[2], x_c6.shape[3],x_c6.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
        
        x_c7 = self.conv7(x_c6)
        x_c7 = torch.cat([x_c7, torch.nn.functional.interpolate(x_c0, size = (x_c7.shape[2], x_c7.shape[3],x_c7.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
        
        x_c8 = self.conv8(x_c7,D1,D2,time_grid+self.padding)

        x_c8 = torch.cat([x_c8,torch.nn.functional.interpolate(x_fc0, size = (x_c8.shape[2], x_c8.shape[3],x_c8.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
        
        if self.padding!=0:
            if self.pad_both:
                x_c8 = x_c8[...,self.padding//2:-self.padding//2]
            else:
                x_c8 = x_c8[...,:-self.padding]

        x_c8 = x_c8.permute(0, 2, 3, 4, 1)

        if getattr(self, 'checkpoint_projection', False) and torch.is_grad_enabled():
            rows = getattr(self, 'projection_chunk_rows', 0)
            # Preserve the original CPU backward path. The CUDA tiled path is
            # checked against the full-width saved training checkpoint.
            if rows and x_c8.device.type == 'cuda':
                x_c8 = x_c8.contiguous()
                x_out = torch.cat([checkpoint(self.project_output, x_c8[:, first:first+rows], use_reentrant=False)
                    for first in range(0, x_c8.shape[1], rows)], dim=1)
            else:
                x_out = checkpoint(self.project_output, x_c8, use_reentrant=False)
        else:
            x_out = self.project_output(x_c8)
        
        return x_out

    def project_output(self, value):
        return self.fc2(F.gelu(self.fc1(value)))
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
