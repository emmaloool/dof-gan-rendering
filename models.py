import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from arch_utils import AdaIN, MLP, conv, deconv


#################################################
#               Generators G_I, G_D             #
#################################################

class ARGenerator(nn.Module):
    
    def __init__(self, noise_size=128, conv_dim=64):
        super(ARGenerator, self).__init__()

        # ----- Weights are shared between both generators for initial layers -----
        self.init_adain = AdaIN(1024, 128)

        self.deconv_1 = deconv(1024, 512, kernel_size=4, norm='none')
        self.adain_1 = AdaIN(512, 128)

        self.deconv_2 = deconv(512, 256, kernel_size=4, norm='none')
        self.adain_2 = AdaIN(256, 128)

        self.deconv_3 = deconv(256, 128, kernel_size=4, norm='none')
        self.adain_3 = AdaIN(128, 128)

        self.deconv_4 = deconv(128, 64, kernel_size=4, norm='adain') 
        self.adain_4 = AdaIN(64, 128)

        # ----- Image generator G_I -----
        self.conv_I = conv(64, 3, kernel_size=4, stride=2, padding=33, norm='none')

        # ----- Depth generator G_D -----
        self.conv_D = conv(64, 1, kernel_size=4, stride=2, padding=33, norm='none')
        self.mlp    = MLP(noise_size, 1)


    # z: random noise (Gaussian-distributed [0,1], dimension 128x1)
    def forward(self, c, z):
        """ Generates a deep DoF image I_d and depth map D 
            from input constant (learned) c and random noise z

            Input
            -----
                z: 128  x 1
                c: 1024 x 4 x 4

            Output
            ------
                I: BS x  3 x 64 x 64
                D: BS x 1 x 64 x 64
        """
        # print("--------------- GENERATOR ---------------")
        # print("Input c shape: ", c.shape)
        # output = self.init_adain(c, z)
        # output = F.relu(output)
        # print("AFTER ADAIN: ", output.shape)
        output = c

        output = self.deconv_1(output)
        output = F.relu(output)
        # output = self.adain_1(output, z)
        # print("AFTER DECONV1: ", output.shape)


        output = self.deconv_2(output)
        output = F.relu(output)
        # print("AFTER DECONV2: ", output.shape)


        output = self.deconv_3(output)
        output = F.relu(output)
        # print("AFTER DECONV3: ", output.shape)


        output = self.deconv_4(output)
        output = F.relu(output)
        # print("AFTER DECONV4: ", output.shape)

        # print("------- I_g -------")
        # Seperably handle I,D
        I = self.conv_I(output)
        I = F.tanh(I)

        # print("I_g: ", I.shape)

        # print("------- D_g -------")
        D = self.conv_D(output)
        D = F.tanh(D)
        # print("After conv_D: ", D.shape)

        img_size = D.shape[0] #TODO: print shape & use correct index for 64
        # print("img_size = ", img_size)
        mlp_z = self.mlp.forward(z).view(16, 1, 1, 1)
        # print("mlp_z: ", mlp_z.shape)
         
        D = torch.mul(mlp_z, 10*D)
        # print("D shape: ", D.shape)
        # print("???: ", torch.mul(mlp_z, 10*D).shape)
        # print("D_g: ", D.shape)

        return I, D


##################################################
#            Depth Expansion Network T           #
##################################################
class DepthExpansionNetwork(nn.Module):

    def __init__(self):
        super(DepthExpansionNetwork, self).__init__()

        self.conv_1 = conv(25, 25, kernel_size=3, stride=1, padding=1, norm='instance')
        self.conv_2 = conv(25, 25, kernel_size=3, stride=1, padding=1, norm='instance')
        self.conv_3 = conv(25, 25, kernel_size=3, stride=1, padding=1, norm='instance')

    def forward(self, x):
        # print("---------- DepthExpansionNetwork -----------")
        # print("x: ", x.shape)
        output = self.conv_1(x)
        output = F.leaky_relu(output)
        # print("After conv_1: ", output.shape)
        
        output = self.conv_2(x)
        output = F.leaky_relu(output)
        # print("After conv_2: ", output.shape)

        output = self.conv_3(x)
        output = F.leaky_relu(output)
        # print("After conv_3: ", output.shape)

        return output



##################################################
#   Discriminator C for DoF Mixture Learning     #
##################################################
class ARDiscriminator(nn.Module):

    def __init__(self, conv_dim=64):
        super(ARDiscriminator, self).__init__()
        self.conv1 = conv(3, 64, kernel_size=5, stride=3, padding=17, norm='none')
        self.conv2 = conv(64, 128,  kernel_size=5, stride=3, padding=9, norm='instance', spectral=True)
        self.conv3 = conv(128, 256, kernel_size=5, stride=3, padding=5, norm='instance', spectral=True)
        self.conv4 = conv(256, 512, kernel_size=5, stride=3, padding=3, norm='instance', spectral=True)
        # self.linear = nn.Linear(512*4*4, 1)
        self.conv5 = conv(512, 1, kernel_size=4, stride=2, padding=0, norm=None)

    def forward(self, x):
        """Outputs the discriminator score given a deep DoF image x
            
                Input
                -----
                    x: BS x 3 x 64 x 64
            
                Output
                ------
                    out: BS x 1 x 1 x 1
        """
        # print("--------------- DISCRIMINATOR ---------------")
        # print("INPUT SHAPE: ", x.shape)
        output = self.conv1(x)
        output = F.leaky_relu(output)
        # print("AFTER CONV1: ", output.shape)

        output = self.conv2(output)
        output = F.leaky_relu(output)
        # print("AFTER CONV2: ", output.shape)

        output = self.conv3(output)
        output = F.leaky_relu(output)
        # print("AFTER CONV3: ", output.shape)

        output = self.conv4(output)
        output = F.leaky_relu(output)
        # print("AFTER CONV4: ", output.shape)

        # Apply linear layer to get logit
        # output = torch.flatten(output)
        # print("AFTER FLATTEN: ", output.shape)
        # output = self.linear(output)
        output = self.conv5(output)
        # print("AFTER 'LINEAR' (CONV5): ", output.shape)

        return output

