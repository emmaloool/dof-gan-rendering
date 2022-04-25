import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.nn.utils as utils

from final_project.arch_utils import AdaIN, MLP, conv, deconv


#################################################
#               Generators G_I, G_D             #
#################################################

class ARGenerator(nn.Module):
    
    def __init__(self, conv_dim=64):
        super(ARGenerator, self).__init__()

        # ----- Weights are shared between both generators for initial layers -----
        self.deconv_1 = deconv(1024, 512, kernel_size=4, norm='none')
        self.deconv_2 = deconv(512, 256, kernel_size=4, norm='none')
        self.deconv_3 = deconv(256, 128, kernel_size=4, norm='none')
        self.deconv_4 = deconv(128, 64, kernel_size=4, norm='none') 

        # ----- Image generator G_I -----
        self.conv_I = conv(64, 3, kernel_size=4, norm='none')

        # ----- Depth generator G_D -----
        self.conv_D = conv(64, 1, kernel_size=4, norm='none')
        self.mlp    = MLP(64*64, 64*64*25)


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
                D: BS x 25 x 64 x 64
        """

        output = AdaIN(c,z)
        output = F.relu(output)

        output = self.deconv_1(output)
        output = F.relu(AdaIN(output, z))

        output = self.deconv_2(output)
        output = F.relu(AdaIN(output, z))

        output = self.deconv_3(output)
        output = F.relu(AdaIN(output, z))

        output = self.deconv_4(output)
        output = F.relu(AdaIN(output, z))

        # Seperably handle I,D
        I = self.conv_I(output)
        I = F.tanh(I)

        D = self.conv_D(output)
        D = 10 * F.tanh(D) 

        # Flatten D (64x64x1) -> (64*64*1), apply MLP(in=64*64*1,out=64*64*25), then reshape into (64x64x25)
        D = torch.flatten(D)
        D = self.mlp.forward(D)
        img_size = D.shape[0]       # TODO: print shape & use correct index for 64
        D = D.view(img_size, img_size, 25)

        return I, D


##################################################
#   Discriminator C for DoF Mixture Learning     #
##################################################
class ARDiscriminator(nn.Module):

    def __init__(self, conv_dim=64):
        super(ARDiscriminator, self).__init__()

        self.conv1 = conv(3, 64, kernel_size=5, stride=3, padding=17, norm='none')
        self.conv2 = utils.spectral_norm(conv(64, 128,  kernel_size=5, stride=3, padding=17, norm='instance'))
        self.conv3 = utils.spectral_norm(conv(128, 256, kernel_size=5, stride=3, padding=17, norm='instance'))
        self.conv4 = utils.spectral_norm(conv(256, 512, kernel_size=5, stride=3, padding=17, norm='instance'))
        self.linear = nn.Linear(512, 1)


    def forward(self, x):
        """Outputs the discriminator score given a deep DoF image x
            
                Input
                -----
                    x: BS x 3 x 64 x 64
            
                Output
                ------
                    out: BS x 1 x 1 x 1
        """
        output = self.conv1(x)
        output = F.leaky_relu(output)

        output = self.conv2(x)
        output = F.leaky_relu(output)

        output = self.conv3(x)
        output = F.leaky_relu(output)

        output = self.conv4(x)
        output = F.leaky_relu(output)

        # Apply linear layer to get logit
        output = torch.flatten(output)
        output = self.linear(output)

        return output

