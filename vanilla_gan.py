# CMU 16-726 Learning-Based Image Synthesis / Spring 2022, Assignment 3
#
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip
# This is the main training file for the first part of the assignment.
#
# Usage:
# ======
#    To train with the default hyperparamters (saves results to checkpoints_vanilla/ and samples_vanilla/):
#       python vanilla_gan.py

import argparse
from ast import Break
import os
import warnings

import imageio

warnings.filterwarnings("ignore")

# Numpy & Scipy imports
import numpy as np

# Torch imports
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Local imports
import utils
from data_loader import get_data_loader
from models import ARGenerator, ARDiscriminator, DepthExpansionNetwork

from diff_augment import DiffAugment
policy = 'color,translation,cutout'

from renderer import warp_depthmap, Render

import time

SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def print_models(G, D):
    """Prints model information for the generators and discriminators.
    """
    print("                    G                  ")
    print("---------------------------------------")
    print(G)
    print("---------------------------------------")

    print("                    D                  ")
    print("---------------------------------------")
    print(D)
    print("---------------------------------------")


def create_model(opts):
    """Builds the generators and discriminators.
    """
    G = ARGenerator(noise_size=opts.noise_size, conv_dim=opts.conv_dim)
    D = ARDiscriminator(conv_dim=opts.conv_dim)
    T = DepthExpansionNetwork()

    # print_models(G, D)

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
        print('Models moved to GPU.')

    return G, D, T


def create_image_grid(array, ncols=None):
    """
    """
    num_images, channels, cell_h, cell_w = array.shape

    if not ncols:
        ncols = int(np.sqrt(num_images))
    nrows = int(np.math.floor(num_images / float(ncols)))
    result = np.zeros((cell_h*nrows, cell_w*ncols, channels), dtype=array.dtype)
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w, :] = array[i*ncols+j].transpose(1, 2, 0)

    if channels == 1:
        result = result.squeeze()
    return result


def checkpoint(iteration, G, D, opts):
    """Saves the parameters of the generator G and discriminator D.
    """
    G_path = os.path.join(opts.checkpoint_dir, 'G_iter%d.pkl' % iteration)
    D_path = os.path.join(opts.checkpoint_dir, 'D_iter%d.pkl' % iteration)
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)


def save_samples(G, fixed_noise_c, fixed_noise_z, iteration, opts):
    generated_images, _ = G(fixed_noise_c, fixed_noise_z)
    generated_images = utils.to_data(generated_images)

    grid = create_image_grid(generated_images)

    # merged = merge_images(X, fake_Y, opts)
    path = os.path.join(opts.sample_dir, 'sample-{:06d}.png'.format(iteration))
    imageio.imwrite(path, grid)
    print('Saved {}'.format(path))


def save_images(images, iteration, opts, name):
    grid = create_image_grid(utils.to_data(images))

    path = os.path.join(opts.sample_dir, '{:s}-{:06d}.png'.format(name, iteration))
    imageio.imwrite(path, grid)
    print('Saved {}'.format(path))


def sample_noise(dim):
    """
    Generate a PyTorch Variable of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Variable of shape (batch_size, dim, 1, 1) containing uniform
      random noise in the range (-1, 1).
    """
    return utils.to_var(torch.rand(batch_size, dim) * 2 - 1).unsqueeze(2).unsqueeze(3)


def training_loop(train_dataloader, opts):
    """Runs the training loop.
        * Saves checkpoints every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """

    # Create generators and discriminators
    G, D, T = create_model(opts)

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(G.parameters(), opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(D.parameters(), opts.lr, [opts.beta1, opts.beta2])

    # Generate fixed noise for sampling from the generator
    # fixed_noise = sample_noise(opts.noise_size)  # batch_size x noise_size x 1 x 1
    fixed_noise_z = sample_noise(opts.noise_size)
    fixed_noise_c = sample_noise(1024*4*4).view(16,1024,4,4)

    iteration = 1

    total_train_iters = opts.num_epochs * len(train_dataloader)


    for epoch in range(opts.num_epochs):

        for batch in train_dataloader:

            real_images, labels = batch
            real_images, labels = utils.to_var(real_images), utils.to_var(labels).long().squeeze()

            if opts.use_diffaug: 
                real_images = DiffAugment(real_images, policy)

            ################################################
            ###         TRAIN THE DISCRIMINATOR         ####
            ################################################

            # FILL THIS IN
            # 1. Compute the discriminator loss on real images
            D_real_loss = torch.mean((D(real_images) - 1)**2)
            
            # 2. Sample noise
            noise_z = sample_noise(opts.noise_size)
            noise_c = sample_noise(1024*4*4).view(16,1024,4,4)
            # print("noise_z: ", noise_z.shape)
            # print("noise_c: ", noise_c.shape)
            
            # 3. Generate fake images from the noise
            I_g, D_g  = G.forward(noise_c, noise_z)

            if opts.use_diffaug:        # ----- Apply diff aug on fake images -----
                I_g = DiffAugment(I_g, policy)

            # 4. Compute the discriminator loss on the fake images
            D_fake_loss = torch.mean((D(I_g.detach())) ** 2)

            D_total_loss = torch.add(torch.div(D_real_loss, 2), torch.div(D_fake_loss, 2))

            # update the discriminator D
            d_optimizer.zero_grad()
            D_total_loss.backward()
            d_optimizer.step()


            ###########################################
            ###          TRAIN THE GENERATOR        ###
            ###########################################

            # FILL THIS IN
            # 1. Sample noise
            I_g, D_g  = G.forward(noise_c, noise_z)

            # 2. Generate fake images from the noise
            '''
                NOTE: Instead of using the generated deep image I_g directly as the fake image,
                we will implement DoF mixture learning, which has us generate a shallow image
                by rendering over deep I_g and D_g scaled by some random s in [0,1] (depth scaled to some extent)
            ''' 

            s = torch.randn(1).uniform_(0,1)    # Sample s from Bernoulli distribution in [0,1]
            I_g, D_g  = G.forward(noise_c, noise_z)
            if opts.use_diffaug:        # ----- Apply diff aug on fake images -----
                I_g = DiffAugment(I_g, policy)

            G_loss = torch.mean((D(I_g) - 1) ** 2)
            
            # ------------------- DoF mixture learning -------------------
            # Generate shallow DoF image from deep DoF image I_g and depth D_g scaled by s
            s = torch.bernoulli(torch.randn(1).uniform_(0,1)).item()    # Sample s from Bernoulli distribution in [0,1]
            print("Iteration " + str(iteration) + ": Rendering........ ")
            start = time.time()
            D_warp = warp_depthmap(s * D_g)
            print("done warping depth map")
            M = T(D_warp)
            I_s = Render(M, I_g)
            print("done rendering")
            end = time.time()
            if iteration % opts.sample_every == 0:
                render_time = int(end-start)
                print("Render time: ", str(render_time/60) + "m" + str(render_time%60) + "s")

            # 3. Compute the generator loss
            G_loss = torch.mean((D(I_s) - 1) ** 2)
            # ---------------------------------------------------------
            
            # update the generator G
            g_optimizer.zero_grad()
            G_loss.backward()
            g_optimizer.step()


            # Print the log info
            if iteration % opts.log_step == 0:
                print('Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss: {:6.4f}'.format(
                       iteration, total_train_iters, D_real_loss.item(), D_fake_loss.item(), G_loss.item()))
                logger.add_scalar('D/fake', D_fake_loss, iteration)
                logger.add_scalar('D/real', D_real_loss, iteration)
                logger.add_scalar('D/total', D_total_loss, iteration)
                logger.add_scalar('G/total', G_loss, iteration)

            # Save the generated samples
            if iteration % opts.sample_every == 0:
                print("Saving...")
                save_samples(G, fixed_noise_c, fixed_noise_z, iteration, opts)
                save_images(real_images, iteration, opts, 'real')

            # Save the model parameters
            if iteration % opts.checkpoint_every == 0:
                checkpoint(iteration, G, D, opts)

            iteration += 1


def main(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create a dataloader for the training images
    dataloader = get_data_loader(opts.data, opts)

    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    training_loop(dataloader, opts)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=64, help='The side length N to convert images to NxN.')
    parser.add_argument('--conv_dim', type=int, default=32)
    parser.add_argument('--noise_size', type=int, default=128)
    parser.add_argument('--use_diffaug', action='store_true', default=False, help='Choose whether to perform differentiable augmentation.')


    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate (default 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Data sources
    parser.add_argument('--data', type=str, default='cat/grumpifyBprocessed', help='The folder of the training dataset.')
    parser.add_argument('--data_preprocess', type=str, default='deluxe', help='data preprocess scheme [basic|deluxe]')
    parser.add_argument('--ext', type=str, default='*.png', help='Choose the file type of images to generate.')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_vanilla')
    parser.add_argument('--sample_dir', type=str, default='./vanilla')
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_every', type=int , default=200)
    parser.add_argument('--checkpoint_every', type=int , default=400)

    return parser


if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()

    batch_size = opts.batch_size
    opts.sample_dir = os.path.join('output/', opts.sample_dir,
                                   '%s_%s' % (os.path.basename(opts.data), opts.data_preprocess))
    
    if opts.use_diffaug:
        opts.sample_dir += '_diffaug'

    if os.path.exists(opts.sample_dir):
        cmd = 'rm %s/*' % opts.sample_dir
        os.system(cmd)
    logger = SummaryWriter(opts.sample_dir)
    print(opts)
    main(opts)
