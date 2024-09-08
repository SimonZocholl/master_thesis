# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
# sources
# dcgan: https://www.youtube.com/watch?v=IZtv9s_Wx9I
# wgan: https://www.youtube.com/watch?v=pG0QZ7OddX4
# cgan: https://www.youtube.com/watch?v=Hp-jWm2SzR8

# Read-through: Wasserstein GAN: https://www.alexirpan.com/2017/02/22/wasserstein-gan.html

# lstm classification: https://medium.com/@nutanbhogendrasharma/pytorch-recurrent-neural-networks-with-mnist-dataset-2195033b540f

# https://towardsdatascience.com/a-comprehensive-guide-to-neural-machine-translation-using-seq2sequence-modelling-using-pytorch-41c9b84ba350

# https://openreview.net/forum?id=rkgcXESxIH


# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

#from utils import gradient_penalty, save_checkpoint, load_checkpoint
#from model import Critic, Generator, initialize_weights


# %%
# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
IMG_SIZE = 64 #64

CHANNELS_IMG = 1
NUM_CLASSES = 10
GEN_EMBEDDING = 100
Z_DIM = 100
NUM_EPOCHS = 3
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10


# %%
class Critic(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(Critic, self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img+1, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )
        self.embed = nn.Embedding(num_classes, img_size*img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0],1,self.img_size,self.img_size)
        x = torch.cat([x,embedding], dim=1) # N x C x img_size (H) x img_size (W)
        return self.disc(x)


# %%
class Generator(nn.Module):
    def __init__(
            self, 
            channels_noise, 
            channels_img, 
            features_g,
            num_classes,
            img_size,
            embed_size
            ):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise+embed_size, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, labels):
        # latent vector z: N x noise_dim x 1 x 1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(2)
        x = torch.cat([x, embedding], dim=1)
        return self.net(x)


# %%
def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Critic(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Critic test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"

def gradient_penalty(critic, labels, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, labels)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]


    gradient = gradient.view(gradient.shape[0], -1)
    #gradient = gradient.reshape(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(state, filename="celeba_wgan_gp.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, gen, disc):
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint['gen'])
    disc.load_state_dict(checkpoint['disc'])

transformss = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

# %%
dataset = datasets.MNIST(root="dataset/", transform=transformss, download=True)
# comment mnist above and uncomment below for training on CelebA
# dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

(freal, flabels) = next(iter(loader))

# %%
dataset


# %%
class LSTM_Critic(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):

        super(LSTM_Critic, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(IMG_SIZE, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x, labels):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        # Passing in the input and hidden state into the model and  obtaining outputs
        x = x.reshape(BATCH_SIZE, IMG_SIZE, IMG_SIZE)
        out, hidden = self.lstm(x, (h0.detach(), c0.detach()))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        #Reshaping the outputs such that it can be fit into the fully connected layer
        
        out = self.fc(out[:, -1, :])
        return out
    
input_size = IMG_SIZE
hidden_size = 128
num_layers = 2
num_classes = 10
device = "cpu"

lstm_crit = LSTM_Critic(input_size, hidden_size, num_layers, num_classes)
cnt_params_lstm_crit = sum(p.numel() for p in lstm_crit.parameters() if p.requires_grad)

print("LSTM_Critic: ", cnt_params_lstm_crit)
print(lstm_crit)

critic_out = lstm_crit(freal, flabels)
print("critic_out.shape: ",critic_out.shape)


# %%
# need more params

class LSTM_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers) -> None:
        super(LSTM_Encoder, self).__init__()

        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        state = (torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device),
				 torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device))
        
        _, state = self.encoder(x, state)
        return state[0]



class LSTM_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers) -> None:
        super(LSTM_Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.decoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, IMG_SIZE)
        self.tanh = nn.Tanh()
        self.embedding = nn.Linear(IMG_SIZE, hidden_size)
    
    def forward(self, state):
        decoder_in = torch.ones()
        xs = []
        for _ in range(IMG_SIZE):
            output, state = self.decoder(decoder_in, state)
            x = self.tanh(self.linear(output))
            xs.append(x)

            decoder_in = self.embedding(x)

        x = torch.stack(xs, dim=2)
        return x


class LSTM_Generator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTM_Generator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Set initial hidden and cell states
        # expects:  N, L, Hin, returns: (L, N, D* Hout)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
    
    def forward(self, x, labels):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)        # Passing in the input and hidden state into the model and  obtaining output

        x = x.view(x.size(0),1, GEN_EMBEDDING).to(device)
        outs = []

        zeros = torch.zeros(x.size(0), 1,GEN_EMBEDDING).to(device)
        input = torch.cat([zeros, x], dim=2 ).to(device)
        
        for i in range(IMG_SIZE):
            out_lstm, (h0,c0) = self.lstm(input, (h0,c0))

            #print("out_lstm.shape: ", out_lstm.shape)
            out = self.tanh(self.fc1(out_lstm)).to(device)
            out = self.tanh(self.fc2(out)).to(device)
            #print("out.shape: ", out.shape)
            outs.append(out)
            
            input = torch.cat([out_lstm, x], dim=2).to(device)
        
        x = torch.stack(outs, dim=2).to(device)
        return x
        
noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).float()

lstm_noise = torch.randn(BATCH_SIZE, Z_DIM, 1).float()
encoder_input_size = 1
encoder_hidden_size = 128
encoder_num_layers = 1
lstm_encoder = LSTM_Encoder(encoder_input_size, encoder_hidden_size,encoder_num_layers)
encoder_out = lstm_encoder(lstm_noise)
encoder_out.shape

decoder_input_size = 1
decoder_hidden_size = 128
decoder_num_layers = 1
lstm_decoder = LSTM_Decoder(decoder_input_size, decoder_hidden_size,decoder_num_layers)
decoder_out = lstm_decoder(encoder_out)
decoder_out.shape

# %%
# cnt_params_crit:  216017
# cnt_params_gen:  1509801
# ratio gen/crit: 6.99

# LSTM_Critic: 213121
print("LSTM_Critic: ", cnt_params_lstm_crit)

# LSTM_Gen: 208064
print("LSTM_Gen: ", cnt_params_lstm_gen)

# ratio gen/crit: 0.98
print(f"ratio gen/crit: {(cnt_params_lstm_gen/ cnt_params_lstm_crit):.2f}")

# %%
# initialize gen and disc, note: critic should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMG_SIZE, GEN_EMBEDDING).to(device)
critic = Critic(CHANNELS_IMG, FEATURES_CRITIC, NUM_CLASSES, IMG_SIZE).to(device)

#gen = LSTM_Generator(200, 100, num_layers, num_classes).to(device)
#critic = LSTM_Critic(input_size, hidden_size, num_layers, num_classes).to(device)

initialize_weights(gen)
initialize_weights(critic)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# for tensorboard plotting
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/GAN_MNIST/real")
writer_fake = SummaryWriter(f"logs/GAN_MNIST/fake")
step = 0

gen.train()
critic.train()

# %%
# number of trainable parameters gen: 1509801
cnt_params_gen = sum(p.numel() for p in gen.parameters() if p.requires_grad)
print("cnt_params_gen: ", cnt_params_gen)
# number of trainable parameters critic: 216017
cnt_params_crit = sum(p.numel() for p in critic.parameters() if p.requires_grad)
print("cnt_params_crit: ", cnt_params_crit)
# ratio gen/crit: 6.99
print(f"ratio gen/crit: {(cnt_params_gen/ cnt_params_crit):.2f}")


# %%
for epoch in range(NUM_EPOCHS):
    #for batch_idx, (real, labels) in enumerate(tqdm(loader)):
    for batch_idx, (_, _) in enumerate(tqdm(loader)):
        #real = real.to(device)
        real = freal.to(device)
        cur_batch_size = real.shape[0]
        #labels = labels.to(device)
        labels = flabels.to(device)

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise, labels)
            critic_real = critic(real, labels).reshape(-1)
            critic_fake = critic(fake, labels).reshape(-1)

            # detach !? required
            #gp = gradient_penalty(critic, labels, real, fake, device=device)
            #loss_critic = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp)
            loss_critic = (-(torch.mean(critic_real) - torch.mean(critic_fake)))
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            for p in critic.parameters():
                p.data.clamp_(-0.01, 0.01)

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake, labels).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(noise, labels)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
