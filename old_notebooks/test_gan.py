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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# https://learnopencv.com/conditional-gan-cgan-in-pytorch-and-tensorflow/

# %%
# https://towardsdatascience.com/build-a-super-simple-gan-in-pytorch-54ba349920e4
# https://github.com/ozanciga/gans-with-pytorch#cgan
# https://github.com/soumith/ganhacks

# https://machinelearningmastery.com/lstm-autoencoders/
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/



# %%
from typing import List
from typing import Tuple
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from random import shuffle

import matplotlib.pyplot as plt
import copy

# required improvements
# conditional gan

# %%
torch.cuda.is_available()

# %%
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
device

# %%
# gan: random -> even number
# cgan: (0|1) | random -> even number < maxint/2 if 0, >= maxint/2 else

# lstm gan: 
# discriminator: n -> 1 LSTM
# how to condition an LSTM ?


# %%
def convert_float_matrix_to_int_list(float_matrix: np.array, threshold: float = 0.5) -> List[int]:
    """Converts generated output in binary list form to a list of integers

    Args:
        float_matrix: A matrix of values between 0 and 1 which we want to threshold and convert to
            integers
        threshold: The cutoff value for 0 and 1 thresholding.

    Returns:
        A list of integers.
    """
    return [int("".join([str(int(y)) for y in x]), 2) for x in float_matrix >= threshold]

def create_binary_list_from_int(number: int) -> List[int]:
    if number < 0 or type(number) is not int:
        raise ValueError("Only Positive integers are allowed")

    return [int(x) for x in list(bin(number))[2:]]


def generate_even_data(max_int: int, batch_size: int=16) -> Tuple[List[int], List[List[int]]]:
    # Get the number of binary places needed to represent the maximum number
    max_length = int(math.log(max_int, 2))

    # Sample batch_size number of integers in range 0-max_int/2
    # so we can multiiply them by 2 later
    sampled_integers = np.random.randint(0, int(max_int / 2), batch_size)

    # create a list of labels all ones because all numbers are even
    labels = [[1]] * batch_size

    # Generate a list of binary numbers for training.
    data = [create_binary_list_from_int(int(x * 2)) for x in sampled_integers]
    data = [([0] * (max_length - len(x))) + x for x in data]

    return labels, data


def generate_even_conditional_data(max_int: int, batch_size: int=16) -> Tuple[List[int], List[Tuple[int,List[int]]]]:
    # Get the number of binary places needed to represent the maximum number
    max_length = int(math.log(max_int, 2))

    # Sample batch_size number of integers in range 0-max_int/2
    # so we can multiiply them by 2 later
    sampled_integers_smax_int = np.random.randint(0, int(max_int / 4), batch_size//2)
    sampled_integers_gemax_int = np.random.randint(max_int / 4, int(max_int / 2), batch_size//2)

    # create a list of labels all ones because all numbers are even
    labels = [[1]] * batch_size

    # Generate a list of binary numbers for training.
    data_smax_int = [create_binary_list_from_int(int(x)*2) for x in sampled_integers_smax_int]
    data_gemax_int = [create_binary_list_from_int(int(x)*2) for x in sampled_integers_gemax_int]

    data = data_smax_int + data_gemax_int
    data = [([0] * (max_length - len(x))) + x for x in data]
    shuffle(data)

    data_labes = [ [int(int.from_bytes(x, "big", signed="False")>= max_int/2)] for x in data]    
    return labels, data, data_labes


# %%
class Generator(nn.Module):
    def __init__(self, input_length: int):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(int(input_length), int(input_length))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense_layer(x))
    
    
class Discriminator(nn.Module):
    def __init__(self, input_length: int):
        super(Discriminator, self).__init__()
        self.dense = nn.Linear(int(input_length), 1);
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense(x))

# https://github.com/Lornatang/CGAN-PyTorch/blob/master/cgan_pytorch/models/generator.py
    
class CondGenerator(nn.Module):
    def __init__(self, input_length: int, num_classes: int = 1):
        super(CondGenerator, self).__init__()
        self.dense_layer1 = nn.Linear(int(input_length)+num_classes, 32)
        self.batchnnorm = nn.BatchNorm1d(32)
        self.dense_layer3 = nn.Linear(32, int(input_length))
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor, labels: list = None) -> torch.Tensor:
        x = torch.cat([inputs, labels], dim=-1)
        x = self.batchnnorm(self.lrelu(self.dense_layer1(x)))
        x = self.sigmoid(self.dense_layer3(x))
        return x
    
    
class CondDiscriminator(nn.Module):
    def __init__(self, input_length: int,num_classes: int = 1):
        super(CondDiscriminator, self).__init__()
        self.dense_layer1 = nn.Linear(int(input_length)+num_classes,32)
        self.dense_layer2 = nn.Linear(32, 1)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor,labels: list = None):
        x = torch.cat([inputs, labels], dim=-1)
        x = self.lrelu(self.dense_layer1(x))
        x = self.sigmoid(self.dense_layer2(x))
        return x


# Use LSTM for prediction
# TODO
class LSTMGenerator(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int = 1):
        super(LSTMGenerator, self).__init__()

        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.encoder_labels = nn.Linear(num_classes,hidden_size)
        self.encoder_data = nn.LSTM()
        self.decoder = nn.LSTM()
        self.activation = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels_encoded = self.encoder_labels(labels)
        inputs_encoded = self.encoder_data(inputs)
        encoding = torch.concat([inputs_encoded, labels_encoded], dim=-1)
        print(encoding.shape)
        out, _ = self.decoder(encoding)
        return out
    
# https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
# Only gives same results !?
class LSTMDiscriminator(nn.Module):
    def __init__(self,hidden_size: int, num_classes: int = 1):

        super(LSTMDiscriminator,self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
        self.encoder_linear = nn.Linear(num_classes,16)
        self.merge_linear1 = nn.Linear(32,16)
        self.batchnorm = nn.BatchNorm1d(16)
        self.merge_linear2 = nn.Linear(16,1)
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU()

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        
        features_encoded,_ = self.lstm(inputs)
        # get last ouput of lstm encoder
        features_encoded = features_encoded[:,-1,:]
        labels_encoded = self.lrelu(self.encoder_linear(labels))

        encoding = torch.concat([features_encoded, labels_encoded], dim=-1)
        out = self.batchnorm(self.lrelu(self.merge_linear1(encoding)))
        out = self.sigmoid(self.merge_linear2(out))
        return out


# %%
max_int = 8
batch_size = 8
max_len = int(math.log(max_int, 2))
labels, data = generate_even_data(max_int,batch_size)
labels, data


# %%
def lstm_train(max_int: int = 128,num_classes: int =1, batch_size: int = 16, training_steps: int = 500):
    input_length = int(math.log(max_int, 2))
    generator = CondGenerator(input_length,num_classes)
    discriminator = LSTMDiscriminator(input_length, num_classes)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    # loss
    loss = nn.BCELoss()
    discriminator_losses = []
    generator_losses = []

    for i in tqdm(range(training_steps)):
        prnt = False

        if i % (training_steps/10) == 0:
            prnt = True
            
        # zero the gradients on each iteration
        generator_optimizer.zero_grad()

        # Create noisy input for generator
        # Need float type instead of int
        noise = torch.normal(mean=0.5,std=torch.ones(batch_size*input_length))
        noise = noise.reshape((batch_size, input_length))
        #noise = torch.normal(0, 2, size=(batch_size, input_length)).float()
        noise_labels = torch.randint(0,2,size=(batch_size,1)).float()

        generated_data = generator(noise, noise_labels)
        generated_data_seq = generated_data.reshape((batch_size, input_length,1))

        # Generate examples of even real data
        true_labels, true_data, true_data_labels = generate_even_conditional_data(max_int, batch_size=batch_size)

        true_labels = torch.tensor(true_labels).float() 
        true_labels = true_labels + (torch.rand_like(true_labels)*0.1)

        true_data = torch.tensor(true_data).float()
        true_data_labels = torch.tensor(true_data_labels).float()
        true_data_seq = true_data.reshape((batch_size, input_length,1))

        # Train the generator
        # We invert the labels here and don't train the discriminator because we want the generator
        # to make things the discriminator classifies as true.
        generator_discriminator_out = discriminator(generated_data_seq, noise_labels)
        generator_loss = loss(generator_discriminator_out, true_labels)
        generator_loss.backward()
        generator_optimizer.step()

        # Train the discriminator on the true/generated data
        discriminator_optimizer.zero_grad()
        true_discriminator_out = discriminator(true_data_seq, true_data_labels)
        true_discriminator_loss = loss(true_discriminator_out, true_labels)
 
    
        # add .detach() here think about this
        generator_discriminator_out = discriminator(generated_data_seq.detach(),noise_labels.detach())
        generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros(batch_size,1)+ (torch.rand_like(true_labels)*0.1))
        discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()

        discriminator_losses.append(discriminator_loss.item())
        generator_losses.append(generator_loss.item())

        if prnt:
            res = convert_float_matrix_to_int_list(generated_data)
            correct = 0
            for r,l in zip(res, noise_labels):
                if r%2 == 0 and ((l ==0 and r < max_int/2) or (l==1 and r >= max_int/2)):
                    correct += 1
            print(i," : ","generated_data: ", res, "correct rate: ", correct / len(res) )
            print(i, " : ", "noise_labels:   ", noise_labels.numpy().flatten())
            print(i," : ","discriminator_loss: ", discriminator_loss.item())
            print(i," : ", "generator_loss:     ", generator_loss.item())

            fig, ax = plt.subplots()
            x = list(range(len(discriminator_losses)))
            ax.plot(x, discriminator_losses,"-b", label="dis_loss")
            ax.plot(x, generator_losses, "-r", label="gen_loss")
            plt.legend(loc="upper left")
            plt.show()
        

max_int = 8
batch_size = 16
training_steps = 10000
num_classes = 1

lstm_train(max_int,num_classes, batch_size, training_steps)


# %%
def ctrain(max_int: int = 128,num_classes: int =1, batch_size: int = 16, training_steps: int = 500):
    input_length = int(math.log(max_int, 2))
    generator = CondGenerator(input_length,num_classes)
    discriminator = CondDiscriminator(input_length, num_classes)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    # loss
    loss = nn.BCELoss()
    discriminator_losses = []
    generator_losses = []
    
    for i in tqdm(range(training_steps)):
        
        prnt = False
        if not i % 1000:
            prnt = True
            
        # zero the gradients on each iteration
        generator_optimizer.zero_grad()

        # Create noisy input for generator
        # Need float type instead of int
        noise = torch.randint(0, 2, size=(batch_size, input_length)).float()
        noise_labels = torch.randint(0,2,size=(batch_size,1)).float()
        generated_data = generator(noise, noise_labels)

        # Generate examples of even real data
        true_labels, true_data, true_data_labels = generate_even_conditional_data(max_int, batch_size=batch_size)
        
        true_labels = torch.tensor(true_labels).float()
        true_data = torch.tensor(true_data).float()
        true_data_labels = torch.tensor(true_data_labels).int()   

        # Train the generator
        # We invert the labels here and don't train the discriminator because we want the generator
        # to make things the discriminator classifies as true.
        generator_discriminator_out = discriminator(generated_data, noise_labels)
        generator_loss = loss(generator_discriminator_out, true_labels)
        generator_loss.backward()
        generator_optimizer.step()


        # Train the discriminator on the true/generated data
        discriminator_optimizer.zero_grad()
        true_discriminator_out = discriminator(true_data, true_data_labels)
        true_discriminator_loss = loss(true_discriminator_out, true_labels)

        # add .detach() here think about this
        generator_discriminator_out = discriminator(generated_data.detach(),noise_labels.detach())
        generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros(batch_size,1))
        discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()

        discriminator_losses.append(discriminator_loss.item())
        generator_losses.append(generator_loss.item())

        if prnt:
            res = convert_float_matrix_to_int_list(generated_data)
            correct = 0
            for r,l in zip(res, noise_labels):
                if r%2 == 0 and ((l ==0 and r < max_int/2) or (l==1 and r >= max_int/2)):
                    correct += 1
            print(i," : ","generated_data: ", res, "correct rate: ", correct / len(res) )
            print(i, " : ", "noise_labels:   ", noise_labels.numpy().flatten())
            print(i," : ","discriminator_loss: ", discriminator_loss.item())
            print(i," : ", "generator_loss:     ", generator_loss.item())
            
            
            fig, ax = plt.subplots()
            x = list(range(len(discriminator_losses)))
            ax.plot(x, discriminator_losses,"-b", label="dis_loss")
            ax.plot(x, generator_losses, "-r", label="gen_loss")
            plt.legend(loc="upper left")
            plt.show()

# %%
max_int = 16
batch_size = 16 
training_steps = 8000
num_classes = 1

ctrain(max_int,num_classes, batch_size, training_steps)


# %%
def train(max_int: int = 16, batch_size: int = 16, training_steps: int = 500):
    input_length = int(math.log(max_int, 2))

    # Models
    generator = Generator(input_length)
    discriminator = Discriminator(input_length)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    # loss
    loss = nn.BCELoss()
    discriminator_losses = []
    generator_losses = []
    for i in tqdm(range(training_steps)):
        
        prnt = False
        if i % 100 == 0:
            prnt = True
            
        # zero the gradients on each iteration
        generator_optimizer.zero_grad()

        # Create noisy input for generator
        # Need float type instead of int
        noise = torch.randint(0, 2, size=(batch_size, input_length)).float()
        generated_data = generator(noise)

        # Generate examples of even real data
        true_labels, true_data = generate_even_data(max_int, batch_size=batch_size)
        true_labels = torch.tensor(true_labels).float()
        true_data = torch.tensor(true_data).float()

        # Train the generator
        # We invert the labels here and don't train the discriminator because we want the generator
        # to make things the discriminator classifies as true.
        generator_discriminator_out = discriminator(generated_data)
        generator_loss = loss(generator_discriminator_out, true_labels)
        generator_loss.backward()
        generator_optimizer.step()


        # Train the discriminator on the true/generated data
        discriminator_optimizer.zero_grad()
        true_discriminator_out = discriminator(true_data)
        true_discriminator_loss = loss(true_discriminator_out, true_labels)

        # add .detach() here think about this
        generator_discriminator_out = discriminator(generated_data.detach())
        generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros(batch_size,1))
        discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()

        discriminator_losses.append(discriminator_loss.item())
        generator_losses.append(generator_loss.item())
        if prnt:
            res = convert_float_matrix_to_int_list(generated_data)
            even = sum([1-(r % 2) for r in res])
            print(i," : ","generated_data: ", convert_float_matrix_to_int_list(generated_data), "abc: ", even / len(res) )

            fig, ax = plt.subplots()
            x = list(range(len(discriminator_losses)))
            ax.plot(x, discriminator_losses,"-b", label="dis_loss")
            ax.plot(x, generator_losses, "-r", label="gen_loss")
            plt.legend(loc="upper left")
            plt.show()

# %%
max_int = 16
batch_size = 16
training_steps = 8000

labels, data = generate_even_data(max_int, batch_size)
train(max_int, batch_size, training_steps)
