import torch
import torch.nn as nn
import torch.nn.functional as F
NOISE_DIM = 10

class ModelGConvTranspose(nn.Module):
    def __init__(self, z_dim):
        self.z_dim = z_dim
        super(ModelGConvTranspose, self).__init__()
        self.fc1 = nn.Linear(self.z_dim + 2 + 3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, 20736)
        
        self.conv1 = nn.ConvTranspose2d(256, 256, 3, stride=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(256, 128, 3)
        self.conv3 = nn.ConvTranspose2d(128, 64, 3)
        self.conv4 = nn.ConvTranspose2d(64, 32, 3)
        self.conv5 = nn.ConvTranspose2d(32, 16, 3)
        self.conv6 = nn.ConvTranspose2d(16, 1, 3)
        
        
    def forward(self, z, ParticleMomentum_ParticlePoint):
        x = F.leaky_relu(self.fc1(
            torch.cat([z, ParticleMomentum_ParticlePoint], dim=1)
        ))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        
        EnergyDeposit = x.view(-1, 256, 9, 9)
        
        EnergyDeposit = F.leaky_relu(self.conv1(EnergyDeposit))
        EnergyDeposit = F.leaky_relu(self.conv2(EnergyDeposit))
        EnergyDeposit = F.leaky_relu(self.conv3(EnergyDeposit))
        EnergyDeposit = F.leaky_relu(self.conv4(EnergyDeposit))
        EnergyDeposit = F.leaky_relu(self.conv5(EnergyDeposit))
        EnergyDeposit = self.conv6(EnergyDeposit)

        return EnergyDeposit