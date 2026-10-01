import torch
from torch import nn


class SimpleTcnEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size):
        # self.device = device
        super(SimpleTcnEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 3 * channel_size), self.activation_fn,
        )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=8, stride=4),
                self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=5, stride=1),
                self.activation_fn,
                nn.Conv1d(in_channels=channel_size, out_channels=channel_size, kernel_size=5, stride=1),
                self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=4, stride=2),
                self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=2, stride=1),
                self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=6, stride=2),
                self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=4, stride=2),
                self.activation_fn,
                nn.Flatten())
        else:
            print("Recommend history length to be 10, 20 or 50")
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=6, stride=2),
                self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=4, stride=2),
                self.activation_fn,
                nn.Flatten())

        if self.encoder_output_tanh:
            activation_fn = nn.Tanh()

        self.linear_output = nn.Sequential(nn.Linear(3 * channel_size, output_size), activation_fn)

    def forward(self, obs):
        # obs.Size([num_envs, history_length, num_base_obs])
        num_envs = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([num_envs * T, -1]))  # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([num_envs, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output
