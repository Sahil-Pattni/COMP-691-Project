#  Author: Sahil Pattni
#  Copyright (c) Sahil Pattni 2023

import torch

import torch.nn as nn


class QoEPredictor(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(QoEPredictor, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (
            torch.zeros(1, 1, self.hidden_layer_size),
            torch.zeros(1, 1, self.hidden_layer_size),
        )

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), 1, -1), self.hidden_cell
        )
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


# Define the model
input_size = 7  # Number of features in your dataset
hidden_layer_size = 100  # Can be tuned
output_size = 1  # Predicting ssim_index, which is a single value

model = QoEPredictor(input_size, hidden_layer_size, output_size)
