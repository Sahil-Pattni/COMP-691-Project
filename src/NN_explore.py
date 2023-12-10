#  Author: Sahil Pattni
#  Copyright (c) Sahil Pattni 2023

import pandas as pd
import hashlib

import torch
from torch import nn

from entities.pcapfile import PcapFile
from utils.logger_config import LoggerCustom
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

logger = LoggerCustom.get_logger(level="DEBUG")
from QoEGuesser import QoEPredictor


#
# # Load the data
# df = None
# window_size = 100
#
# for i in range(1, window_size + 1):
#     logger.debug(f"Loading chunk {i}...")
#     if df is None:
#         df = pd.read_feather(f"out/chunk_{i+1}.ftr")
#     else:
#         df = pd.concat([df, pd.read_feather(f"out/chunk_{i+1}.ftr")])
#
# df.sort_values(by="time", inplace=True)
# df.reset_index(drop=True, inplace=True)
#
# # Group by 'session_uid' to derive session-level features
# df["inter_arrival_time"] = df.groupby("session_uid")["time"].diff().dt.total_seconds()
# df.dropna(inplace=True)
#
# jitter_per_session = df.groupby("session_uid")["inter_arrival_time"].std()
# df["jitter"] = df["session_uid"].map(jitter_per_session)
# df.reset_index(drop=True, inplace=True)
#
# df = df.groupby(["session_uid", pd.Grouper(key="time", freq="1S")]).agg(
#     {
#         # Summed to be more useful later
#         "size": "sum",  # Total size of packets in this second
#         "payload_size": "sum",  # Total payload size of packets in this second
#         "inter_arrival_time": "mean",  # Mean inter-arrival time of packets in this second
#         # All other column values are the same for a given session
#         "jitter": "first",
#         "src": "first",
#         "dst": "first",
#         "dst_port": "first",
#         "src_port": "first",
#         "proto": "first",
#         "provider": "first",
#         "tcp": "first",
#         "udp": "first",
#         "fragmentation": "first",
#     }
# )
#
# # Calculate the bitrate in bits per second
# df["size"] = df["size"] * 8 / 60
# df.rename(columns={"size": "bitrate"}, inplace=True)
# df.reset_index(
#     inplace=True
# )  # Reset the index so that 'session_uid' and 'time' are columns again
#
# # One-Hot-Encode the categorical variables (provider)
# provider_dummies = pd.get_dummies(df["provider"], prefix="provider")
# df = pd.concat([df, provider_dummies], axis=1)
# df.sort_values(by=["session_uid", "time"])
# df.drop(columns=["time", "provider", "src", "dst"], inplace=True)
#
# # Frequency encode session_uid
# session_uid_counts = df["session_uid"].value_counts()
# df["session_uid"] = df["session_uid"].map(session_uid_counts)
#
# df.dropna(inplace=True)
#
# logger.debug("Completed preprocessing.")
#
# X, y = df.drop(columns=["bitrate", "jitter"]), df["bitrate"]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y)
# # Convert the data to DataLoader objects
# train_data = TensorDataset(torch.Tensor(X_train.values), torch.Tensor(y_train.values))
# train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
#
# logger.debug("Data prepared for model.")
# # Define the model
# input_size = train_data.tensors[0].shape[1]
# hidden_layer_size = 100  # Can be tuned
# output_size = 1  # Predicting low, medium, high
# model = QoEPredictor(input_size, hidden_layer_size, output_size)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# logger.debug("Model defined. Training...")
# # Train the model
# epochs = 50  # or whatever number of epochs you choose
# for epoch in range(epochs):
#     logger.debug(f"Epoch {epoch}...")
#     for features, labels in train_loader:
#         optimizer.zero_grad()
#         model.hidden_cell = (
#             torch.zeros(1, 1, model.hidden_layer_size),
#             torch.zeros(1, 1, model.hidden_layer_size),
#         )
#
#         y_pred = model(features)
#         single_loss = criterion(y_pred, labels)
#         single_loss.backward()
#         optimizer.step()
#
#     if epoch % 10 == 0:  # Adjust the print frequency as needed
#         print(f"epoch: {epoch} loss: {single_loss.item()}")
#
# logger.debug("Model trained.")
# # %%
#
# # Evaluate the model
# test_data = TensorDataset(torch.Tensor(X_test.values), torch.Tensor(y_test.values))
# test_loader = DataLoader(test_data, batch_size=1)
#
# losses = []
# model.eval()
# with torch.no_grad():
#     for features, labels in test_loader:
#         model.hidden = (
#             torch.zeros(1, 1, model.hidden_layer_size),
#             torch.zeros(1, 1, model.hidden_layer_size),
#         )
#         y_test_pred = model(features)
#         test_loss = criterion(y_test_pred, labels)
#         losses.append(test_loss.item())
#
#
# # Average loss
# sum(losses) / len(losses)
