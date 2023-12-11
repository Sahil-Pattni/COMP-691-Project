#  Author: Sahil Pattni
#  Copyright (c) Sahil Pattni 2023
from typing import List

import pandas as pd
import torch
from tqdm import trange
from IPython.display import clear_output
from entities.pcapfile import PcapFile
from utils.logger_config import LoggerCustom
from QoEGuesser import QoEPredictor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from model_training import load_chunks, preprocess, categorize_bitrate

# %%

df, _ = load_chunks(1, 4, output_directory="out/chunks")
# %%
scaler = StandardScaler()
df.sort_values(by="time", inplace=True)
# train_df.reset_index(drop=True, inplace=True)

# Derive sessions-level features
df["inter_arrival_time"] = df.groupby("session_uid")["time"].diff().dt.total_seconds()
df.reset_index(drop=True, inplace=True)

# Aggregate by session_uid at 1 second granularity
df = df.groupby(["session_uid", pd.Grouper(key="time", freq="1S")]).agg(
    {
        # Summed to be more useful later
        "packet_size": "sum",  # Total payload size of packets in this second
        "payload_size": "sum",  # Total payload size of packets in this second
        "inter_arrival_time": "mean",  # Mean inter-arrival time of packets in this second
        # All other column values are the same for a given sessions
        "src": "first",
        "dst": "first",
        "dst_port": "first",
        "src_port": "first",
        "proto": "first",
        "provider": "first",
        "tcp": "first",
        "udp": "first",
        "fragmentation": "first",
    }
)
# Calculate the bitrate in bits per second
df["packet_size"] = df["size"] * 8
df["payload_size"] = df["payload_size"] * 8
df.rename(columns={"payload_size": "bitrate"}, inplace=True)
df.reset_index(
    inplace=True
)  # Reset the index so that 'session_uid' and 'time' are columns again
# %%
# Calculate the bitrate in bits per second
df["size"] = df["size"] * 8
df.rename(columns={"size": "bitrate"}, inplace=True)
df.reset_index(
    inplace=True
)  # Reset the index so that 'session_uid' and 'time' are columns again

# %%
df.sort_values(by=["session_uid", "time"])

# Drop columns that are no longer needed
df.drop(columns=["time", "provider", "src", "dst"], inplace=True)

# Frequency encode session_uid
session_uid_counts = df["session_uid"].value_counts()
df["session_uid"] = df["session_uid"].map(session_uid_counts)
df.dropna(inplace=True)

_old_nrows = df.shape[0]

# Filter down video-only data
lower_limit = 400 * 1000  # in kbps
df = df[(df["bitrate"] >= lower_limit)]

# Categorize the bitrate
df["bitrate"] = df["bitrate"].apply(categorize_bitrate)

# Segment by session_uid
grouped = df.groupby("session_uid")
sessions = []
columns: List[str] = None
for name, group in grouped:
    filtered_group = group.drop(columns=["bitrate", "session_uid"])
    features = filtered_group.values
    if columns is None:
        columns = filtered_group.columns
    labels = group["bitrate"].values

    # If scaler has not been fit yet
    if not (hasattr(scaler, "mean_") and hasattr(scaler, "var_")):
        scaler.fit(features)

    sessions.append(
        (
            torch.tensor(scaler.transform(features), dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )
    )
