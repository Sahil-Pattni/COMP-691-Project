#  Author: Sahil Pattni
#  Copyright (c) Sahil Pattni 2023

import pandas as pd
import hashlib

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import trange
from IPython.display import clear_output
from entities.pcapfile import PcapFile
from utils.logger_config import LoggerCustom
from QoEGuesser import QoEPredictor
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


logger = LoggerCustom.get_logger(level="DEBUG")


def load_chunks(current_chunk: int, window_size: int):
    """
    Load the data.
    Args:
        current_chunk (int): The current chunk number.
        window_size (int): The number of chunks to load.

    Returns:
        df: The dataframe containing the loaded chunks.
    """
    df = None
    for i in range(current_chunk, current_chunk + window_size):
        try:
            if df is None:
                df = pd.read_feather(f"../out/chunk_{current_chunk}.ftr")
            else:
                df = pd.concat(
                    [df, pd.read_feather(f"../out/chunk_{current_chunk}.ftr")]
                )
            current_chunk += 1
        except FileNotFoundError:
            logger.debug(f"Chunk {current_chunk:,} not found.")
            break
    return df, i


def __categorize_bitrate(bitrate: float):
    # Convert to Kbps
    bitrate = bitrate / 1000

    if bitrate < 800:
        return 0  # "LOW_QUALITY"
    elif bitrate <= 1200:
        return 1  # "MED_QUALITY"
    elif bitrate <= 2500:
        return 2  # "HIGH_QUALITY"
    else:
        return 3  # "ULTRA_QUALITY"


def __preprocess(df: pd.DataFrame):
    """
    Preprocess the data.
    Args:
        df: The dataframe to preprocess.

    Returns:
        X: The features.
        y: The labels.
    """
    df.sort_values(by="time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Derive session-level features
    df["inter_arrival_time"] = (
        df.groupby("session_uid")["time"].diff().dt.total_seconds()
    )
    df.dropna(inplace=True)
    jitter_per_session = df.groupby("session_uid")["inter_arrival_time"].std()
    df["jitter"] = df["session_uid"].map(jitter_per_session)
    df.reset_index(drop=True, inplace=True)

    # Aggregate by session_uid at 1 second granularity
    df = df.groupby(["session_uid", pd.Grouper(key="time", freq="1S")]).agg(
        {
            # Summed to be more useful later
            "size": "sum",  # Total size of packets in this second
            "payload_size": "sum",  # Total payload size of packets in this second
            "inter_arrival_time": "mean",  # Mean inter-arrival time of packets in this second
            # All other column values are the same for a given session
            "jitter": "first",
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
    df["size"] = df["size"] * 8
    df.rename(columns={"size": "bitrate"}, inplace=True)
    df.reset_index(
        inplace=True
    )  # Reset the index so that 'session_uid' and 'time' are columns again

    df.sort_values(by=["session_uid", "time"])

    # Drop columns that are no longer needed
    df.drop(columns=["time", "provider", "src", "dst"], inplace=True)

    # Frequency encode session_uid
    session_uid_counts = df["session_uid"].value_counts()
    df["session_uid"] = df["session_uid"].map(session_uid_counts)
    df.dropna(inplace=True)
    logger.debug("Completed preprocessing.")

    _old_nrows = df.shape[0]
    # Filter down video-only data
    lower_limit = 400 * 1000  # in kbps
    df = df[(df["bitrate"] >= lower_limit)]
    logger.debug(
        f"Filtered out {(_old_nrows - df.shape[0]):,} rows, leaving {df.shape[0]:,} rows."
    )

    # Categorize the bitrate
    df["bitrate"] = df["bitrate"].apply(__categorize_bitrate)
    df.drop(columns=["jitter"], inplace=True)

    # Segment by session_uid
    grouped = df.groupby("session_uid")
    sessions = []
    for name, group in grouped:
        features = group.drop(columns=["bitrate", "session_uid"]).values
        labels = group["bitrate"].values
        sessions.append(
            (
                torch.tensor(features, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long),
            )
        )

    return sessions


if __name__ == "__main__":
    logger = LoggerCustom.get_logger(level="DEBUG")

    BATCH_SIZE: int = 50  # How many chunks to load at once
    N_BATCHES: int = 4  # How many batches to load
    NUM_EPOCHS: int = 100
    HIDDEN_LAYER_SIZE: int = 100
    INPUT_SIZE: int = 8
    OUTPUT_SIZE: int = 4

    # Create the model
    model = QoEPredictor(INPUT_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_SIZE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    current_chunk: int = 1
    current_epoch: int = 1

    losses = []
    for batch_number in range(N_BATCHES):
        logger.debug(
            f"Batch {batch_number:,}:\nLoading chunks {current_chunk:,} to {current_chunk + BATCH_SIZE:,}..."
        )
        df, num_chunks_loaded = load_chunks(current_chunk, BATCH_SIZE)
        if df is None:
            logger.info("No more chunks to load. Exiting...")
            break

        sessions = __preprocess(df)

        logger.debug(f"Training model on {len(sessions):,} row(s)...")

        current_batch_losses = []
        # Train the model
        with trange(NUM_EPOCHS, desc="Epochs", disable=False) as epochs:
            for inner_epoch in epochs:
                for features, labels in sessions:
                    optimizer.zero_grad()
                    model.hidden_cell = (
                        torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size),
                    )

                    y_pred = model(features)
                    single_loss = loss_function(y_pred, labels)
                    current_batch_losses.append(single_loss.item())

                    single_loss.backward()
                    optimizer.step()

                _loss = np.mean(current_batch_losses)
                # Add additional metrics to the progress bar
                epochs.set_postfix(loss=_loss)

                current_epoch += 1

        losses.append(np.mean(current_batch_losses))

        current_chunk += num_chunks_loaded

    # Evaluate the model
    logger.debug("Evaluating model...")
    df, num_chunks_loaded = load_chunks(75, 100)
    sessions = __preprocess(df)
    test_losses = []

    y_preds = []
    y_labels = []

    model.eval()
    with torch.no_grad():
        for features, labels in sessions:
            model.hidden = (
                torch.zeros(1, 1, model.hidden_layer_size),
                torch.zeros(1, 1, model.hidden_layer_size),
            )
            y_test_pred = model(features)
            test_loss = loss_function(y_test_pred, labels)
            test_losses.append(test_loss.item())

            y_preds.append(torch.argmax(y_test_pred, dim=1))
            y_labels.append(labels)

        # Classification report
        y_preds = torch.cat(y_preds)
        y_labels = torch.cat(y_labels)
        print(classification_report(y_labels, y_preds))

        cm = confusion_matrix(y_labels, y_preds)

        # Normalize the confusion matrix by dividing each row by its sum
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, cmap="Blues", vmin=0, vmax=1)
        plt.gca().invert_yaxis()
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        plt.show()

    logger.debug(f"Average test loss: {np.mean(test_losses)}")
