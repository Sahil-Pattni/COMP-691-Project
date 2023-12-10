#  Author: Sahil Pattni
#  Copyright (c) Sahil Pattni 2023
import os
from typing import List

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
from sklearn.preprocessing import StandardScaler
import seaborn as sns


logger = LoggerCustom.get_logger(level="DEBUG")


def load_chunks(
    current_chunk: int,
    window_size: int,
    output_directory: str = "../out/processed_chunks",
):
    """
    Load the data.
    Args:
        current_chunk (int): The current chunk number.
        window_size (int): The number of chunks to load.

    Returns:
        df: The dataframe containing the loaded chunks.
    """
    df = None
    counter: int = 0
    for _ in range(current_chunk, current_chunk + window_size):
        file_path = f"{output_directory}/chunk_{current_chunk}.ftr"
        try:
            if df is None:
                df = pd.read_feather(file_path)
            else:
                df = pd.concat(
                    [
                        df,
                        pd.read_feather(
                            file_path,
                        ),
                    ]
                )
            counter += 1
        except FileNotFoundError:
            logger.debug(f"File `{file_path}` not found.")
            break
    return df, counter


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


def preprocess(df: pd.DataFrame, scaler):
    """
    Preprocess the data.
    Args:
        df: The dataframe to preprocess.

    Returns:
        X: The features.
        y: The labels.
    """
    df.sort_values(by="time", inplace=True)
    # df.reset_index(drop=True, inplace=True)

    # Derive session-level features
    df["inter_arrival_time"] = (
        df.groupby("session_uid")["time"].diff().dt.total_seconds()
    )
    # df.dropna(inplace=True)
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

    return sessions, columns


def train_model(
    model: nn.Module,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    sessions: List[tuple],
    num_epochs: int,
) -> List[float]:
    """
    Train the model.
    Args:
        model (nn.Module): The model to train.
        loss_function (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        sessions (List[tuple]): The list of sessions to train on.
        num_epochs (int): The number of epochs to train for.
    """
    losses = []
    with trange(num_epochs, desc="Epochs", disable=False) as epochs:
        for _ in epochs:
            for features, labels in sessions:
                optimizer.zero_grad()
                model.hidden_cell = (
                    torch.zeros(1, 1, model.hidden_layer_size),
                    torch.zeros(1, 1, model.hidden_layer_size),
                )

                y_pred = model(features)
                single_loss = loss_function(y_pred, labels)
                losses.append(single_loss.item())

                single_loss.backward()
                optimizer.step()

            _loss = np.mean(losses)
            # Add additional metrics to the progress bar
            epochs.set_postfix(loss=_loss)

    return losses


def evaluate_model(model: nn.Module, sessions: List[tuple], loss_function: nn.Module):
    """
    Evaluate the model.
    Args:
        model (nn.Module): The model to evaluate.
        sessions (List[tuple]): The list of sessions to evaluate on.
    """
    # Evaluate the model

    losses = []
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
            y_preds.append(y_test_pred)
            y_labels.append(labels)
            test_loss = loss_function(y_test_pred, labels)
            losses.append(test_loss.item())

    logger.debug(f"Average loss: {np.mean(losses)}")

    y_preds = torch.cat(y_preds)
    y_labels = torch.cat(y_labels)

    return y_preds, y_labels, losses


def save(model: nn.Module, scaler, directory: str, suffix: str = ""):
    """
    Save the model.
    Args:
        model (nn.Module): The model to save.
        directory (str): The directory to save the model to.
    """
    # Create a hash out of tunable parameters
    if suffix != "":
        suffix = "_" + suffix
    torch.save(model.state_dict(), f"{directory}/model{suffix}.pt")
    torch.save(scaler, f"{directory}/scaler{suffix}.pt")


def load(model: nn.Module, directory: str, suffix: str = ""):
    """
    Load the model.
    Args:
        model (nn.Module): The model to load.
        directory (str): The directory to load the model from.
    """
    model.load_state_dict(torch.load(f"{directory}/model{suffix}.pt"))
    return torch.load(f"{directory}/scaler{suffix}.pt")


if __name__ == "__main__":
    logger = LoggerCustom.get_logger(level="DEBUG")

    # Tunable parameters
    LOAD: bool = False  # Whether to load the model or train a new one
    BATCH_SIZE: int = 100  # How many chunks to load at once
    N_BATCHES: int = 3  # How many batches to load
    NUM_EPOCHS: int = 500
    HIDDEN_LAYER_SIZE: int = 200
    scaler = StandardScaler()  # Global scaler
    scaler_fit = False
    SUFFIX: str = ""  # Used to differentiate between models
    MODEL_OUTPUT_DIRECTORY: str = "../out/model"
    MODEL_PATH: str = f"{MODEL_OUTPUT_DIRECTORY}/model{SUFFIX}.pt"

    INPUT_SIZE: int = 8
    OUTPUT_SIZE: int = 4

    # Create the model
    model = QoEPredictor(INPUT_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_SIZE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    current_chunk: int = 1
    current_epoch: int = 1

    if LOAD and os.path.exists(MODEL_PATH):
        load(model, MODEL_PATH)
        logger.info("Model loaded.")
    else:
        losses: List[float] = []
        for batch_number in range(N_BATCHES):
            logger.debug(
                f"Batch {batch_number:,}:\nLoading chunks {current_chunk:,} to {current_chunk + BATCH_SIZE:,}..."
            )

            df, num_chunks_loaded = load_chunks(current_chunk, BATCH_SIZE)

            if df is None:
                logger.info("No more chunks to load. Exiting...")
                break

            sessions, columns = preprocess(df, scaler)
            n_rows = sum([len(session[0]) for session in sessions])

            logger.debug(f"Training model on {n_rows:,} row(s)...")

            # Train the model
            batch_losses: List[float] = train_model(
                model, loss_function, optimizer, sessions, NUM_EPOCHS
            )

            # Track average batch loss
            losses.append(np.mean(batch_losses))

            # TODO: Remove this
            # current_batch_losses = []
            # # Train the model
            # with trange(NUM_EPOCHS, desc="Epochs", disable=False) as epochs:
            #     for inner_epoch in epochs:
            #         for features, labels in sessions:
            #             optimizer.zero_grad()
            #             model.hidden_cell = (
            #                 torch.zeros(1, 1, model.hidden_layer_size),
            #                 torch.zeros(1, 1, model.hidden_layer_size),
            #             )
            #
            #             y_pred = model(features)
            #             single_loss = loss_function(y_pred, labels)
            #             current_batch_losses.append(single_loss.item())
            #
            #             single_loss.backward()
            #             optimizer.step()
            #
            #         _loss = np.mean(current_batch_losses)
            #         # Add additional metrics to the progress bar
            #         epochs.set_postfix(loss=_loss)
            #
            #         current_epoch += 1
            #
            # losses.append(np.mean(current_batch_losses))

            current_chunk += num_chunks_loaded
