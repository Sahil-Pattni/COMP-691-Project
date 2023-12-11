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


logger = LoggerCustom.get_logger()


def load_session(
    idx: int, output_directory: str = "../out/sessions", prefix: str = "train"
):
    """
    Load the data.
    Args:
        idx (int): The index of the session to load.
        output_directory (str): The directory containing the session.

    Returns:
        Tuple of (features, labels)
    """
    file_path = f"{output_directory}/{prefix}_session_{idx}.pt"

    try:
        features, labels = torch.load(file_path)
        return features, labels
    except FileNotFoundError:
        logger.debug(f"File `{file_path}` not found.")
        return None, None


def train_model(
    model: nn.Module,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    session,
    num_epochs: int,
) -> List[float]:
    """
    Train the model.
    Args:
        model (nn.Module): The model to train.
        loss_function (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        session: The list of session to train on.
        num_epochs (int): The number of epochs to train for.
    """
    with trange(num_epochs, desc="Epochs", disable=False) as epochs:
        epoch_losses = []
        for _ in epochs:
            losses = []
            for features, labels in session:
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

            avg_loss_per_epoch = np.mean(losses)
            epoch_losses.append(avg_loss_per_epoch)
            # Add additional metrics to the progress bar
            _agg_loss = (
                epoch_losses[-1] if len(epoch_losses) > 0 else avg_loss_per_epoch
            )
            epochs.set_postfix(loss=_agg_loss)

    return epoch_losses


def evaluate(model: nn.Module, session, loss_function: nn.Module):
    """
    Evaluate the model.
    Args:
        model (nn.Module): The model to evaluate.
        session (List[tuple]): The list of session to evaluate on.
    """
    # Evaluate the model

    losses = []
    y_preds = []
    y_labels = []
    model.eval()
    with torch.no_grad():
        for features, labels in session:
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
    NUM_EPOCHS: int = 300
    HIDDEN_LAYER_SIZE: int = 300
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

        sessions = [load_session(i, prefix="train") for i in range(1, 14)]
        # Train the model
        epoch_losses: List[float] = train_model(
            model, loss_function, optimizer, sessions, NUM_EPOCHS
        )

        logger.info(f"Model trained. Final loss: {epoch_losses[-1]}")

        # Save the model
        save(model, scaler, MODEL_OUTPUT_DIRECTORY, suffix=SUFFIX)
