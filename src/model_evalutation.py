#  Author: Sahil Pattni
#  Copyright (c) Sahil Pattni 2023
import numpy as np
import torch
from torch import nn
import os
from QoEGuesser import QoEPredictor
from utils.logger_config import LoggerCustom
from sklearn.metrics import classification_report, confusion_matrix
from model_training import load_session, load


def evaluate(model: nn.Module, sessions, loss_function: nn.Module):
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

    return y_preds, y_labels


if __name__ == "__main__":
    logger = LoggerCustom.get_logger(level="DEBUG")

    # Load model
    model = QoEPredictor(8, 100, 4)
    load(model, "../out/model")
    sessions = [
        torch.load(f"../out/sessions/test_session_{i}.pt")
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
    ]

    # Confusion matrix
    y_preds, y_labels = evaluate(model, sessions, nn.CrossEntropyLoss())

    y_preds = torch.cat(y_preds)
    y_labels = torch.cat(y_labels)

    y_preds = torch.argmax(y_preds, dim=1)

    print(classification_report(y_labels, y_preds))
