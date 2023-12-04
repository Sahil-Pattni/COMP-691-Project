# %%
"""
This script contains utility functions for the project.

Author: Sahil Pattni

Copyright (c) Sahil Pattni
"""

# ------- IMPORTS, GLOBALS & OTHER SETUP ---------- #
import os
import re
import pickle
import pandas as pd

from scapy.all import PacketList
from typing import List
from tqdm import tqdm

# Suppress warnings
import warnings

# Logging
import logging

# Set logging level to ONLY show errors
logging.basicConfig(level=logging.ERROR)

warnings.filterwarnings("ignore")


DATA_PATH: str = "data/Puffer/fake_data/"
# -------------------------- #


def get_dataset_filepaths(jupyter_prefix: bool = True) -> List[str]:
    """Get file paths from a directory

    Args:
        dir (str): Directory to get file paths from
        jupyter_prefix (bool, optional): If running on jupyter cells, use
            prefix: str = "../". Defaults to False.

    Returns:
        List[str]: List of file paths
    """

    prefix: str = get_prefix(jupyter_prefix)
    return [f"{prefix}{file}" for file in os.listdir(prefix) if file.endswith(".pcap")]


def get_prefix(flag: bool):
    return "../" if flag else ""


def common_columns(
    df1: pd.DataFrame, df2: pd.DataFrame, columns: List[str] = []
) -> List[str]:
    """
    Get common columns between two DataFrames.

    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.
        columns (List[str], optional): List of columns to check.
        Defaults to an empty list.

    Returns:
        List[str]: List of common columns.
    """
    columns = columns if columns else df1.columns.to_list()
    return [col for col in columns if col in df1.columns and col in df2.columns]
