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

from scapy.all import TCP, UDP, PacketList
from typing import List
from tqdm import tqdm

# Suppress warnings
import warnings

# Logging
import logging

# Set logging level to ONLY show errors
logging.basicConfig(level=logging.ERROR)

warnings.filterwarnings("ignore")


DATA_PATH: str = "data/"
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

    prefix: str = __get_prefix(jupyter_prefix)
    return [f"{prefix}{file}" for file in os.listdir(prefix) if file.endswith(".pcap")]


def __get_prefix(flag: bool):
    return f"../{DATA_PATH}" if flag else DATA_PATH
