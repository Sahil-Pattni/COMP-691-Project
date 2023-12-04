#  Author: Sahil Pattni
#  Copyright (c) Sahil Pattni 2023

# %%
"""
A class to read in and process Puffer data.


"""

# %%
#


import warnings

warnings.filterwarnings("ignore")

import logging
from utils.logger_config import LoggerCustom

logger = LoggerCustom.get_logger("Data Processing", level=logging.DEBUG)

from entities.sessions import Sessions
from utils.utils import get_prefix

import pandas as pd


def merge_dataframes(df1, df2, suffixes=("_x", "_y"), how="left"):
    """
    Merge two dataframes while keeping the number of rows of the first dataframe constant.

    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.
        suffixes (tuple, optional): Suffixes to add to columns with the same name in both DataFrames. Defaults to ("_x", "_y").
        how (str, optional): How to merge the DataFrames. Defaults to "left".

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    # Find common columns to merge on (excluding those with suffixes in df1)
    common_cols = [
        col for col in df1.columns if col in df2.columns and not col.endswith(suffixes)
    ]

    # Perform the merge
    merged_df = pd.merge(df1, df2, on=common_cols, how=how, suffixes=suffixes)

    return merged_df


if __name__ == "__main__":
    s = Sessions(
        f"{get_prefix(flag=True)}data/Puffer/2023-12-02T2023-12-03", nrows=10000
    )
    print(f"Client Buffer: {s.client_buffer.shape}")
    print(f"Video Acked: {s.video_acked.shape}")
    print(f"Video Sent: {s.video_sent.shape}")
    print(f"Video Size: {s.video_size.shape}")
    print(f"SSIM: {s.ssim.shape}")
