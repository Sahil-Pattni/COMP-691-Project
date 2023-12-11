#  Author: Sahil Pattni
#  Copyright (c) Sahil Pattni 2023

from typing import List
import os
import random
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from utils.logger_config import LoggerCustom

logger = LoggerCustom.get_logger()


def load_chunks(
    indices: List[int],
    output_directory: str = "../out/chunks",
):
    """
    Load the data.
    Args:
        indices (List[int]): The indices of the chunks to load.
        output_directory (str): The directory containing the chunks.

    Returns:
        train_df: The dataframe containing the loaded chunks.
    """
    df = None
    counter: int = 0
    for idx in indices:
        file_path = f"{output_directory}/chunk_{idx}.ftr"
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


def categorize_bitrate(bitrate: float):
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


def preprocess(df: pd.DataFrame):
    """
    Preprocess the data.
    Args:
        df: The dataframe to preprocess.

    Returns:
        X: The features.
        y: The labels.
    """
    df.sort_values(by="time", inplace=True)
    # train_df.reset_index(drop=True, inplace=True)

    # Derive sessions-level features
    df["inter_arrival_time"] = (
        df.groupby("session_uid")["time"].diff().dt.total_seconds()
    )
    df.reset_index(drop=True, inplace=True)

    # Aggregate by session_uid at 1 second granularity
    df = df.groupby(["session_uid", pd.Grouper(key="time", freq="1S")]).agg(
        {
            # Summed to be more useful later
            "packet_size": "sum",  # Total payload size of packets in this second
            "payload_size": "sum",  # Total payload size of packets in this second
            "inter_arrival_time": "mean",  # Mean inter-arrival time of packets in this second
            # All other column values are the same for a given sessions
            "dst_port": "first",
            "src_port": "first",
            "proto": "first",
            "tcp": "first",
            "udp": "first",
            "fragmentation": "first",
        }
    )
    # Calculate the bitrate in bits per second
    df["packet_size"] = df["packet_size"] * 8
    df["payload_size"] = df["payload_size"] * 8
    df.rename(columns={"payload_size": "bitrate"}, inplace=True)
    df.reset_index(
        inplace=True
    )  # Reset the index so that 'session_uid' and 'time' are columns again

    df.sort_values(by=["session_uid", "time"])

    # Drop columns that are no longer needed
    df.drop(columns=["time"], inplace=True)

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
    df["bitrate"] = df["bitrate"].apply(categorize_bitrate)

    return df


def export_sessions(df: pd.DataFrame, prefix: str):
    sessions = []
    columns: List[str] = None
    # Segment by session_uid
    grouped = df.groupby("session_uid")
    for name, group in grouped:
        filtered_group = group.drop(columns=["bitrate", "session_uid"])
        features = filtered_group.values
        if columns is None:
            columns = filtered_group.columns
        labels = group["bitrate"].values

        # Create a tensor of features and labels
        sessions.append(
            (
                torch.tensor(features, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long),
            )
        )

        # Export the sessions tensor
        fp = os.path.join("../out/sessions", f"{prefix}_session_{name}.pt")
        torch.save(
            (
                torch.tensor(features, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long),
            ),
            fp,
        )
        logger.info(f"Saved session tensor to `{fp}`")

    # Export columns
    fp = os.path.join("../out/sessions", f"columns.pt")
    torch.save(columns, fp)


if __name__ == "__main__":
    seed = 42
    indices = list(range(1, 361))  # We have 360 chunks

    # Randomly select 80% of the chunks
    random.seed(seed)
    random.shuffle(indices)
    training_indices = indices[: int(len(indices) * 0.8)]
    test_indices = indices[int(len(indices) * 0.8) :]

    # Load the training data
    train_df, _ = load_chunks(training_indices)
    logger.info(f"Loaded {len(training_indices):,} training chunks")
    train_df = preprocess(train_df)

    # Fit the scaler on the training data
    scaler = StandardScaler()
    cols_to_scale = ["packet_size", "inter_arrival_time"]

    # Transform the training data
    train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
    logger.info("Scaler fitted and applied to training data.")

    export_sessions(train_df, prefix="train")

    # Load the test data
    test_df, _ = load_chunks(test_indices)
    logger.info(f"Loaded {len(test_indices):,} test chunks")
    test_df = preprocess(test_df)

    # Transform the test data
    test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])

    export_sessions(test_df, prefix="test")
