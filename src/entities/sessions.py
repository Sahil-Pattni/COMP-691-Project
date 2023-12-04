# %%

"""
A class that represents a Puffer session, including 
both client-side and server-side data.

Author: Sahil Pattni

Copyright (c) Sahil Pattni
"""

from typing import List
import pandas as pd
import logging
import re
import os

import warnings

warnings.filterwarnings("ignore")

# Create a logger object
logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)
# Create a handler (e.g., console handler)
handler = logging.StreamHandler()
# Define a custom format without a prefix
formatter = logging.Formatter("%(message)s")
# Set the formatter for the handler
handler.setFormatter(formatter)
# Add the handler to the logger
logger.addHandler(handler)
# Prevent the logger from propagating messages to the root logger
logger.propagate = False


class Sessions:
    # Attributes of a Puffer session.
    attributes = [
        # Unique identifier for each video session, used to track and analyze individual
        # streaming experiences.
        "session_id",
        # Identifier for the specific pair of Adaptive Bitrate (ABR) and congestion
        # control algorithms used in the session.
        "expt_id",
        # Timestamp of the event or measurement in nanoseconds, indicating when the data
        # was recorded.
        "time_ns_gmt",
        # Name or identifier of the TV channel or video stream.
        "channel",
        # Type of event reported by the client, crucial for understanding user experience.
        "event_type",
        # Current size of the playback buffer in seconds, indicating preloaded video.
        "buffer_size",
        # Cumulative time spent rebuffering during the current stream, reflecting
        # interruptions in playback.
        "cum_rebuf",
        # Structural Similarity Index Measure of the video chunk, comparing streamed
        # video to an ideal version.
        "ssim_index",
        # Encoding settings of the video chunk, typically including resolution and CRF.
        "video_format",
        # Size of the individual video chunk in bytes, impacting data transmission.
        "chunk_size",
        # Presentation timestamp of the video chunk, indicating playback time in the stream.
        "video_ts",
        # Size of the congestion window in packets, indicating the volume of data being
        # sent before receiving an acknowledgment.
        "cwnd",
        # Number of unacknowledged packets 'in flight', a metric of network congestion.
        "in_flight_packets",
        # Minimum Round-Trip Time in microseconds, a key network performance metric.
        "min_rtt",
        # Smoothed Round-Trip Time estimate in microseconds, providing an averaged measure
        # of network latency.
        "smoothed_rtt",
        # Estimated rate of data delivery in bytes/second, a critical metric for
        # understanding network throughput.
        "delivery_rate",
    ]

    def __init__(self, directory: str):
        """
        Initialize a Puffer session.

        Args:
            directory (str, optional): Directory to load Puffer sessions
            data from.
        """
        self.__files: dict = self.__find_files(directory)
        self.__load()

    # ----- PUBLIC METHODS ----- #
    # Getters
    @property
    def client_buffer(self) -> pd.DataFrame:
        return self.__client_buffer

    @property
    def video_acked(self) -> pd.DataFrame:
        return self.__video_acked

    @property
    def video_sent(self) -> pd.DataFrame:
        return self.__video_sent

    @property
    def video_size(self) -> pd.DataFrame:
        return self.__video_size

    @property
    def ssim(self) -> pd.DataFrame:
        return self.__ssim

    # ----- PRIVATE METHODS ----- #
    def __find_files(self, directory: str) -> dict:
        """
        Finds the CSV filepaths in a directory for the following datasets:
            - Client buffer
            - Video acked
            - Video sent
            - Video size
            - SSIM

        Args:
            directory (str): Directory to search for files.

        Returns:
            dict: Dictionary of filepaths.
        """
        results: dict = {}
        files: List[str] = os.listdir(directory)

        base_regex: str = r"_.+.csv"
        for file in files:
            if re.search(f"client_buffer{base_regex}", file):
                results["client_buffer"] = os.path.join(directory, file)
                logger.info(f"Found client buffer file: {file}")
            elif re.search(f"video_acked{base_regex}", file):
                results["video_acked"] = os.path.join(directory, file)
                logger.info(f"Found video acked file: {file}")
            elif re.search(f"video_sent{base_regex}", file):
                results["video_sent"] = os.path.join(directory, file)
                logger.info(f"Found video sent file: {file}")
            elif re.search(f"video_size{base_regex}", file):
                results["video_size"] = os.path.join(directory, file)
                logger.info(f"Found video size file: {file}")
            elif re.search(f"ssim{base_regex}", file):
                results["ssim"] = os.path.join(directory, file)
                logger.info(f"Found SSIM file: {file}")

        return results

    def __load(self) -> None:
        """
        Load a Puffer sessions dataset from a CSV file.

        Returns:
            pd.DataFrame: Pandas DataFrame containing Puffer sessions data.
        """
        # Burn last
        self.__df = pd.DataFrame()

        self.__client_buffer: pd.DataFrame = self.load_client_buffer()
        self.__video_acked: pd.DataFrame = self.load_video_acked()
        self.__video_sent: pd.DataFrame = self.load_video_sent()
        self.__video_size: pd.DataFrame = self.load_video_size()
        self.__ssim: pd.DataFrame = self.load_ssim()

    def load_client_buffer(self) -> pd.DataFrame:
        """
        Load a Puffer client buffer dataset from a CSV file.

        Returns:
            pd.DataFrame: Pandas DataFrame containing Puffer client buffer data.
        """

        return self.__read_csv(self.__files["client_buffer"])

    def load_video_acked(self) -> pd.DataFrame:
        """
        Load a Puffer video acked dataset from a CSV file.

        Returns:
            pd.DataFrame: Pandas DataFrame containing Puffer video acked data.
        """

        return self.__read_csv(self.__files["video_acked"])

    def load_video_sent(self) -> pd.DataFrame:
        """
        Load a Puffer video sent dataset from a CSV file.

        Returns:
            pd.DataFrame: Pandas DataFrame containing Puffer video sent data.
        """

        return self.__read_csv(self.__files["video_sent"])

    def load_video_size(self) -> pd.DataFrame:
        """
        Load a Puffer video size dataset from a CSV file.

        Returns:
            pd.DataFrame: Pandas DataFrame containing Puffer video size data.
        """
        return self.__read_csv(self.__files["video_size"])

    def load_ssim(self) -> pd.DataFrame:
        """
        Load a Puffer SSIM dataset from a CSV file.

        Args:
            filepath (str): Filepath of the CSV file to load.

        Returns:
            pd.DataFrame: Pandas DataFrame containing Puffer SSIM data.
        """

        return self.__read_csv(self.__files["ssim"])

    def __read_csv(self, filepath: str) -> pd.DataFrame:
        """
        Read a CSV file into a Pandas DataFrame.

        Args:
            filepath (str): Filepath of the CSV file to read.

        Returns:
            pd.DataFrame: Pandas DataFrame containing CSV data.
        """

        # Replace time (ns GMT) with 'time_ns_gmt' if present
        df = pd.read_csv(filepath).rename(columns={"time (ns GMT)": "time_ns_gmt"})

        return df


# ----- MAIN ----- #
# if __name__ == "__main__":
# Suppress warnings

# Create a Puffer sessions object
s = Sessions("../../data/Puffer/fake_data/")

# %%
