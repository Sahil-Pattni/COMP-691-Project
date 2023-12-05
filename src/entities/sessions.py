# %%

"""
A class that represents a Puffer session, including 
both client-side and server-side data.

Author: Sahil Pattni

Copyright (c) Sahil Pattni
"""
import logging
from typing import List
import pandas as pd

import re
import os

import utils.logger_config

from src.utils.logger_config import LoggerCustom

logger = LoggerCustom.get_logger("Sessions", level=logging.DEBUG)


class Sessions:
    # Attributes of a Puffer session.
    attributes = {
        # Unique identifier for each video session, used to track and analyze individual
        # streaming experiences.
        "session_id"
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
    }

    def __init__(self, directory: str, **kwargs):
        """
        Initialize a Puffer session.

        Args:
            directory (str, optional): Directory to load Puffer sessions
            data from.
        """
        self.__files: dict = self.__find_files(directory)
        self.__load(**kwargs)

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

    # Setters
    @client_buffer.setter
    def client_buffer(self, client_buffer: pd.DataFrame):
        self.__client_buffer = client_buffer

    @video_acked.setter
    def video_acked(self, video_acked: pd.DataFrame):
        self.__video_acked = video_acked

    @video_sent.setter
    def video_sent(self, video_sent: pd.DataFrame):
        self.__video_sent = video_sent

    @video_size.setter
    def video_size(self, video_size: pd.DataFrame):
        self.__video_size = video_size

    @ssim.setter
    def ssim(self, ssim: pd.DataFrame):
        self.__ssim = ssim

    def get_all_data(self, index: int = None) -> List:  # type: ignore
        """
        Get all Puffer sessions data, or the data at a specific index.
        The order of the data is as follows:
            0. Client buffer
            1. Video acked
            2. Video sent
            3. Video size
            4. SSIM

        Args:
            index (int, optional): Index of the data to return. Defaults to None.

        Returns:
            List: List of Puffer sessions' data.

        Raises:
            ValueError: If the index is invalid.
        """
        data_mapping = {
            0: self.__client_buffer,
            1: self.__video_acked,
            2: self.__video_sent,
            3: self.__video_size,
            4: self.__ssim,
        }

        if index is None:
            return [data_mapping[i] for i in range(len(data_mapping))]
        else:
            if index not in data_mapping.keys():
                raise ValueError(f"Invalid index: {index}")
            return [data_mapping[index]]

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
            # Has to be a CSV file
            if not file.endswith(".csv"):
                continue
            if re.search(f"client_buffer{base_regex}", file):
                results["client_buffer"] = os.path.join(directory, file)
                logger.debug(f"Found client buffer file: {file}")
            elif re.search(f"video_acked{base_regex}", file):
                results["video_acked"] = os.path.join(directory, file)
                logger.debug(f"Found video acked file: {file}")
            elif re.search(f"video_sent{base_regex}", file):
                results["video_sent"] = os.path.join(directory, file)
                logger.debug(f"Found video sent file: {file}")
            elif re.search(f"video_size{base_regex}", file):
                results["video_size"] = os.path.join(directory, file)
                logger.debug(f"Found video size file: {file}")
            elif re.search(f"ssim{base_regex}", file):
                results["ssim"] = os.path.join(directory, file)
                logger.debug(f"Found SSIM file: {file}")

        return results

    def __load(self, **kwargs) -> None:  # type: ignore
        """
        Load a Puffer sessions dataset from a CSV file.

        Args:
            limit (int, optional): Limit the number of rows to load. Defaults to None.

        Returns:
            pd.DataFrame: Pandas DataFrame containing Puffer sessions data.
        """
        # Burn last
        self.__client_buffer: pd.DataFrame = self.__load_client_buffer(**kwargs)
        self.__video_acked: pd.DataFrame = self.__load_video_acked(**kwargs)
        self.__video_sent: pd.DataFrame = self.__load_video_sent(**kwargs)
        self.__video_size: pd.DataFrame = self.__load_video_size(**kwargs)
        self.__ssim: pd.DataFrame = self.__load_ssim(**kwargs)
        pass

    def __cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean up a Puffer session's dataset.
        Steps:
            1. Remove rows with null values from a Pandas DataFrame.
            2. Sort the DataFrame by time_ns_gmt and then by session_id.
            3. Convert time data to datetime objects.
            3. Resamples the DataFrame into 1S intervals.


        Args:
            df (pd.DataFrame): Pandas DataFrame to clean up.

        Returns:
            pd.DataFrame: Cleaned up Pandas DataFrame.
        """
        # Step 1 : Remove rows with null values
        old_rows: int = df.shape[0]
        df = df.dropna()
        logger.debug(f"Removed {old_rows - df.shape[0]} rows with null values.")

        # Step 2 : Sort the DataFrame by time_ns_gmt and then by session_id
        columns = [
            column for column in ["time_ns_gmt", "session_id"] if column in df.columns
        ]
        logger.debug(f"Sorting DataFrame by columns: {columns}")
        df = df.sort_values(by=columns)

        # Step 3 : Convert datetime objects
        for _col in ["time_ns_gmt"]:
            if _col in df.columns:
                df[_col] = pd.to_datetime(df[_col])

        # Step 4 : Resample the DataFrame into 1S intervals
        df = (
            df.resample(on="time_ns_gmt", rule="1S")
            .max()
            .dropna()
            .sort_values(
                by=[
                    col
                    for col in ["time_ns_gmt", "session_id", "index"]
                    if col in df.columns
                ]
            )
        )

        return df

    def __load_client_buffer(self, **kwargs) -> pd.DataFrame:
        """
        Load a Puffer client buffer dataset from a CSV file.

        Returns:
            pd.DataFrame: Pandas DataFrame containing Puffer client buffer data.
        """

        return self.__read_csv(self.__files["client_buffer"], **kwargs)

    def __load_video_acked(self, **kwargs) -> pd.DataFrame:
        """
        Load a Puffer video acked dataset from a CSV file.

        Returns:
            pd.DataFrame: Pandas DataFrame containing Puffer video acked data.
        """

        return self.__read_csv(self.__files["video_acked"], **kwargs)

    def __load_video_sent(self, **kwargs) -> pd.DataFrame:
        """
        Load a Puffer video sent dataset from a CSV file.

        Returns:
            pd.DataFrame: Pandas DataFrame containing Puffer video sent data.
        """

        return self.__read_csv(self.__files["video_sent"], **kwargs)

    def __load_video_size(self, **kwargs) -> pd.DataFrame:
        """
        Load a Puffer video size dataset from a CSV file.

        Returns:
            pd.DataFrame: Pandas DataFrame containing Puffer video size data.
        """
        return self.__read_csv(self.__files["video_size"], **kwargs)

    def __load_ssim(self, **kwargs) -> pd.DataFrame:
        """
        Load a Puffer SSIM dataset from a CSV file.

        Args:
            filepath (str): Filepath of the CSV file to load.

        Returns:
            pd.DataFrame: Pandas DataFrame containing Puffer SSIM data.
        """

        return self.__read_csv(self.__files["ssim"], **kwargs)

    def __read_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Read a CSV file into a Pandas DataFrame.

        Args:
            filepath (str): Filepath of the CSV file to read.

        Returns:
            pd.DataFrame: Pandas DataFrame containing CSV data.
        """

        # Replace time (ns GMT) with 'time_ns_gmt' if present
        df = pd.read_csv(filepath, **kwargs).rename(
            columns={"time (ns GMT)": "time_ns_gmt"}
        )

        return self.__cleanup(df)


# %%
