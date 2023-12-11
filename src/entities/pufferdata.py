# %%

"""
A class that represents a Puffer sessions, including
both client-side and server-side data.

Author: Sahil Pattni

Copyright (c) Sahil Pattni
"""
import hashlib
from typing import List
import pandas as pd
import numpy as np
from pandas.io.parsers.readers import TextFileReader
import re
import os


from src.utils.logger_config import LoggerCustom

logger = LoggerCustom.get_logger()


class PufferData:
    """
    A class that represents a Puffer sessions, including
    both client-side and server-side data.

    The following fields are across the DataFrames:
        time: Timestamp (nanoseconds since Unix epoch) when the chunk is sent.
        session_id: Unique ID for the video sessions.
        index: Index field used to group streams within the same sessions.
        expt_id: Unique ID identifying information associated with a 'scheme'.
        channel: TV channel name, indicating a different stream for the same session_id.
        video_ts: Presentation timestamp of the chunk.
        format: Encoding settings of the chunk, represented as 'WxH-CRF'.
        size: Chunk size in bytes.
        ssim_index: SSIM of the chunk relative to a canonical version of the chunk.
        cwnd: Congestion window size in packets.
        in_flight: Number of unacknowledged packets in flight.
        min_rtt: Minimum RTT in microseconds.
        rtt: Smoothed RTT estimate in microseconds.
        delivery_rate: TCP's estimation of delivery rate in bytes/second.
        buffer: Playback buffer size in seconds.
        cum_rebuf: Total time in seconds spent rebuffering in the current stream.
    """

    @staticmethod
    def __preprocess(func):
        """
        Decorator to clean up and pre-process the dataframes after reading
        from CSV files.

        Returns:
            pd.DataFrame: Cleaned up DataFrame.
        """

        def wrapper(self, *args, **kwargs):
            df = func(self, *args, **kwargs)

            # -- Step : Remove rows with null values -- #
            old_rows: int = df.shape[0]
            df = df.dropna()
            logger.debug(f"Removed {old_rows - df.shape[0]} rows with null values.")

            # -- Step : Rename time column -- #
            df.rename(
                columns={"time (ns GMT)": "time_ns_gmt"},
                inplace=True,
            )
            # -- Step : Convert datetime objects -- #
            self.__convert_to_datetime(df)

            # -- Step : Split format column into width, height, and fps columns -- #
            self.__decompose_format(df)

            # -- Step : Convert columns to int64 -- #
            self.__convert_to_Int64(df)

            # -- Step : Sort the data -- #
            self.__sort_data(df)

            # -- Step (Deprecated) : One hot encode the columns -- #
            # This is now hashed instead.
            # self.__one_hot_encode(train_df)

            # -- Step : Hash the session_id column -- #
            self.__hash(df)

            # -- Step (Deprecated) : Resample the DataFrame into 1S intervals -- #
            # train_df = train_df.resample(on="time_ns_gmt", rule="1S").mean().dropna()

            return df.dropna()

        return wrapper

    def __init__(self, directory: str, chunk_size: int):
        """
        Initialize a Puffer sessions.

        Args:
            directory (str, optional): Directory to load Puffer sessions
            data from.
            chunk_size (int, optional): Number of rows to load at a time. Defaults to 100.
        """
        self.__files: dict = self.__find_files(directory)
        self.chunk_size = chunk_size
        self.__init_dispensers()

    # ----- PUBLIC METHODS ----- #
    def load_next_chunk(self) -> None:  # type: ignore
        """
        Load a Puffer sessions dataset from a CSV file.

        Args:
            limit (int, optional): Limit the number of rows to load. Defaults to None.

        Returns:
            pd.DataFrame: Pandas DataFrame containing Puffer sessions data.

        Raises:
            StopIteration: If there are no more chunks to load.
        """
        self.client_buffer = self.__step(self.__client_buffer_dispenser)
        logger.debug(f"Loaded client buffer chunk: {self.client_buffer.shape}")

        self.video_acked = self.__step(self.__video_acked_dispenser)
        logger.debug(f"Loaded video acked chunk: {self.video_acked.shape}")

        self.video_sent = self.__step(self.__video_sent_dispenser)
        logger.debug(f"Loaded video sent chunk: {self.video_sent.shape}")

        self.video_size = self.__step(self.__video_size_dispenser)
        logger.debug(f"Loaded video size chunk: {self.video_size.shape}")

        self.ssim = self.__step(self.__ssim_dispenser)
        logger.debug(f"Loaded ssim chunk: {self.ssim.shape}")

    # ----- PRIVATE METHODS ----- #

    @__preprocess
    def __step(self, dispenser: TextFileReader) -> pd.DataFrame:
        return next(dispenser)

    def __init_dispensers(self):
        self.__client_buffer_dispenser = self.__init_csv_chunk_dispenser(
            self.__files["client_buffer"],
            chunk_size=self.chunk_size,
        )
        logger.debug("Initialized client buffer dispenser")

        self.__video_acked_dispenser = self.__init_csv_chunk_dispenser(
            self.__files["video_acked"],
            chunk_size=self.chunk_size,
        )
        logger.debug("Initialized video acked dispenser")

        self.__video_sent_dispenser = self.__init_csv_chunk_dispenser(
            self.__files["video_sent"],
            chunk_size=self.chunk_size,
        )
        logger.debug("Initialized video sent dispenser")

        self.__video_size_dispenser = self.__init_csv_chunk_dispenser(
            self.__files["video_size"],
            chunk_size=self.chunk_size,
        )
        logger.debug("Initialized video size dispenser")

        self.__ssim_dispenser = self.__init_csv_chunk_dispenser(
            self.__files["ssim"],
            chunk_size=self.chunk_size,
        )
        logger.debug("Initialized SSIM dispenser")

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

    def __one_hot_encode(self, df):
        categorical_columns = [_c for _c in ["channel"] if _c in df.columns]
        new_cols = pd.get_dummies(df[categorical_columns], dtype=np.int8)

        # Add the OHE columns to the DataFrame
        for column in new_cols:
            df[column] = new_cols[column]

        # Drop the original categorical columns
        df.drop(columns=categorical_columns, inplace=True)

    def __hash(self, df):
        def hash256(x) -> int:
            hasher = hashlib.sha256()
            hasher.update(str(x).encode("utf-8"))
            return int(hasher.hexdigest(), 16) % 2**64

        _cols = [c for c in ["session_id", "event", "channel"] if c in df.columns]
        for column in _cols:
            df[column] = df[column].apply(lambda x: hash256(x)).astype(np.int64)

    def __sort_data(self, df):
        if "time_ns_gmt" in df.columns:
            df.sort_values(by=["time_ns_gmt"], inplace=True)

    def __convert_to_Int64(self, df):
        _columns_to_convert = [
            "expt_id",
            "size",
            "cwnd",
            "in_flight",
            "rtt",
            "min_rtt",
            "width",
            "height",
            "fps",
        ]

        for _col in _columns_to_convert:
            if _col in df.columns:
                logger.debug(
                    f"Converting column `{_col}` from {df[_col].dtype} to `int64`."
                )
                df[_col] = df[_col].astype("Int64")

    def __convert_to_datetime(self, df):
        if "time_ns_gmt" in df.columns:
            df["time_ns_gmt"] = pd.to_datetime(df["time_ns_gmt"])

    def __decompose_format(self, df):
        if "format" in df.columns:
            pattern = r"(\d+)x(\d+)-(\d+)"
            df[["width", "height", "fps"]] = df["format"].str.extract(pattern)
            df.drop(columns=["format"], inplace=True)

    def __init_csv_chunk_dispenser(
        self, filepath: str, chunk_size: int
    ) -> TextFileReader:
        """
        Read a CSV file into a Pandas DataFrame.

        Args:
            filepath (str): Filepath of the CSV file to read.

        Returns:
            pd.DataFrame: Pandas DataFrame containing CSV data.
        """
        # Return a generator of DataFrames
        return pd.read_csv(filepath, chunksize=chunk_size)

    # ----- PROPERTIES ----- #
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
