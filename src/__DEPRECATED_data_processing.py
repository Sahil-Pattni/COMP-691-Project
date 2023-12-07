#  Author: Sahil Pattni
#  Copyright (c) Sahil Pattni 2023

# %%
"""
NOTE: This class has been deprecated. Please use src/data_processing.py instead.

A class to read in and process Puffer data.
"""

# %%
#


import warnings
import os
from typing import Callable

from src.utils.utils import get_prefix


from entities.pufferdata import PufferData

import pandas as pd


class DataProcesser:
    def __init__(
        self,
        input_directory: str,
        output_directory: str = "out",
        chunk_size: int = 100,
        limit: int = None,
    ):
        self.__input_directory = input_directory
        self.__output_directory = output_directory
        self.__chunk_size = chunk_size
        self.__puffer_data = PufferData(input_directory, chunk_size=self.__chunk_size)
        self.__limit = limit

        self.__video_data = None
        self.client_buffer = None

    def clear_output_directory(self):
        """
        Clears the output directory.
        """
        if os.path.exists(self.__output_directory):
            for file in os.listdir(self.__output_directory):
                if file.endswith(".parquet"):
                    os.remove(os.path.join(self.__output_directory, file))
            logger.info("Cleared output directory.")

    def run(self) -> None:
        """
        Runs the data processing pipeline.
        """
        logger.info("Starting data processing pipeline...")

        chunks_processed: int = 0

        try:
            while True:
                # Keep three checkpoints
                checkpoint = chunks_processed % 3 + 1
                self.__step()
                self.__process()
                self.__video_data.dropna(inplace=True)
                self.__align_video_data()
                self.__save(
                    self.__video_data,
                    title=f"processed_video_data_checkpoint_{checkpoint}",
                )

                chunks_processed += 1
                logger.info(
                    f"Processed {chunks_processed:,} chunk(s). Last checkpoint: {checkpoint}"
                )

                if self.__limit is not None and chunks_processed >= self.__limit:
                    logger.info(f"Reached limit of {self.__limit:,} chunks.")
                    return
        except StopIteration:
            logger.info("Reached end of file.")
            return

    def __step(self):
        self.__puffer_data.load_next_chunk()

    def export(self):
        network_metrics = [
            "time_ns_gmt_sent",
            "time_ns_gmt_acked",
            "rtt",
            "min_rtt",
            "delivery_rate",
            "in_flight",
            "size",
            "ssim_index",
        ]
        # TODO: Change from feather to something that can be loaded in chunks
        self.__video_data[network_metrics].to_feather(
            f"{self.__output_directory}/video_data.feather"
        )

    def load_from_file(self, title: str):
        self.__video_data = pd.read_parquet(
            f"{self.__output_directory}/{title}.parquet"
        )

    def __process(self):
        """
        Processes the current chunk and appends it to the dataframe.
        """

        self.__video_data = pd.merge(
            self.__puffer_data.video_acked,
            self.__puffer_data.video_sent,
            on=["session_id", "index", "video_ts"],
            how="inner",
            suffixes=("_acked", "_sent"),
        )

        # Sort columns
        self.__video_data = self.__video_data.reindex(
            sorted(self.__video_data.columns), axis=1
        )

        # Why did I put this here?
        self.client_buffer = self.__puffer_data.client_buffer

        self.__video_data.dropna(inplace=True)

        # Sort columns (Deprecated)
        # self.__video_data.reindex(sorted(self.__video_data.columns), axis=1)

    def calculate_similarity_score(self, idx, offset, _df, forward=True):
        ack_idx = (idx + offset) % _df.shape[0]
        sent_idx = idx
        if not forward:
            ack_idx = idx
            sent_idx = (idx + offset) % _df.shape[0]
        sent = _df.iloc[sent_idx]["time_ns_gmt_sent"]
        ack = _df.iloc[ack_idx]["time_ns_gmt_acked"]
        matchers = ["session_id", "eventHow "]
        if ack > sent:
            time_difference = (
                pd.Timedelta(ack - sent).total_seconds() * 1e6
            )  # Convert to microseconds
            score = -abs(
                time_difference - _df.iloc[idx]["rtt"]
            )  # Negative score for smaller differences
            return score
        else:
            logger.debug(f"ACK ({ack}) is not after sent ({sent})")
            return -float(
                "inf"
            )  # Return a very low score if ack time is not after sent time

    def __align_video_data(
        self,
        window_size: int = 1000,
        n_steps: int = 1000,
        start_index: int = 0,
    ):
        # Return a very low score if ack time is not after sent time

        # Should never run
        if self.__video_data is None:
            raise ValueError("Video data is None.")

        # Input sanitization
        start_index = min(max(start_index, 0), len(self.__video_data) - 1)
        n_rows: int = len(self.__video_data)
        n_steps = min(n_steps, n_rows)  # Ensure n_steps is not greater than n_rows
        window_size = min(
            window_size, n_rows
        )  # Ensure window_size is not greater than n_rows

        start = start_index
        end = (start_index + window_size) % n_rows
        # Swap start and end if end is less than start
        if end < start:
            start = end
        window = self.__video_data.iloc[start:end]

        best_score = -float("inf")
        best_alignment = None
        original_score = None

        for offset in range(n_steps):
            score = self.calculate_similarity_score(start_index, offset, window)
            logger.debug(f"Testing offset: {offset:,} (score: {score:,.4f})")
            if original_score is None:
                original_score = score
                best_alignment = offset

            if score > best_score:
                best_score = score
                best_alignment = offset

        logger.info(
            f"Best alignment: {best_alignment:,} (score: {best_score})(original score: {original_score})(Difference: {original_score - best_score})"
        )

        # Align entire dataframe
        # self.__video_data["time_ns_gmt_acked"] = self.__video_data["time_ns_gmt_acked"].shift(best_alignment)

    def __save(self, df: pd.DataFrame, title: str):
        """
        Saves the current chunk as a parquet file,
        or appends to an existing parquet file.

        Args:
            df (pd.DataFrame): Dataframe to save.
            title (str): Name of the file to save to.
        """
        filename = f"{self.__output_directory}/{title}.parquet"

        if not os.path.exists(self.__output_directory):
            os.makedirs(self.__output_directory)

        df.to_parquet(
            filename,
            engine="fastparquet",
            compression="gzip",
            index=False,
            # Disables this if the file doesn't exist
            append=os.path.isfile(filename),
        )
        logger.debug(f"Saved {title} to {filename}.")

    @property
    def puffer_data(self) -> PufferData:
        return self.__puffer_data

    @property
    def video_data(self) -> pd.DataFrame:
        return self.__video_data


if __name__ == "__main__":
    LOAD = False
    dp = DataProcesser(
        input_directory=f"{get_prefix(True)}data/Puffer/2023-12-02T2023-12-03",
        output_directory=f"{get_prefix(True)}out",
        chunk_size=1000,
        limit=1,
    )

    if LOAD:
        logger.info("Loading from file...")
        dp.load_from_file("processed_video_data_checkpoint_1")
        logger.info("Loaded from file.")

    else:
        pass
        dp.clear_output_directory()  # Fresh start
        dp.run()

        p = dp.puffer_data
        df = dp.video_data
