#  Author: Sahil Pattni
#  Copyright (c) Sahil Pattni 2023

# %%
"""
A class to read in and process Puffer data.


"""

# %%
#


import warnings
import os

import numpy as np

from src.utils.utils import get_prefix

warnings.filterwarnings("ignore")

import logging
from utils.logger_config import LoggerCustom

logger = LoggerCustom.get_logger("Data Processing", level=logging.INFO)

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
                self.__align_df()
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

    def load_from_file(self, title: str):
        self.__video_data = pd.read_parquet(
            f"{self.__output_directory}/{title}.parquet"
        )

    def __process(self):
        """
        Processes the current chunk and appends it to the dataframe.
        """
        common = self.__puffer_data.video_acked.columns.intersection(
            self.__puffer_data.video_sent.columns
        ).drop(["video_ts", "time_ns_gmt"])

        self.__video_data = self.__puffer_data.video_acked.set_index("video_ts").join(
            self.__puffer_data.video_sent.set_index("video_ts", drop=False).drop(
                columns=common
            ),
            on="video_ts",
            lsuffix="_acked",
            rsuffix="_sent",
        )

        self.client_buffer = self.__puffer_data.client_buffer

        self.__video_data.dropna(inplace=True)

        # Sort columns (Deprecated)
        # self.__video_data.reindex(sorted(self.__video_data.columns), axis=1)

    import pandas as pd
    import numpy as np

    def __align_video_data(self):
        if self.__video_data is None:
            raise ValueError("Video data is None.")

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

    def __align_df(self):
        """
        Aligns the dataframe with the current chunk.
        """
        # TODO: Implement this

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
        chunk_size=10000,
        limit=1,
    )

    if LOAD:
        logger.info("Loading from file...")
        dp.load_from_file("processed_video_data_checkpoint_3")
        logger.info("Loaded from file.")

    else:
        pass
        dp.clear_output_directory()  # Fresh start
        dp.run()

        p = dp.puffer_data
        df = dp.video_data
        df.join(
            dp.client_buffer.set_index("session_id"),
            on=["session_id", "index"],
            how="inner",
        ).drop(columns=df.columns.intersection(p.client_buffer.columns))
