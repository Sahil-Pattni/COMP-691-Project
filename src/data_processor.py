#  Author: Sahil Pattni
#  Copyright (c) Sahil Pattni 2023
import pickle
import warnings

from entities.pcapfile import PcapFile
from utils.logger_config import LoggerCustom

warnings.filterwarnings("ignore")

# Import the necessary libraries to read .pcap files
from scapy.all import *

import pandas as pd
import os
import re

logger = LoggerCustom.get_logger()


class DataProcessor:
    @property
    def chunk_size(self) -> int:
        return self.__chunk_size

    @chunk_size.setter
    def chunk_size(self, value: int):
        self.__chunk_size = value

    def __init__(self, chunk_size: int = 10000):
        self.__chunk_size = chunk_size
        self.__chunk_number = 1

    # ----- PUBLIC METHODS ----- #

    def run(
        self,
        input_directory: str,
        output_directory: str,
        n_chunks=float("inf"),
        append=True,
    ) -> None:
        """

        Args:
            input_directory (str): The directory containing the .pcap files
            output_directory (str): The directory to save the output files
            n_chunks (int): The number of chunks to process. Defaults to infinity, which processes all chunks.
        """
        logger.info("Starting Data processing...")
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        if not append:
            self.__clear(output_directory)
        self.__load_from_directory(output_directory)

        chunk_df: pd.DataFrame = None
        num_chunks_processed: int = 0
        processed_files = set()
        for root, dirs, files in os.walk(input_directory):
            for idx, file in enumerate(files):
                # Skip if the file has already been processed
                if file in self.__processed_files:
                    logger.debug(f"Skipping file: {file}")
                    continue
                try:
                    # If invalid filename, this will throw an exception
                    current_df: pd.DataFrame = PcapFile(root, file).read_pcap_file()
                    if chunk_df is None:
                        chunk_df = current_df
                    else:
                        chunk_df = pd.concat([chunk_df, current_df], ignore_index=True)
                        logger.debug(f"Concatenated file: {file}")

                    processed_files.add(file)

                    # If the chunk size is reached, save the chunk and begin a new one
                    if len(chunk_df) >= self.__chunk_size:
                        chunk_df = self.__save(
                            chunk_df, processed_files, output_directory
                        )
                        num_chunks_processed += 1
                        self.__chunk_number += 1

                    # If the number of chunks processed is reached, stop
                    if num_chunks_processed >= n_chunks or idx == len(files) - 1:
                        logger.info(
                            f"Processed {num_chunks_processed:,} chunks. Stopping..."
                        )
                        if chunk_df is not None:
                            self.__save(chunk_df, processed_files, output_directory)
                        break

                except ValueError as e:
                    logger.error(f"Error while processing file: {file}, error: {e}")

    def __save(self, chunk_df, files_processed, output_directory):
        logger.debug(f"Saving chunk #{self.__chunk_number:,}")

        self.__processed_files = self.__processed_files.union(files_processed)
        # Save the processed files
        with open(f"{output_directory}/processed_files.pickle", "wb") as f:
            pickle.dump(self.__processed_files, f)

        return self.__save_df(chunk_df, output_directory)

    def __save_df(self, chunk_df, output_directory):
        chunk_df.to_feather(f"{output_directory}/chunk_{self.__chunk_number}.ftr")
        chunk_df = None  # Free up memory
        return chunk_df

    def __load_from_directory(self, output_directory: str):
        """
        Finds the last chunk number in the output directory

        Returns:
            int: The last chunk number
        """
        # Get the list of files in the output directory
        files: List[str] = os.listdir(output_directory)
        self.__chunk_number = self.__get_last_chunk_number(files) + 1

        try:
            with open(f"{output_directory}/processed_files.pickle", "rb") as f:
                self.__processed_files = pickle.load(f)
        except FileNotFoundError:
            logger.debug("No processed_files.pickle file found. Creating new one...")
            self.__processed_files = set()

    def __get_last_chunk_number(self, files: List[str]) -> int:
        files = [file for file in files if file.endswith(".ftr")]

        if len(files) == 0:
            return 0

        # Get the list of chunk numbers
        chunk_numbers: List[int] = [
            int(re.match(r"chunk_(\d+).ftr", file).group(1)) for file in files
        ]

        # Return the max chunk number
        return max(chunk_numbers) if len(chunk_numbers) > 0 else 0

    def __clear(self, output_directory: str) -> None:
        """
        Clears the output directory.

        Args:
            output_directory (str): The directory to clear.
        """
        if os.path.exists(output_directory):
            for file in os.listdir(output_directory):
                os.remove(os.path.join(output_directory, file))
            logger.info("Cleared output directory.")
