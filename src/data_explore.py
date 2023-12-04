# %%
import logging
from typing import List
from tqdm import tqdm
from scapy.all import TCP, UDP, PacketList
import utils
from scapy.all import rdpcap as read_pcap
import pyarrow.feather as ftr


def compile_data(filepaths: List[str], limit: int = None) -> dict:
    """
    Compile data from a list of filepaths into a dictionary of datasets.

    Args:
        filepaths (List[str]): List of filepaths to compile data from
        limit (int, optional): Limit the number of files to compile. Defaults to None.

    Returns:
        dict: Dictionary of datasets.
    """

    results: dict = {}

    iterations: int = 0

    for file in tqdm(filepaths):
        dataset = {}
        logging.info(f"Reading file: {file}")
        dataset["packets"]: PacketList = read_pcap(file)
        provider: str = utils.extract_provider(file)

        # Attach by provider
        results[provider] = results.get(provider, []) + [dataset]

        iterations += 1
        if limit and iterations >= limit:
            break

    return results


# %%
file = utils.get_dataset_filepaths()[0]
packet_log = utils.read_pcap(file)
# %%


if __name__ == "__main__":
    # Suppress warnings
    import warnings

    warnings.filterwarnings("ignore")

    # Set logging level to ONLY show errors
    logging.basicConfig(level=logging.ERROR)

    data = compile_data(utils.get_dataset_filepaths(limit=3))
