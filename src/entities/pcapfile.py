#  Author: Sahil Pattni
#  Copyright (c) Sahil Pattni 2023
import hashlib
import re
import os
from typing import List

import pandas as pd
import scapy
from scapy.layers.inet import IP, TCP, UDP
from scapy.utils import rdpcap


class PcapFile:
    """
    A class to represent a pcap file name's details.
    """

    @property
    def filepath(self):
        return self.__filepath

    @property
    def filename(self):
        return self.__filename

    @property
    def subdomain(self):
        # e.g. "www"
        return self.__subdomain

    @property
    def domain_name(self):
        # e.g. "google"
        return self.__domain_name

    @property
    def top_level_domain(self):
        # e.g. "com"
        return self.__top_level_domain

    # @property
    # def unknown(self):
    #     # e.g. "1" (unknown)
    #     return self.__unknown
    #
    # @property
    # def address(self):
    #     # ip address
    #     return self.__address

    def __init__(self, root: str, file: str):
        """
        Initialize the PcapFile class.

        Args:
            root (str): The root directory of the pcap file.
            file (str): The pcap file name.
        """
        self.__BASE_REGEX = r"(\w+).(\w+).(\w+)[\._].*.pcap"
        self.__extract_filename_details(root, file)

    def read_pcap_file(self) -> pd.DataFrame:
        """
        Read the pcap file and return a list of dictionaries containing the packets' data.

        Args:
            file_path (str): The directory to the pcap file.

        Returns:
            List[dict]: A list of dictionaries containing the packets' data.
        """
        packets: scapy.plist.PacketList = rdpcap(self.__filepath)
        data: List[dict] = []

        for packet in packets:
            if IP in packet:
                comm_protocol = TCP if TCP in packet else UDP if UDP in packet else None
                data.append(
                    {
                        "src": packet[IP].src,
                        "dst": packet[IP].dst,
                        "src_port": packet[comm_protocol].sport
                        if comm_protocol
                        else None,
                        "dst_port": packet[comm_protocol].dport
                        if comm_protocol
                        else None,
                        "proto": packet[IP].proto,
                        "size": packet[IP].len,
                        "time": packet[IP].time,
                        "provider": hash(self.__domain_name),
                        "tcp": 1 if TCP in packet else 0,
                        "udp": 1 if UDP in packet else 0,
                        "payload_size": len(packet.payload),
                        "fragmentation": packet.frag if packet.payload else 0,
                    }
                )

        df = pd.DataFrame(data)
        df = self.__preprocess(df)
        return df

    def __preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data.
        Args:
            df: The dataframe to preprocess.

        Returns:
            pd.DataFrame: The preprocessed dataframe.
        """
        # Convert 'time' column to float before converting to datetime
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df["tcp"] = df["tcp"].astype("uint8")
        df["udp"] = df["udp"].astype("uint8")
        # Apply the function to each row in the DataFrame to create a new 'session_uid' column
        df["session_uid"] = df.apply(self.__create_session_uid_hash, axis=1)

        return df

    def __create_session_uid_hash(self, row):
        # Create a concatenated string of the relevant columns
        combined_string = f"{row['src']}{row['dst']}{row['src_port']}{row['dst_port']}{row['provider']}"
        # Create a hash of this string
        return hashlib.md5(combined_string.encode()).hexdigest()

    def __extract_filename_details(self, root: str, filename: str):
        """
        Extract the details from the pcap file name.
        Args:
            root (str): The root directory of the pcap file.
            filename (str): The pcap file name.
        """
        self.__filename: str = filename
        self.__filepath: str = os.path.join(root, filename)

        match_groups = re.match(self.__BASE_REGEX, filename)

        if not match_groups:
            raise ValueError(f"Invalid file name: {filename}")

        self.__subdomain = match_groups.group(1)
        self.__domain_name = match_groups.group(2)
        self.__top_level_domain = match_groups.group(3)

    @staticmethod
    def is_pcap_file(self, filename: str) -> bool:
        """
        Check if the given file is a pcap file.

        Args:
            filename (str): The file name to check.

        Returns:
            bool: True if the file is a pcap file, False otherwise.
        """
        try:
            self.__extract_filename_details("", filename)
            return True
        except ValueError:
            return False
