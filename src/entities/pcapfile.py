#  Author: Sahil Pattni
#  Copyright (c) Sahil Pattni 2023
import re
from typing import List

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
    def subdomain(self):
        return self.__subdomain

    @property
    def domain_name(self):
        return self.__domain_name

    @property
    def top_level_domain(self):
        return self.__top_level_domain

    @property
    def unknown(self):
        return self.__unknown

    @property
    def address(self):
        return self.__address

    def __init__(self, filepath: str):
        self.__filepath = filepath

        def read_pcap_file(self, file_path: str) -> List[dict]:
            """
            Read the pcap file and return a list of dictionaries containing the packets' data.

            Args:
                file_path (str): The path to the pcap file.

            Returns:
                List[dict]: A list of dictionaries containing the packets' data.
            """
            packets: scapy.plist.PacketList = rdpcap(file_path)
            data: List[dict] = []
            for packet in packets:
                if IP in packet:
                    src = packet[IP].src
                    dst = packet[IP].dst
                    proto = packet[IP].proto
                    size = packet[IP].len
                    time = packet.time

                    if TCP in packet:
                        s_port = packet[TCP].sport
                        d_port = packet[TCP].dport
                    elif UDP in packet:
                        s_port = packet[UDP].sport
                        d_port = packet[UDP].dport
                    else:
                        s_port = 0
                        d_port = 0
                    data.append(
                        {
                            "packet": packet,
                            "src": src,
                            "dst": dst,
                            "s_port": s_port,
                            "d_port": d_port,
                            "proto": proto,
                            "size": size,
                            "time": time,
                        }
                    )
            return data

    def __generate_regex(self):
        # Default regex to get all .pcap files
        subdomain_regex: str = r"\w+"  # e.g. "www"
        domain_name_regex: str = r"\w+"  # e.g. "google"
        top_level_domain_regex: str = r"\w+"  # e.g. "com"
        unknown_regex: str = r"[\d]+"  # e.g. "1" (idk what this is)
        address_regex: str = r"[\d\.]+"  # irrelevant for now

        self.__regex = rf"({subdomain_regex}).({domain_name_regex}).{top_level_domain_regex}_({unknown_regex})_({address_regex}).pcap"

        match_groups = re.match(self.__regex, self.__filepath)
        self.__subdomain = match_groups.group(1)
        self.__domain_name = match_groups.group(2)
        self.__top_level_domain = match_groups.group(3)
        self.__unknown = match_groups.group(4)
        self.__address = match_groups.group(5)
