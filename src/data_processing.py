# %%
"""
A class to read in and process Puffer data.
"""

# Create a Puffer sessions object
from entities.sessions import Sessions
from utils.utils import get_prefix

import warnings

warnings.filterwarnings("ignore")

from utils.logger_config import setup_logger

logger = setup_logger(__name__)

s = Sessions(f"{get_prefix(flag=False)}data/Puffer/fake_data/")

# %%
