'''

A set of functions designed for extracting session-specific information
from the Allen Institute Visual Behavior Neuropixels Dataset

'''

import pandas as pd
import numpy as np

def extract_units(session : pd.DataFrame):
    # Extract units
    units = session.get_units()


def extract_channels(session : pd.DataFrame):
    # Extract channels
    channels = session.get_channels()


def extract_probes(session : pd.DataFrame):
    return

def extract_behavior(session : pd.DataFrame):
    return

