import os
import json
import tqdm
import fsspec
import pandas as pd
import numpy as np
import datamol as dm
from loguru import logger
from rdkit import Chem
from openff.toolkit.typing.engines.smirnoff import ForceField
from odd_datasets.collections.base import ConformersDataset
from openff.toolkit.topology import Molecule, Topology


class RefinedSetProteinLigand(ConformersDataset):
    
    def __init__(self, server_info_file: Union[str, None] = None):
        super().__init__(server_info_file)

    def load_refined_set():
        