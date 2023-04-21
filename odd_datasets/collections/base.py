import sys
import json
import typer
import logging
import datetime
import subprocess
import numpy as np
import pandas as pd
import datamol as dm

from typing import Optional, Union
from loguru import logger
from openff.units import unit
from openff.qcsubmit.factories import OptimizationDatasetFactory
import qcfractal.interface as interface

# disable openff-toolkit warnings
logging.getLogger("openff.toolkit").setLevel(logging.ERROR)


def parse_server_info(server_info_file=None):
    if server_info_file is not None:
        with open(server_info_file, "r") as f:
            server_info = json.load(f)
    else:
        server_info = {"address": "localhost:7777", "verify": False}

    address = server_info["address"]
    verify = server_info.get("verify", False)
    username = server_info.get("username", None)
    password = server_info.get("password", None)

    return dict(
        address=address, 
        verify=verify, 
        username=username, 
        password=password
    )

class CollectionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def check_computation_status(server_info_file=None):
    server_info = parse_server_info(server_info_file)
    client = interface.FractalClient(**server_info)

    # res = client.get_collection("OptimizationDataset", 
    #                             "qmodd_semiempirical_optimization_geometry_dataset")
    # print(res.)
    completed = client.query_procedures()
    results = list(map(lambda x: x.dict(), completed))
    with open("qmodd_semi_geo_dataset.json", "w") as f:
        json.dump(results, f, indent=2, sort_keys=True, cls=CollectionEncoder)
    # print(res)


def geometry_optimization_specification():
    """ Specification for semi-empirical geometry optimization """
    # Create the spec
    qc_specifications = QCSpec(
        method="GFN2-xTB",
        program="xtb",
        basis=None,
        spec_name="odd_se_geometry",
        spec_description="ODD SE geometry optimization",
        maxiter=200,
        keywords={
            "wcombine": True,
            "scf_type": "df",
            "accuracy": 1.0,
            "electronic_temperature": 300.0,
            "max_iterations": 50,
            "solvent": "none", # "water",
        }
    )

    return qc_specifications


class QCFractalOptDataset:
    """
    Base class for datasets of molecular conformers opimized at the semi-empirical level"""

    def __init__(
        self,
        server_info_file: Union[str, None] = None,        
    ):
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        dm.disable_rdkit_log()

        res = subprocess.run(["git", "config", "user.name"], stdout=subprocess.PIPE)
        self._git_username = res.stdout.strip().decode()

        self._parse_server_info(server_info_file)
        self.connect_to_server()
            
    
    def connect_to_server(self):
        """Connect to the server"""
        try:
            client = interface.FractalClient(**self.server_info)
        except ConnectionRefusedError as error:
            if self.server_info["verify"]:
                raise typer.BadParameter("Connection refused. Try passing the --no-verify flag") from error
            raise error
        self.computation_server = client

    def _parse_server_info(self, server_info_file=None):
        self.server_info = parse_server_info(server_info_file)


    def submit(self, mol_with_conformers, collection_name, tagline, description):
        """
        Create a dataset and submit to the server
        This does not yet run any QM computations. It just preprocesses the molecular structures (e.g. deduplication).
        """  

        # Create the factory
        spec = geometry_optimization_specification()
        factory = OptimizationDatasetFactory(
            qc_specifications={spec.spec_name: spec},
        )
        logger.info(str(factory.dict()))

        # Create the dataset
        self._dataset = factory.create_dataset(
            dataset_name=collection_name,
            molecules=mol_with_conformers,
            tagline=tagline,
            description=description,
            verbose=True,
        )

        self._dataset.metadata.submitter = self._git_username
        
        responses = self._dataset.submit(self.computation_server, verbose=True)
        logger.info(f"Submitted {len(responses)} tasks to {self.server_info['address']}")
        return responses

