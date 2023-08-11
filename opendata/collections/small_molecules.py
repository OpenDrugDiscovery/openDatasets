import os
import time
import click
import dotenv
import random
import numpy as np
import pandas as pd
import datamol as dm

import qcelemental as qcel
from loguru import logger
from qcportal import PortalClient
from openff.toolkit import Molecule
from qcportal.record_models import PriorityEnum
from rdkit.Chem.MolStandardize import rdMolStandardize
from qcportal.singlepoint.dataset_models import SinglepointDatasetNewEntry
from qcportal.singlepoint.record_models import QCSpecification
from qcportal.singlepoint.record_models import SinglepointDriver
from datamol.descriptors import compute_many_descriptors, n_stereo_centers_unspecified
from opendata.preprocessing.small_molecules import load_fragment_collection
from opendata.conformers.generator import ConformerGenerator
from opendata.collections.specs import get_specs


def submit_smiles(smi, client, dataset_name, use_replica_exchange=True):
    mol = dm.to_mol(smi)
    inchikey = dm.to_inchikey(mol)
    props = compute_many_descriptors(mol, add_properties=False)

    omol = ConformerGenerator(use_replica_exchange=use_replica_exchange).from_smiles(smi)
    _create_entry = lambda conf: SinglepointDatasetNewEntry(
            name=inchikey, 
            molecule=conf, 
            attributes=props, 
            comment=None)
              
    # Build the entries
    if omol is not None:
        entries = [_create_entry(omol.to_qcschema(conformer=c)) 
                for c in np.arange(omol.n_conformers)]
    else:
        entries = []

    submit_sp_collection(client=client,
                         dataset_name=dataset_name,
                         entries=entries,
                         spec_name="gaz_phase_se",
                         tag='sm_se')

    submit_sp_collection(client=client,
                         dataset_name=dataset_name,
                         entries=entries,
                         spec_name="gaz_phase_dft",
                         tag='sm_dft')
    return entries


def submit_sp_collection(client, dataset_name, entries, dataset_kwargs, spec_name, tag):
    dataset_type = "singlepoint"
    ds_list = client.list_datasets()
    dataset_exits = any([(x["dataset_type"] == dataset_name) for x in ds_list])
    if not dataset_exits:
        client.add_dataset(dataset_type=dataset_type, name=dataset_name, **dataset_kwargs)

    ds = client.get_dataset(dataset_type, dataset_name)
    if spec_name not in ds.specification_names:
        spec = get_specs(spec_name)
        client.add_specification(name=spec_name, specification=spec)
    ds.add_entries(entries)
    ds.submit(tag=tag)


@click()
@click.option("--chunk-id", "-i", type=int, default=0, help="chunk id starting at 0")
@click.option("--chunk-size", "-s", type=int, default=1000, help="Chunk size to divide and conquer.")
def submit_small_molecules(chunk_id, chunk_size):
    client = PortalClient(
        address="https://openfractal-test-pgzbs3yryq-uc.a.run.app",
        username=os.environ["OPENFRACTAL_USERNAME"],
        password=os.environ["OPENFRACTAL_PASSWORD"],
    )
    np.random.seed(42)
    res = load_fragment_collection(include_iso_tauto=True)
    all_smiles = list(res.keys())
    idxs = np.arange(len(all_smiles))
    np.random.shuffle(idxs)
    all_smiles = [all_smiles[i] for i in idxs[chunk_id*chunk_size:(chunk_id+1)*chunk_size]]

    logger.info(f"Generating conformers and submit entries for {len(all_smiles)} molecules")
    res = dm.parallelized(submit_smiles,
        inputs_list=all_smiles,
        n_jobs=1,
        progress=True,
        client=client, 
        dataset_name="small_molecules", 
        use_replica_exchange=True
    )


if __name__ == "__main__":
    submit_small_molecules()