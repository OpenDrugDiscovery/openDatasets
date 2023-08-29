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

def _create_entry(id, mol_qcschema, props):
    kwargs = dict(
        name=id,
        molecule=mol_qcschema,
        additional_keywords={},
        attributes=props,
        comment=None
    )
    print(dir(mol_qcschema))
    return SinglepointDatasetNewEntry(**kwargs)


def submit_smiles(smi, client, dataset_name, 
                  use_replica_exchange=True, n_conformers=None):
    mol = dm.to_mol(smi)
    inchikey = dm.to_inchikey(mol)
    props = compute_many_descriptors(mol, add_properties=False)

    omol = ConformerGenerator(use_replica_exchange=use_replica_exchange,
                              n_conformers=n_conformers).from_smiles(smi)
              
    # Build the entries
    if omol is not None:
        entries = [_create_entry(f"{inchikey}_{c}", omol.to_qcschema(conformer=c), props) 
                   for c in np.arange(omol.n_conformers)]
    else:
        entries = []
    print(smi, len(entries))
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
    
    submit_sp_collection(client=client,
                         dataset_name=dataset_name,
                         entries=entries,
                         spec_name="implicit_water_se",
                         tag='sm_se')

    submit_sp_collection(client=client,
                         dataset_name=dataset_name,
                         entries=entries,
                         spec_name="implicit_water_dft",
                         tag='sm_dft')
    return entries


def submit_sp_collection(client, dataset_name, entries, spec_name, tag, **dataset_kwargs):
    dataset_type = "singlepoint"
    ds_list = client.list_datasets()
    dataset_exits = any([(x["dataset_name"] == dataset_name) for x in ds_list])
    if not dataset_exits:
        client.add_dataset(dataset_type=dataset_type, name=dataset_name, **dataset_kwargs)
    ds = client.get_dataset(dataset_type, dataset_name)

    if spec_name not in ds.specification_names:
        spec = get_specs(spec_name)
        ds.add_specification(name=spec_name, specification=spec)
    ds.add_entries(entries)
    res = ds.submit(tag=tag)
    print(res)
    return res


@click.command()
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
    f = lambda x: submit_smiles(x, use_replica_exchange=False, 
                                client=client, dataset_name="small_molecules")
    res = dm.parallelized(f,
        inputs_list=all_smiles,
        n_jobs=1,
        progress=True,
    )


if __name__ == "__main__":
    submit_small_molecules()