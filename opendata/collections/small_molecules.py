import os
import time
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
from opendata.conformers.conformers import generate_conformers_from_smiles
from opendata.collections.specs import get_specs


def enumerate_tautomers(
    mol: dm.Mol,
    n_variants: int = 20,
    max_transforms: int = 1000,
    reassign_stereo: bool = True,
    remove_bond_stereo: bool = True,
    remove_sp3_stereo: bool = True,
    timeout_seconds = None
):
    """Enumerate the possible tautomers of the current molecule.

    Args:
        mol: The molecule whose state we should enumerate.
        n_variants: The maximum amount of molecules that should be returned.
        max_transforms: Set the maximum number of transformations to be applied. This limit is usually
            hit earlier than the n_variants limit and leads to a more linear scaling
            of CPU time with increasing number of tautomeric centers (see Sitzmann et al.).
        reassign_stereo: Whether to reassign stereo centers.
        remove_bond_stereo: Whether stereochemistry information is removed from double bonds involved in tautomerism.
            This means that enols will lose their E/Z stereochemistry after going through tautomer enumeration because
            of the keto-enolic tautomerism.
        remove_sp3_stereo: Whether stereochemistry information is removed from sp3 atoms involved in tautomerism. This
            means that S-aminoacids will lose their stereochemistry after going through tautomer enumeration because
            of the amido-imidol tautomerism.
    """
    enumerator = rdMolStandardize.TautomerEnumerator()

    # Configure the enumeration
    enumerator.SetMaxTautomers(n_variants)
    enumerator.SetMaxTransforms(max_transforms)
    enumerator.SetReassignStereo(reassign_stereo)
    enumerator.SetRemoveBondStereo(remove_bond_stereo)
    enumerator.SetRemoveSp3Stereo(remove_sp3_stereo)

    tautomer_enumerator = enumerator.Enumerate(mol)

    start = time.time()
    duration = 0

    tautomers = []
    while timeout_seconds is None or duration < timeout_seconds:
        try:
            tautomer = next(tautomer_enumerator)
            tautomers.append(tautomer)
        except StopIteration:
            break

        duration = time.time() - start

    return tautomers


def smi_aug(smi):
    mol = dm.to_mol(smi)
    if mol is None:
        res = []
    else:
        
        with dm.without_rdkit_log():
            t0 = time.time()
            nste = n_stereo_centers_unspecified(mol)

            if nste:
                enums = dm.enumerate_stereoisomers(mol, n_variants=0, 
                                                    undefined_only=False, 
                                                    rationalise=True, 
                                                    timeout_seconds=30)
                if len(enums) == 0:
                    candidate_stereo = mol
                else:
                    idx = np.random.choice(len(enums))
                    candidate_stereo = enums.pop(idx)
                t1 = time.time()
                logger.info(f"{mol.GetNumAtoms(), nste, len(enums), t1 - t0}")
            else:
                candidate_stereo = mol
                enums = []
            enums_idx = dm.enumerate_tautomers(candidate_stereo, n_variants=0)
            res = [dm.to_smiles(x) for x in enums+enums_idx]

            
   
    return res  


def smi2entries(smi):
    mol = dm.to_mol(smi)
    inchikey = dm.to_inchikey(mol)
    props = compute_many_descriptors(
            mol, 
            add_properties=False
            )

    omol = generate_conformers_from_smiles(smi)
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


def submit_sm_test():
    client = PortalClient(
        address="https://openfractal-test-pgzbs3yryq-uc.a.run.app",
        username=os.environ["OPENFRACTAL_USERNAME"],
        password=os.environ["OPENFRACTAL_PASSWORD"],
    )
    exit()
    res = load_fragment_collection(include_iso_tauto=False)
    all_smiles = list(res.keys())
    idxs = np.random.choice(len(all_smiles), replace=False, size=500)
    all_smiles = [all_smiles[i] for i in idxs]

    logger.info(f"Augmenting {len(all_smiles)} smiles selection with isomers and tautomers")
    res = dm.parallelized(
        smi_aug,
        inputs_list=all_smiles,
        n_jobs=-1,
        progress=True
    )
    all_smiles = sum(res, [])

    logger.info(f"Generating conformers for {len(all_smiles)} molecules")
    res = dm.parallelized(smi2entries,
        inputs_list=all_smiles,
        n_jobs=1,
        progress=True
    )
    entries = sum(res, [])

    # submit_sp_collection(client=client,
    #                      dataset_name="5k_small_molecules_test",
    #                      entries=entries,
    #                      spec_name="gaz_phase_se",
    #                      tag='sm_se')

    # submit_sp_collection(client=client,
    #                      dataset_name="5k_small_molecules_test_",
    #                      entries=entries,
    #                      spec_name="gaz_phase_dft",
    #                      tag='sm_dft')

if __name__ == "__main__":
    submit_sm_test()