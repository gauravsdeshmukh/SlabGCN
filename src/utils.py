"""Utility functions."""

from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader
from ase.io.trajectory import Trajectory
from pathlib import Path

from .featurizers import (
    OneHotEncoder,
    list_of_edge_featurizers,
    list_of_node_featurizers,
)
from .graphs import AtomsGraph


def partition_structure(atoms, z_cutoffs=[], discard_parts=[]):
    """Partition atomic structue into bulk, surface, and/or adsorbates.

    If layer_cutoffs is empty, the slab will not be partitioned. The whole
    structure will be considered as a single partition.

    Parameters
    ----------
    atoms: ase.Atoms object
        The structure to be partitioned
    z_cutoffs: list or np.ndarray (default = [])
        List of z-coordinate cutoffs. xy planes are placed at the specified
        cutoffs to partition atoms above and below them. The length of z-cutoffs
        should be equal to one less than the number of partitions.
    discard_parts: list or np.ndarray (default = [])
        Indices of partititons to discard. For example, if discard parts contains
        0, then the first partition is discarded and not returned.
    """
    # Set number of partitions equal to 1 more than the length of z_cutoffs
    n_partitions = int(len(z_cutoffs) + 1)

    # Add 0 and infinity to cutoffs
    if n_partitions > 1:
        z_cutoffs = np.insert(z_cutoffs, 0, 0)
        z_cutoffs = np.insert(z_cutoffs, len(z_cutoffs), np.inf)
    else:
        z_cutoffs = np.array([0.0, np.inf])

    # Get positions
    pos = atoms.get_positions()

    # Iterate over number of partitions
    part_atoms = []
    for i in range(n_partitions):
        if i not in discard_parts:
            part_idx = (
                np.argwhere((pos[:, -1] >= z_cutoffs[i]) & (pos[:, -1] < z_cutoffs[i + 1]))
                .flatten()
                .tolist()
            )
            part_atoms.append(part_idx)

    return part_atoms


def partition_structure_by_layers(atoms, layer_cutoffs=[], discard_parts=[]):
    """Partition atomic structue into bulk, surface, and/or adsorbates by layers.

    If layer_cutoffs is empty, the slab will not be partitioned. The whole
    structure will be considered as a single partition.

    Parameters
    ----------
    atoms: ase.Atoms object
        The structure to be partitioned
    layer_cutoffs: list or np.ndarray (default = [])
        List of layer cutoffs. xy planes are placed above the specified layer
        cutoffs to partition atoms above and below them. The length of layer
        should be equal to one less than the number of partitions.
    discard_parts: list or np.ndarray (default = [])
        Indices of partititons to discard. For example, if discard parts contains
        0, then the first partition is discarded and not returned.

    Returns
    -------
    part_atoms: list of np.ndarray
        Array of indices of the partitioned atoms
    """
    # Set number of partitions equal to 1 more than the length of z_cutoffs
    n_partitions = int(len(layer_cutoffs) + 1)

    if n_partitions > 1:
        # Calculate interlayer distance
        z_array = np.unique(np.sort(atoms.get_positions()[:, -1]))
        z_min = z_array.min()
        d_interlayer = abs(z_array[1] - z_array[0])
        z_cutoffs = z_min + (np.array(layer_cutoffs) - 1) * d_interlayer + 0.1

        # Add 0 and infinity to cutoffs
        z_cutoffs = np.insert(z_cutoffs, 0, 0)
        z_cutoffs = np.insert(z_cutoffs, len(z_cutoffs), np.inf)
    else:
        z_cutoffs = np.array([0.0, np.inf])

    # Get positions
    pos = atoms.get_positions()

    # Iterate over number of partitions
    part_atoms = []
    for i in range(n_partitions):
        if i not in discard_parts:
            part_idx = (
                np.argwhere((pos[:, -1] >= z_cutoffs[i]) & (pos[:, -1] < z_cutoffs[i + 1]))
                .flatten()
                .tolist()
            )
            part_atoms.append(part_idx)

    return part_atoms


def featurize_atoms(
    atoms,
    select_idx,
    node_features,
    edge_features,
    max_atoms=None,
    encoder=OneHotEncoder(),
    include_neighbors=True,
):
    """Featurize atoms and bonds with the chosen featurizers.

    Parameters
    ----------
    atoms: ase.Atoms objet
        Atoms object containing the whole structure
    select_idx: list
        List of indices for atoms to featurize. Typically, this will be provided
        by the partition_structure function
    node_features: list or np.ndarray
        Names of node featurizers to use (current options: atomic_number, dband
        center, valence electrons, coordination, reactivity). The "reactivity"
        featurizer uses valence electrons for adsorbates (atomic number < 21) and
        d-band center for larger transition metal atoms (atomic number >= 21).
    edge_features: list or np.ndarray
        Names of edge featurizers to use (current options: bulk_bond_distance,
        surface_bond_distance, adsorbate_bond_distance). All of these encode
        bond distance using a one-hot encoder, but the bounds for each vary.
    max_atoms: int (default = None)
        Maximum number of allowed atoms. If it is not None, graphs
        that have fewer nodes than max_atoms are padded with 0s to ensure
        that the total number of nodes is equal to max_atoms.
    encoder: encoder object from featurizers.py
        Currently only the OneHotEncoder is supported

    Returns
    -------
    dict
        Dictionary with keys "node_tensor", "edge_tensor", and "edge_indices" and
        corresponding tensors as values.
    """
    # Create graph
    atoms_graph = AtomsGraph(
        atoms=atoms, select_idx=select_idx, max_atoms=max_atoms, include_neighbors=include_neighbors
    )

    # Collect node featurizers
    node_feats = []
    node_intervals = []
    for node_featurizer in list_of_node_featurizers:
        if node_featurizer.name() in node_features:
            nf = node_featurizer(deepcopy(encoder))
            node_intervals.append(nf.n_intervals)
            node_feats.append(nf)

    # Collect edge featurizers
    edge_feats = []
    edge_intervals = []
    for edge_featurizer in list_of_edge_featurizers:
        if edge_featurizer.name() in edge_features:
            ef = edge_featurizer(deepcopy(encoder))
            edge_intervals.append(ef.n_intervals)
            edge_feats.append(ef)

    # Store node matrices from each feaurizer
    node_mats = []
    for nf in node_feats:
        nf.featurize_graph(atoms_graph)
        node_mats.append(nf.feat_tensor)
    # Stack node mats to create node tensor
    node_tensor = torch.hstack(node_mats)

    # Store edge matrices from each featurizer
    edge_mats = []
    for ef in edge_feats:
        ef.featurize_graph(atoms_graph)
        edge_mats.append(ef.feat_tensor)
    # Stack edge mats to create edge tensor
    edge_tensor = torch.hstack(edge_mats)

    # Store edge indices
    edge_indices = ef.edge_indices

    return {
        "node_tensor": node_tensor,
        "edge_tensor": edge_tensor,
        "edge_indices": edge_indices,
        "map_idx_node": atoms_graph.map_idx_node,
    }


def create_dataloaders(proc_data, sample_idx, batch_size, num_proc=0):
    """Create training, validation, and/or test dataloaders.

    Parameters
    ----------
    proc_data: AtomsDataset or AtomsDatapoints
        Processed dataset object
    sampler: dict
        A dictionary with "train", "val", and "test" indices returned by a Sampler
        object.
    batch_size: int
        Batch size
    num_proc: int (default = 0)
        Number of cores to be used for parallelization. Defaults to serial.

    Returns
    -------
    dataloader_dict: dict
        Dictionary of "train", "val", and "test" dataloaders
    """
    # Create dataloader dict
    dataloader_dict = {"train": [], "val": [], "test": []}

    for key in dataloader_dict.keys():
        if sample_idx[key].shape[0] > 0.0:
            sampler = SubsetRandomSampler(sample_idx[key])
            dataloader_dict[key] = DataLoader(
                dataset=proc_data,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_proc,
            )

    return dataloader_dict


def get_composition_string(atoms):
    """Get a composition string.

    The string has atom symbols and number of atoms for each element separated by
    underscores.

    """
    symbols = atoms.get_chemical_symbols()
    uniq_symbols, counts = np.unique(symbols, return_counts=True)
    comp_string = ""
    for sym, count in zip(uniq_symbols, counts):
        comp_string += f"{sym}_{count}_"

    return comp_string

def process_opt_results(opt_directory, iteration, adsorbate_symbol="S"):
    """Processs results of an optimization run.

    This function creates a dictionary with the number of sulfur atoms as the key
    and the respective Atoms objects as well as outputs as keys. The structure of
    the dictionary is shown below:

    num_sulfur_atoms--> atoms: [atoms_1, atoms_2, ...]
                        outputs --> 0 : [output_0_1, output_0_2, ...]
                                    1 : [output_1_1, output_1_2, ...]
    
    Parameters
    ----------
    opt_directory: str or Path
        Path to directory containing results of an optimization run
    iteration: int
        Iteration number
    adsorbate_symbol: str (default = "S")
        Adsorbate symbol
    
    Returns
    -------
    opt_dict: dict
        Optimization results dictionary

    """
    # Create path object
    opt_superpath = Path(opt_directory)
    opt_path = opt_superpath / str(iteration)

    # Open trajectory file
    traj_path = opt_path / "structures" / "structures.traj"
    traj = Trajectory(traj_path, mode="r")

    # Open outputs
    outputs_path = opt_path / "outputs" / "outputs.csv"
    df = pd.read_csv(outputs_path, header=None)
    n_outputs = len(df.columns)

    # Create dictionary
    opt_dict = {}
    for i in range(len(traj)):
        # Get number of adsorbates
        n_ads = traj[i].get_chemical_symbols().count(adsorbate_symbol)
        if n_ads not in opt_dict.keys():
            opt_dict[n_ads] = {"atoms": [], "outputs": {i: [] for i in range(n_outputs)}}
        else:
            opt_dict[n_ads]["atoms"].append(deepcopy(traj[i]))
            for j in range(n_outputs):
                opt_dict[n_ads]["outputs"][j].append(df.loc[i, j])

    return opt_dict


if __name__ == "__main__":
    from ase.io import read

    atoms = read("CONTCAR")

    part_atoms = partition_structure(atoms, z_cutoffs=[15, 23.5])
    print(part_atoms)

    feat_dict = featurize_atoms(
        atoms,
        part_atoms[0],
        ["atomic_number", "dband_center"],
        ["bulk_bond_distance"],
        max_atoms=34,
    )
    print(feat_dict)
