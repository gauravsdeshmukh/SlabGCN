"""Tools to analyze graphs."""

import numpy as np
import torch

from .constants import REPO_PATH
from .featurizers import (
    list_of_edge_featurizers,
    list_of_node_featurizers,
    ScalarEncoder
)
from .graphs import AtomsGraph
from .utils import featurize_atoms, partition_structure_by_layers


def mean_analysis(atoms, featurizer, layer_cutoffs, analyze_partition=None):
    """
    Calculate mean features for an AtomsGraph.

    Parameters
    ----------
    atoms: ase.Atoms object
        The slab to be analyzed
    featurizer: Featurizer object
        Initialized featurizer with the ScalarEncoder
    layer_cutoffs: list
        Layer cutoffs to partition the structure
    analyze_partition: int
        The number of the partition to be analyzed. If it is None, the mean
        feature is averaged over all partitions

    Returns
    -------
    mean_scalar: float
        Mean feature representing the structure

    """
    # Partition structure
    part_atoms = partition_structure_by_layers(atoms, layer_cutoffs)

    # Mean feature list
    mean_feats = []

    # Go over each partition
    for i, part_idx in enumerate(part_atoms):
        # Featurize graph
        atoms_graph = AtomsGraph(atoms, select_idx=part_idx)
        featurizer.featurize_graph(atoms_graph)

        # Get non-zero feature tensor
        feat_tensor = featurizer.feat_tensor
        non_zero_feats = feat_tensor[feat_tensor != 0.0]

        # Average over feats
        mean_feat = torch.mean(non_zero_feats)

        # Add to list
        mean_feats.append(mean_feat.numpy())

    # Check if particular partition mean is required
    if analyze_partition is not None:
        return mean_feats[analyze_partition]
    else:
        return np.mean(mean_feats)
    
def mean_analysis_scalar(atoms, featurizer_name, layer_cutoffs):
    """
    Calculate mean features for an AtomsGraph.

    Parameters
    ----------
    atoms: ase.Atoms object
        The slab to be analyzed
    featurizer: str
        Name of node or edge featurizer
    layer_cutoffs: list
        Layer cutoffs to partition the structure

    Returns
    -------
    dict
        Dictionary with 

    """
    # Partition structure
    part_atoms = partition_structure_by_layers(atoms, layer_cutoffs)

    # Mean feature list
    mean_feats = []

    # Get featurizer
    for feat in list_of_node_featurizers + list_of_edge_featurizers:
        if feat.name() in featurizer_name:
            featurizer = feat(ScalarEncoder())

    # Go over each partition
    for i, part_idx in enumerate(part_atoms):
        # Featurize graph
        atoms_graph = AtomsGraph(atoms, select_idx=part_idx)
        featurizer.featurize_graph(atoms_graph)

        # Get non-zero feature tensor
        feat_tensor = featurizer.feat_tensor
        non_zero_feats = feat_tensor[feat_tensor != 0.0]

        # Average over feats
        mean_feat = torch.mean(non_zero_feats)

        # Add to list
        mean_feats.append(mean_feat.numpy())

    return {
        0: mean_feats[0],
        1: mean_feats[1],
        2: mean_feats[2]
    }
