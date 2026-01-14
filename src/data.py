"""Store graph data using PyTorch Geometric abstractions."""

import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
from ase import Atoms
from ase.io import read
from torch_geometric.data import Data, Dataset, InMemoryDataset

from .constants import REPO_PATH
from .featurizers import OneHotEncoder
from .utils import featurize_atoms, partition_structure_by_layers, partition_structure


class AtomsDataset(Dataset):
    """Class to hold a dataset containing graphs of atomic_structures."""

    def __init__(self, root, prop_csv):
        """Initialize an AtomsDataset.

        Atomic structures stored as .cif files in the root directory are loaded.

        Paramters
        ---------
        root: str
            Path to the directory in which atomic structures are stored
        prop_csv: str
            Path to the file mapping atomic structure filename and property.
            This filename will typically have two columns, the first with the
            names of the cif files and the second with the
            corresponding target property values.
        """
        super().__init__(root)
        self.root_path = Path(self.root)

        # Create processed path if it doesn't exist
        self.processed_path = Path(self.processed_dir)
        self.processed_path.mkdir(exist_ok=True)

        # Read csv
        self.prop_csv = prop_csv
        self.names = []
        self.props = []
        with open(self.prop_csv, "r") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                self.names.append(str(row[0]))
                self.props.append([float(row[i]) for i in range(1, len(row))])

        # Create name to property map
        self.map_name_prop = {name: prop for name, prop in zip(self.names, self.props)}

        # Create map list
        self.map_list = []

        # Load index.csv if processed
        self.index_path = self.processed_path / "index.csv"
        if self.processed_status():
            self.df_name_idx = pd.read_csv(self.index_path)

    def process_data(
        self,
        layer_cutoffs,
        node_features,
        edge_features,
        max_atoms=None,
        encoder=OneHotEncoder(),
        discard_parts=[],
        include_neighbors=True,
        silence=False,
    ):
        """Process raw data in the root directory into PyTorch Data and save.

        Each atomic structure in the root directory is partitioned based on the
        given z_cutoffs and each partition is featurized according to the given
        node_features and edge_features. The featurized graphs are converted
        into Data objects and stored in the "processed" directory under root.

        Parameters
        ----------
        layer_cutoffs: list or np.ndarray
            List of layer cutoffs based on which atomic structures are
            partitioned. The number of partitions is equal to one more than the
            length of z_cutoffs.
        node_features: list[list]
            List of lists of node featurization methods to be used for each
            partition. For e.g., specify [["atomic_number", "dband_center"],
            ["atomic_number", "reactivity"], ["atomic_number", "reactivity"]] for
            a typical bulk + surface + adsorbate partition.
        edge_features: list[list]
            List of lists of edge featurization methods to be used for each
            partition. For e.g., specify [["bulk_bond_distance"],
            ["surface_bond_distance"], ["adsorbate_bond_distance"]] for
            a typical bulk + surface + adsorbate partition.
        max_atoms: int (default = None)
            Maximum number of nodes in graph. If a value is provided, graphs are
            padded to make sure the total number of nodes matches max_atoms.
        encoder: OneHotEncoder object
            Encoder to convert properties to vectors
        discard_parts: list or np.ndarray (defaut = [])
            Indices of partitions to be discarded. By default, no partitions are
            discarded.
        include_neighbors: bool or list or np.ndarray (default = False)
            Whether to include neighbors of the atoms in the graph (that are not
            already in the graph)
        silence: bool
            Whether to print output while processing data.
        """
        # Create empty dataframe to store index and name correspondence
        self.df_name_idx = pd.DataFrame(
            {"index": [0] * len(self.names), "name": [""] * len(self.names)}
        )

        # Create empty dictionary to store atoms
        self.atoms = {}

        # Store layer cutoffs
        self.layer_cutoffs = layer_cutoffs

        # Handle include_neighbors
        if isinstance(include_neighbors, bool):
            include_neighbors = [include_neighbors] * (len(layer_cutoffs) + 1)

        # Choose partition function
        if isinstance(self.layer_cutoffs[0], int):
            part_func = partition_structure_by_layers
        else:
            part_func = partition_structure

        # Iterate over files and process them
        for i, name in tqdm.tqdm(
            enumerate(self.names), desc="Processing data", total=len(self.names),
            disable=silence
        ):
            # Map index to name
            self.df_name_idx.loc[i, "index"] = i
            self.df_name_idx.loc[i, "name"] = name

            # Set file path
            file_path = self.root_path / name

            # Read structure
            atoms = read(str(file_path))
            self.atoms[i] = atoms

            # Partition structure
            part_atoms = part_func(
                atoms, layer_cutoffs, discard_parts
            )

            # Featurize partitions
            data_objects = []
            map_objects = []
            for j, part_idx in enumerate(part_atoms):
                feat_dict = featurize_atoms(
                    atoms,
                    part_idx,
                    node_features=node_features[j],
                    edge_features=edge_features[j],
                    max_atoms=max_atoms,
                    encoder=encoder,
                    include_neighbors=include_neighbors[j],
                )

                # Save index-node mapping
                map_objects.append(feat_dict["map_idx_node"])

                # Convert to Data object
                data_obj = Data(
                    x=feat_dict["node_tensor"].to(torch.float32),
                    edge_index=feat_dict["edge_indices"],
                    edge_attr=feat_dict["edge_tensor"],
                    y=torch.Tensor([*self.map_name_prop[name]]),
                )
                data_objects.append(data_obj)

            # Save data objects
            torch.save(data_objects, self.processed_path / f"data_{i}.pt")

            # Save mapping
            self.map_list.append(map_objects)

        # Save name-index dataframe
        self.df_name_idx.to_csv(self.index_path, index=None)


    def len(self):
        """Return size of the dataset."""
        return len(self.names)

    def get(self, i):
        """Fetch the processed graph(s) at the i-th index."""
        data_objects = torch.load(self.processed_path / f"data_{i}.pt")
        return data_objects
    
    def get_atoms(self, i):
        """Fetch the atoms object at the i-th index."""
        return self.atoms[i]
    
    def get_map(self, i):
        """Fetch the index-node mapping at the i-th index."""
        map_objects = self.map_list[i]
        return map_objects

    def processed_status(self):
        """Check if the dataset is processed."""
        if Path(self.index_path).exists():
            return True
        else:
            return False


class AtomsDatapoints(InMemoryDataset):
    """Class to hold atomic structures as a datapoints.

    This main difference between this class and AtomsDataset is that this is
    initialized with a list of atoms objects (as opposed to a directory with
    files containing atomic structures) and can be initialized without targets.
    This is useful to make predictions on atomic structures for which true target
    values are not known, i.e., previously unseen structures, or to create
    combined datasets on-the-fly.
    """

    def __init__(self, atoms, targets=None):
        """Initialize an AtomsDatapoint.

        Atomic structures provided in the list are initialized.

        Paramters
        ---------
        atoms: ase.Atoms object or a list of ase.Atoms objects
            Structures for which predictions are to be made.
        targets: list or np.ndarray (default = None)
            Corresponding target values for the given structures. If targets is
            None, this dataset cannot be used for training, it can only be used
            for prediction.
        """
        super().__init__()
        # If single object, convert to list
        if isinstance(atoms, Atoms):
            atoms = [atoms]

        # Save object
        self.atoms = atoms
        self.targets = targets
        self.data_list = []
        self.map_list = []

    def process_data(
        self,
        layer_cutoffs,
        node_features,
        edge_features,
        max_atoms=None,
        encoder=OneHotEncoder(),
        discard_parts=[],
        include_neighbors=True,
        silence=False,
    ):
        """Process list of Atoms objects into PyTorch Data and save.

        Each atomic structure in the root directory is partitioned based on the
        given z_cutoffs and each partition is featurized according to the given
        node_features and edge_features. The featurized graphs are converted
        into Data objects and stored in the "processed" directory under root.

        Parameters
        ----------
        layer_cutoffs: list or np.ndarray
            List of layer cutoffs based on which atomic structures are
            partitioned. The number of partitions is equal to one more than the
            length of z_cutoffs.
        node_features: list[list]
            List of lists of node featurization methods to be used for each
            partition. For e.g., specify [["atomic_number", "dband_center"],
            ["atomic_number", "reactivity"], ["atomic_number", "reactivity"]] for
            a typical bulk + surface + adsorbate partition.
        edge_features: list[list]
            List of lists of edge featurization methods to be used for each
            partition. For e.g., specify [["bulk_bond_distance"],
            ["surface_bond_distance"], ["adsorbate_bond_distance"]] for
            a typical bulk + surface + adsorbate partition.
        max_atoms: int (default is None)
            Maximum number of nodes in graph. If a value is provided, graphs are
            padded to make sure the total number of nodes matches max_atoms.
        encoder: OneHotEncoder object
            Encoder to convert properties to vectors
        discard_parts: list or np.ndarray (default = [])
            Indices of partitions to be discarded. By default, no partitions are
            discarded.
        include_neighbors: bool or list or np.ndarray (default = False)
            Whether to include neighbors of the atoms in the graph (that are not
            already in the graph)
        silence: bool
            Whether to print output while processing data.
        """
        # Store layer cutoffs
        self.layer_cutoffs = layer_cutoffs

        # Handle include_neighbors
        if isinstance(include_neighbors, bool):
            include_neighbors = [include_neighbors] * (len(layer_cutoffs) + 1)

        # Choose partition function
        if isinstance(self.layer_cutoffs[0], int):
            part_func = partition_structure_by_layers
        else:
            part_func = partition_structure

        # Iterate over files and process them
        for i, atoms_obj in tqdm.tqdm(
            enumerate(self.atoms), desc="Processing data", total=len(self.atoms),
            disable=silence
        ):
            # Partition structure
            part_atoms = part_func(
                atoms_obj, layer_cutoffs, discard_parts
            )

            # Featurize partitions
            data_objects = []
            map_objects = []
            for j, part_idx in enumerate(part_atoms):
                feat_dict = featurize_atoms(
                    atoms_obj,
                    part_idx,
                    node_features=node_features[j],
                    edge_features=edge_features[j],
                    max_atoms=max_atoms,
                    encoder=encoder,
                    include_neighbors=include_neighbors[j]
                )

                # Save index-node mapping
                map_objects.append(feat_dict["map_idx_node"])

                # Convert to Data object
                if self.targets is None:
                    data_obj = Data(
                        x=feat_dict["node_tensor"].to(torch.float32),
                        edge_index=feat_dict["edge_indices"],
                        edge_attr=feat_dict["edge_tensor"],
                    )
                else:
                    data_obj = Data(
                        x=feat_dict["node_tensor"].to(torch.float32),
                        edge_index=feat_dict["edge_indices"],
                        edge_attr=feat_dict["edge_tensor"],
                        y=torch.Tensor([*self.targets[i]]),
                    )
                data_objects.append(data_obj)

            # Save maps
            self.map_list.append(map_objects)

            # Save data objects
            self.data_list.append(data_objects)

    def len(self):
        """Return size of the dataset."""
        return len(self.atoms)

    def get(self, i):
        """Fetch the processed graph(s) at the i-th index."""
        data_objects = self.data_list[i]
        return data_objects

    def get_atoms(self, i):
        """Fetch the atoms object at the i-th index."""
        return self.atoms[i]
    
    def get_map(self, i):
        """Fetch the index-node mapping at the i-th index."""
        map_objects = self.map_list[i]
        return map_objects

    def append(self, atomsdp):
        """Append one or more AtomsDatapoints.

        The object is modified in-place.

        Parameters
        ----------
        atomsdp: AtomsDatapoints or list of AtomsDatapoints
            List of AtomsDatapoints objects to be appended
        """
        # Check type of argument
        if isinstance(atomsdp, AtomsDatapoints):
            atomsdp = [atomsdp]

        # Append objects
        for dp in atomsdp:
            self.atoms.extend(dp.atoms)
            self.targets.extend(dp.targets)
            self.data_list.extend(dp.data_list)


def load_dataset(root, prop_csv, process_dict=None, load_in_memory=False):
    """Load an AtomsDataset or AtomsDatapoints at the path given by root.

    If process_dict is provided, the process_data method of AtomsDataset is called
    to convert the atomic structures to graphs based on the given parameters in
    process_dict. This should be used when the dataset is created for the first
    time. If load_in_memory is True, the data is loaded as an AtomsDatapoints
    object.

    Parameters
    ----------
    root: str
        Path to the dataset
    prop_csv: str
        Path to the file mapping atomic structure filename and property.
        This filename will typically have two columns, the first with the
        names of the cif files and the second with the
        corresponding target property values.
    process_dict: dict (default = None)
        If this is provided, atomic structures at root will be processed into
        graphs and stored under a "processed" subdirectory. Only use this when
        creating a new dataset. This should contain the following keys: z_cutoffs,
        node_features, edge_features, max_atoms (optional), encoder (optional).
        Refer to the documentation of process_atoms for more information regarding
        these parameters.
    load_in_memory: bool (default = False)
        If this is True, the dataset is loaded as an AtomsDatapoints object.

    Returns
    -------
    dataset: AtomsDataset or AtomsDatapoints
        Initialized AtomsDataset or AtomsDatapoints object
    """
    if not load_in_memory:
        dataset = AtomsDataset(root, prop_csv)
        if process_dict is not None:
            dataset.process_data(**process_dict)
    else:
        # Get names and targets
        names = []
        targets = []
        with open(prop_csv, "r") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                names.append(str(row[0]))
                targets.append([float(row[i]) for i in range(1, len(row))])

        # Get atoms
        atoms_list = []
        for name in names:
            file_path = os.path.join(root, name)
            atoms = read(file_path)
            atoms_list.append(atoms)

        # Load datapoints
        dataset = load_datapoints(
            atoms=atoms_list, process_dict=process_dict, targets=targets
        )

    return dataset


def load_datapoints(atoms, process_dict, targets=None):
    """Load AtomsDatapoints for the provided ase.Atoms or list of ase.Atoms.

    If process_dict is provided, the process_data method of AtomsDatapoints is called
    to convert the atomic structures to graphs based on the given parameters in
    process_dict. This should be used when the dataset is created for the first
    time.

    Parameters
    ----------
    atoms: ase.Atoms object or a list of ase.Atoms objects
        Structures for which predictions are to be made.
    process_dict: dict
        Parameters to process the provided Atoms objects into graphs.
        This should contain the following keys: z_cutoffs, node_features,
        edge_features, max_atoms (optional), encoder (optional). Refer to the
        documentation of process_atoms for more information regarding these
        parameters.
    targets: list (default = None)
        Corresponding targets for structures (optional)

    Returns
    -------
    datapoints: AtomsDatapoints
        Initialized AtomsDatapoints object
    """
    datapoints = AtomsDatapoints(atoms, targets)
    if process_dict is not None:
        datapoints.process_data(**process_dict)

    return datapoints


def combine_datasets(
    root_paths=None,
    prop_csv_paths=None,
    atoms_list=None,
    targets_list=None,
    path_mix_fractions=None,
    atoms_mix_fraction=None,
    seed=0,
):
    """
    Combine multiple datasets into a single AtomsDatapoints object.

    The datasets to be combined can be precursors to AtomsDataset objects, i.e.,
    in directories (specified by root_paths and prop_csv_paths), or in the form
    of a list of atoms objects and a list of targets (specified by atoms_list and
    targets_list), or a mix of both. Ensure that at least one of (root_paths,
    prop_csv_paths) or (atoms_list, targets_list) is specified.

    Parameters
    ----------
    root_paths: str or list of str (default = None)
        Path(s) to directories with structures that are to be combined in a single
        dataset.
    prop_csv_paths: str or list of str (default = None)
        Path(s) to csvs with the names and properties of structures.
    atoms_list: list of Atoms objects (default = None)
        List of atoms objects to be combined into a single dataset.
    targets_list: list (default = None)
        List of targets of the corresponding atoms objects.
    path_mix_fractions: list (default = None)
        List of fractions specifying the fraction of total structures to be
        randomly sampled from each root_path.
    atoms_mix_fraction: float
        Fraction of total structures in atoms_list to be randomly sampled.
    seed: int
        Seed for random sampling from each dataset (only used if any one of the
        fractions is lower than 1).

    Returns
    -------
    comb_dataset: AtomsDatapoints object
        The combined dataset as an AtomsDatapoints object with targets.
    """
    # Check for errors
    if root_paths is None and atoms_list is None:
        raise ValueError("Both root_paths and atoms_list cannot be None.")
    if root_paths is not None:
        if prop_csv_paths is None:
            raise ValueError("Specify prop_csv_paths.")
        if path_mix_fractions is None:
            raise ValueError("Specify path_mix_fractions.")
    if atoms_list is not None:
        if atoms_mix_fraction is None:
            raise ValueError("Specify atoms_mix_fraction.")
        if targets_list is None:
            raise ValueError("Specify targets_list.")

    # Create master lists
    comb_atoms_objects = []
    comb_targets = []

    # Create randomizer
    rng = np.random.default_rng(seed=seed)

    # Combine datasets
    if root_paths is not None:
        for i, rp in enumerate(root_paths):
            # Get all cif files
            struct_files = [f for f in os.listdir(rp) if f.endswith(".cif")]

            # Get prop_csv
            df = pd.read_csv(prop_csv_paths[i], header=None)

            # Get number of files and sample accordingly
            len_struct_files = len(struct_files)
            n_samples = int(
                min(
                    np.round(path_mix_fractions[i] * len_struct_files), len_struct_files
                )
            )
            sampled_struct_files = rng.choice(
                struct_files, size=n_samples, replace=False
            )
            for f in sampled_struct_files:
                sampled_struct = read(os.path.join(rp, f))
                comb_atoms_objects.append(sampled_struct)
                sampled_target = df.loc[df[0] == f, 1].values[0]
                comb_targets.append(sampled_target)

    # Combine with atoms objects if present
    if atoms_list is not None:
        # Get length of atoms list
        len_atoms_list = len(atoms_list)

        # Sample accordingly
        n_atoms_samples = int(
            min(np.round(path_mix_fractions[i] * len_struct_files), len_struct_files)
        )
        sampled_atoms_idx = rng.choice(
            len_atoms_list, size=n_atoms_samples, replace=False
        )

        # Get atoms and targets
        for i in range(len_atoms_list):
            if i in sampled_atoms_idx:
                comb_atoms_objects.append(atoms_list[i])
                comb_targets.append(targets_list[i])

    # Create AtomsDatapoints object
    comb_dataset = AtomsDatapoints(atoms=comb_atoms_objects, targets=comb_targets)

    return comb_dataset


if __name__ == "__main__":
    # Get path to root directory
    data_root_path = Path(REPO_PATH) / "data" / "S_calcs"
    prop_csv_path = data_root_path / "name_prop.csv"

    # Create datapoint
    atoms = read(data_root_path / "Pt_3_Rh_9_-7-7-S.cif")
    datapoint = AtomsDatapoints(atoms)
    datapoint.process_data(
        layer_cutoffs=[3, 6],
        node_features=[
            ["atomic_number", "dband_center"],
            ["atomic_number", "reactivity"],
            ["atomic_number", "reactivity"],
        ],
        edge_features=[
            ["bulk_bond_distance"],
            ["surface_bond_distance"],
            ["adsorbate_bond_distance"],
        ],
    )
    print(datapoint.get(0))
