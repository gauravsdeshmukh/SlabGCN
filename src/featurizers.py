"""Node and bond featurizers."""

import abc
import csv
import os

import networkx as nx
import numpy as np
import torch
from mendeleev import element
from torch.nn.functional import one_hot

from .constants import DBAND_FILE_PATH, REPO_PATH, DBAND_DICT, VALENCE_DICT, EA_DICT
from .graphs import AtomsGraph


class OneHotEncoder:
    """Featurize a property using a one-hot encoding scheme."""

    def __init__(self):
        """Blank constructor."""
        pass

    def fit(self, min, max, n_intervals):
        """Fit encoder based on min, max, and number of intervals parameters.

        Parameters
        ----------
        min: int
            Minimum possible value of the property.
        max: int
            Maximum possible value of the property.
        n_intervals: int
            Number of elements in the one-hot encoded array.
        """
        self.min = min
        self.max = max
        self.n_intervals = n_intervals

    def transform(self, property):
        """Transform a given property vector/matrix/tensor.

        Parameters
        ----------
        property: list or np.ndarray or torch.Tensor
            Tensor containing value(s) of the property to be transformed. The
            tensor must have a shape of N where N is the number of atoms.
        """
        # Transform property to tensor
        property = torch.Tensor(property)

        # Scale between 0 and num_intervals
        scaled_prop = ((property - self.min) / (self.max - self.min)) * self.n_intervals
        scaled_prop[scaled_prop >= self.n_intervals] = self.n_intervals - 1
        scaled_prop[scaled_prop < self.min] = 0

        # Apply floor function
        floor_prop = torch.floor(scaled_prop)

        # Create onehot encoding
        onehot_prop = one_hot(floor_prop.to(torch.int64), num_classes=self.n_intervals)
        onehot_prop = onehot_prop.type(torch.float) 

        return onehot_prop


class ScalarEncoder:
    """Featurize a property as a scalar."""

    def __init__(self):
        """Blank constructor."""
        pass

    def fit(self, min, max, n_intervals):
        """Fit encoder based on min, max, and number of intervals parameters.

        Parameters
        ----------
        min: int
            Dummy argument.
        max: int
            Dummy argument.
        n_intervals: int
            Dummy argument.
        """
        self.min = min
        self.max = max
        self.n_intervals = n_intervals

    def transform(self, property):
        """Transform a given property vector/matrix/tensor.

        Parameters
        ----------
        property: list or np.ndarray or torch.Tensor
            Tensor containing value(s) of the property to be transformed. The
            tensor must have a shape of N where N is the number of atoms.
        """
        # Transform property to tensor
        property = torch.Tensor(property)

        return property


class Featurizer(abc.ABC):
    """Meta class for defining featurizers."""

    @abc.abstractmethod
    def __init__(self, encoder):
        """Initialize class variables and fit encoder.

        Parameters
        ----------
        encoder: OneHotEncoder
            Initialized object of class OneHotEncoder.
        """
        pass

    @abc.abstractmethod
    def featurize_graph(self, atoms_graph):
        """Featurize an AtomsGraph.

        This class should create a feature tensor from the given graph. This
        feature tensor should have a shape of (N, M) where N = number of atoms
        and M = n_intervals provided to the encoder. The feature tensor should
        be saved as self._feat_tensor.

        Parameters
        ----------
        graph: AtomsGraph
            A graph of a collection of bulk, surface, or adsorbate atoms.
        """
        pass

    @abc.abstractproperty
    def feat_tensor(self):
        """Return the featurized node tensor.

        Returns
        -------
        feat_tensor: torch.Tensor
            Featurized tensor having shape (N, M) where N = number of atoms and
            M = n_intervals provided to the encoder
        """
        pass

    @abc.abstractstaticmethod
    def name(self):
        """Return the name of the featurizer.

        Returns
        -------
        _name = str
            Name of the featurizer.
        """
        return "abstract_featurizer"


class AtomNumFeaturizer(Featurizer):
    """Featurize nodes based on atomic number."""

    def __init__(self, encoder, min=0, max=80, n_intervals=10):
        """Initialize featurizer with min = 0, max = 80, n_intervals = 10.

        Parameters
        ----------
        encoder: OneHotEncoder
            Initialized object of class OneHotEncoder.
        min: int
            Minimum value of atomic number
        max: int
            Maximum value of atomic number
        n_intervals: int
            Number of intervals
        """
        # Initialize variables
        self.min = min
        self.max = max
        self.n_intervals = n_intervals

        # Fit encoder
        self.encoder = encoder
        self.encoder.fit(self.min, self.max, self.n_intervals)

    def featurize_graph(self, atoms_graph):
        """Featurize an AtomsGraph.

        Parameters
        ----------
        graph: AtomsGraph
            A graph of a collection of bulk, surface, or adsorbate atoms.
        """
        # Get atomic numbers
        atom_num_dict = nx.get_node_attributes(atoms_graph.graph, "atomic_number")
        atom_num_arr = np.array(list(atom_num_dict.values()))
        zero_idx = np.argwhere(atom_num_arr == 0.0)

        # Create node feature matrix
        self._feat_tensor = self.encoder.transform(atom_num_arr)
        if len(self._feat_tensor.shape) > 1:
            self._feat_tensor[zero_idx, :] = 0.0

    @property
    def feat_tensor(self):
        """Return the featurized node tensor.

        Returns
        -------
        feat_tensor: torch.Tensor
            Featurized tensor having shape (N, M) where N = number of atoms and
            M = n_intervals provided to the encoder
        """
        return self._feat_tensor

    @staticmethod
    def name():
        """Return the name of the featurizer."""
        return "atomic_number"


class DBandFeaturizer(Featurizer):
    """Featurize nodes based on close-packed d-band center."""

    def __init__(self, encoder, min=-5, max=3, n_intervals=10):
        """Initialize featurizer with min = -5, max = 3, n_intervals = 10.

        Parameters
        ----------
        encoder: OneHotEncoder
            Initialized object of class OneHotEncoder.
        min: int
            Minimum value of d-band center
        max: int
            Maximum value of d-band center
        n_intervals: int
            Number of intervals
        """
        # Initialize variables
        self.min = min
        self.max = max
        self.n_intervals = n_intervals

        # Fit encoder
        self.encoder = encoder
        self.encoder.fit(self.min, self.max, self.n_intervals)

        # Get dband centers from csv
        self.map_dict = DBAND_DICT

    def featurize_graph(self, atoms_graph):
        """Featurize an AtomsGraph.

        Parameters
        ----------
        graph: AtomsGraph
            A graph of a collection of bulk, surface, or adsorbate atoms.
        """
        # Get atomic numbers
        atom_num_dict = nx.get_node_attributes(atoms_graph.graph, "atomic_number")
        atom_num_arr = np.array(list(atom_num_dict.values()))
        zero_idx = np.argwhere(atom_num_arr == 0.0)

        # Map from atomic number to d-band center
        dband_arr = np.vectorize(self.map_dict.__getitem__)(atom_num_arr)

        # Create node feature matrix
        self._feat_tensor = self.encoder.transform(dband_arr)
        if len(self._feat_tensor.shape) > 1:
            self._feat_tensor[zero_idx, :] = 0.0

    @property
    def feat_tensor(self):
        """Return the featurized node tensor.

        Returns
        -------
        feat_tensor: torch.Tensor
            Featurized tensor having shape (N, M) where N = number of atoms and
            M = n_intervals provided to the encoder
        """
        return self._feat_tensor

    @staticmethod
    def name():
        """Return the name of the featurizer."""
        return "dband_center"


class ValenceFeaturizer(Featurizer):
    """Featurize nodes based on number of valence electrons."""

    def __init__(self, encoder, min=1, max=12, n_intervals=12):
        """Initialize featurizer with min = 1, max = 12, n_intervals = 12.

        Parameters
        ----------
        encoder: OneHotEncoder
            Initialized object of class OneHotEncoder.
        min: int
            Minimum value of valence electrons
        max: int
            Maximum value of valence electrons
        n_intervals: int
            Number of intervals
        """
        # Initialize variables
        self.min = min
        self.max = max
        self.n_intervals = n_intervals

        # Fit encoder
        self.encoder = encoder
        self.encoder.fit(self.min, self.max, self.n_intervals)

        # Create a map between atomic number and number of valence electrons
        self.map_dict = VALENCE_DICT

    def featurize_graph(self, atoms_graph):
        """Featurize an AtomsGraph.

        Parameters
        ----------
        graph: AtomsGraph
            A graph of a collection of bulk, surface, or adsorbate atoms.
        """
        # Get atomic numbers
        atom_num_dict = nx.get_node_attributes(atoms_graph.graph, "atomic_number")
        atom_num_arr = np.array(list(atom_num_dict.values()))
        zero_idx = np.argwhere(atom_num_arr == 0.0)

        # Get valence electrons for each atom
        valence_arr = np.vectorize(self.map_dict.__getitem__)(atom_num_arr)

        # Create node feature matrix
        self._feat_tensor = self.encoder.transform(valence_arr)
        if len(self._feat_tensor.shape) > 1:
            self._feat_tensor[zero_idx, :] = 0.0

    @property
    def feat_tensor(self):
        """Return the featurized node tensor.

        Returns
        -------
        feat_tensor: torch.Tensor
            Featurized tensor having shape (N, M) where N = number of atoms and
            M = n_intervals provided to the encoder
        """
        return self._feat_tensor

    @staticmethod
    def name():
        """Return the name of the featurizer."""
        return "valence"

class EAFeaturizer(Featurizer):
    """Featurize nodes based on electron affinity values in eV."""

    def __init__(self, encoder, min=0, max=3, n_intervals=15):
        """Initialize featurizer with min = 0, max = 3, n_intervals = 15.

        Parameters
        ----------
        encoder: OneHotEncoder
            Initialized object of class OneHotEncoder.
        min: int
            Minimum value of atomic number
        max: int
            Maximum value of atomic number
        n_intervals: int
            Number of intervals
        """
        # Initialize variables
        self.min = min
        self.max = max
        self.n_intervals = n_intervals

        # Fit encoder
        self.encoder = encoder
        self.encoder.fit(self.min, self.max, self.n_intervals)

        # Create a map between atomic number and number of valence electrons
        self.map_dict = EA_DICT

    def featurize_graph(self, atoms_graph):
        """Featurize an AtomsGraph.

        Parameters
        ----------
        graph: AtomsGraph
            A graph of a collection of bulk, surface, or adsorbate atoms.
        """
        # Get atomic numbers
        atom_num_dict = nx.get_node_attributes(atoms_graph.graph, "atomic_number")
        atom_num_arr = np.array(list(atom_num_dict.values()))
        zero_idx = np.argwhere(atom_num_arr == 0.0)

        # Get electron affinity for each atom
        ea_arr = np.vectorize(self.map_dict.__getitem__)(atom_num_arr)

        # Create node feature matrix
        self._feat_tensor = self.encoder.transform(ea_arr)
        if len(self._feat_tensor.shape) > 1:
            self._feat_tensor[zero_idx, :] = 0.0

    @property
    def feat_tensor(self):
        """Return the featurized node tensor.

        Returns
        -------
        feat_tensor: torch.Tensor
            Featurized tensor having shape (N, M) where N = number of atoms and
            M = n_intervals provided to the encoder
        """
        return self._feat_tensor

    @staticmethod
    def name():
        """Return the name of the featurizer."""
        return "electron_affinity"


class ReactivityFeaturizer(Featurizer):
    """Featurize nodes based on close-packed d-band center and/or valence."""

    def __init__(self, encoder, min=-5, max=3, n_intervals=10):
        """Initialize featurizer with min = -5, max = 3, n_intervals = 10.

        Parameters
        ----------
        encoder: OneHotEncoder
            Initialized object of class OneHotEncoder.
        min: int
            Minimum value of d-band center
        max: int
            Maximum value of d-band center
        n_intervals: int
            Number of intervals
        """
        # Initialize variables
        self.min = min
        self.max = max
        self.n_intervals = n_intervals

        # Fit encoder
        self.encoder_dband = encoder
        self.encoder_dband.fit(self.min, self.max, self.n_intervals)

        # Fit valence encoder
        self.encoder_val = encoder
        self.encoder_val.fit(1, 12, self.n_intervals)

        # Get dband centers from csv
        self.map_dict_dband = {}
        with open(DBAND_FILE_PATH, "r") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                self.map_dict_dband[int(row[0])] = float(row[1])

        # Create a map between atomic number and number of valence electrons
        self.map_dict_val = {0: 0, 1: 1, 2: 0}
        for i in range(3, 21, 1):
            self.map_dict_val[i] = np.min([element(i).ec.get_valence().ne(), 12])

    def featurize_graph(self, atoms_graph):
        """Featurize an AtomsGraph.

        Parameters
        ----------
        graph: AtomsGraph
            A graph of a collection of bulk, surface, or adsorbate atoms.
        """
        # Get atomic numbers
        atom_num_dict = nx.get_node_attributes(atoms_graph.graph, "atomic_number")
        atom_num_arr = np.array(list(atom_num_dict.values()))
        zero_idx = np.argwhere(atom_num_arr == 0.0)

        # Map from atomic number to d-band center
        react_arr = np.zeros_like(atom_num_arr)
        for i, n in enumerate(atom_num_arr):
            if n < 21:
                react_arr[i] = self.map_dict_val[n]
            else:
                react_arr[i] = self.map_dict_dband[n]

        # Create node feature matrix
        self._feat_tensor = self.encoder.transform(react_arr)
        if len(self._feat_tensor.shape) > 1:
            self._feat_tensor[zero_idx, :] = 0.0

    @property
    def feat_tensor(self):
        """Return the featurized node tensor.

        Returns
        -------
        feat_tensor: torch.Tensor
            Featurized tensor having shape (N, M) where N = number of atoms and
            M = n_intervals provided to the encoder
        """
        return self._feat_tensor

    @staticmethod
    def name():
        """Return the name of the featurizer."""
        return "reactivity"


class CoordinationFeaturizer(Featurizer):
    """Featurize nodes based on coordination number."""

    def __init__(self, encoder, min=1, max=15, n_intervals=15):
        """Initialize featurizer with min = 1, max = 15, n_intervals = 15.

        Parameters
        ----------
        encoder: OneHotEncoder
            Initialized object of class OneHotEncoder.
        min: int
            Minimum value of valence electrons
        max: int
            Maximum value of valence electrons
        n_intervals: int
            Number of intervals
        """
        # Initialize variables
        self.min = min
        self.max = max
        self.n_intervals = n_intervals

        # Fit encoder
        self.encoder = encoder
        self.encoder.fit(self.min, self.max, self.n_intervals)

    def featurize_graph(self, atoms_graph):
        """Featurize an AtomsGraph.

        Parameters
        ----------
        graph: AtomsGraph
            A graph of a collection of bulk, surface, or adsorbate atoms.
        """
        # Get atomic numbers
        atom_num_dict = nx.get_node_attributes(atoms_graph.graph, "atomic_number")
        atom_num_arr = np.array(list(atom_num_dict.values()))
        zero_idx = np.argwhere(atom_num_arr == 0.0)

        # Get coordination numbers
        cn_dict = nx.get_node_attributes(atoms_graph.graph, "coordination")
        cn_arr = np.array(list(cn_dict.values()))

        # Create node feature matrix
        self._feat_tensor = self.encoder.transform(cn_arr)
        if len(self._feat_tensor.shape) > 1:
            self._feat_tensor[zero_idx, :] = 0.0

    @property
    def feat_tensor(self):
        """Return the featurized node tensor.

        Returns
        -------
        feat_tensor: torch.Tensor
            Featurized tensor having shape (N, M) where N = number of atoms and
            M = n_intervals provided to the encoder
        """
        return self._feat_tensor

    @staticmethod
    def name():
        """Return the name of the featurizer."""
        return "coordination"


class BondDistanceFeaturizer(Featurizer):
    """Featurize edges based on bond distance."""

    def __init__(self, encoder, min, max, n_intervals):
        """Initialize bond distance featurizer.

        Parameters
        ----------
        encoder: OneHotEncoder
            Initialized object of class OneHotEncoder.
        min: int
            Minimum value of atomic number
        max: int
            Maximum value of atomic number
        n_intervals: int
            Number of intervals
        """
        # Initialize variables
        self.min = min
        self.max = max
        self.n_intervals = n_intervals

        # Fit encoder
        self.encoder = encoder
        self.encoder.fit(self.min, self.max, self.n_intervals)

    def featurize_graph(self, atoms_graph):
        """Featurize an AtomsGraph.

        Parameters
        ----------
        graph: AtomsGraph
            A graph of a collection of bulk, surface, or adsorbate atoms.
        """
        # Get atomic numbers
        bond_dist_dict = nx.get_edge_attributes(atoms_graph.graph, "bond_distance")
        bond_dist_arr = np.array(list(bond_dist_dict.values()))

        # Create node feature matrix
        self._feat_tensor = self.encoder.transform(bond_dist_arr)

        # Create list of edge indices
        self._edge_indices = torch.LongTensor(list(atoms_graph.graph.edges())).transpose(0, 1)

    @property
    def feat_tensor(self):
        """Return the featurized node tensor.

        Returns
        -------
        feat_tensor: torch.Tensor
            Featurized tensor having shape (N, M) where N = number of atoms and
            M = n_intervals provided to the encoder
        """
        return self._feat_tensor

    @property
    def edge_indices(self):
        """Return list of edge indices.

        Returns
        -------
        edge_indices: torch.Tensor
            Tensor with edge indices
        """
        return self._edge_indices

    @staticmethod
    def name():
        """Return the name of the featurizer."""
        return "bond_distance"

class InteractionFeaturizer(Featurizer):
    """Featurize edges based on interaction distance."""

    def __init__(self, encoder, min, max, n_intervals):
        """Initialize bond distance featurizer.

        Parameters
        ----------
        encoder: OneHotEncoder
            Initialized object of class OneHotEncoder.
        min: int
            Minimum value of atomic number
        max: int
            Maximum value of atomic number
        n_intervals: int
            Number of intervals
        """
        # Initialize variables
        self.min = min
        self.max = max
        self.n_intervals = n_intervals

        # Fit encoder
        self.encoder = encoder
        self.encoder.fit(self.min, self.max, self.n_intervals)

    def featurize_graph(self, atoms_graph):
        """Featurize an AtomsGraph.

        Parameters
        ----------
        graph: AtomsGraph
            A graph of a collection of bulk, surface, or adsorbate atoms.
        """
        # Get atomic numbers
        inter_dist_dict = nx.get_edge_attributes(
            atoms_graph.graph, "interaction_distance"
        )
        inter_dist_arr = np.array(list(inter_dist_dict.values()))

        # Create node feature matrix
        self._feat_tensor = self.encoder.transform(inter_dist_arr)

        # Create list of edge indices
        self._edge_indices = torch.LongTensor(list(atoms_graph.graph.edges())).transpose(0, 1)

    @property
    def feat_tensor(self):
        """Return the featurized node tensor.

        Returns
        -------
        feat_tensor: torch.Tensor
            Featurized tensor having shape (N, M) where N = number of atoms and
            M = n_intervals provided to the encoder
        """
        return self._feat_tensor

    @property
    def edge_indices(self):
        """Return list of edge indices.

        Returns
        -------
        edge_indices: torch.Tensor
            Tensor with edge indices
        """
        return self._edge_indices

    @staticmethod
    def name():
        """Return the name of the featurizer."""
        return "interaction_distance"
    
class AngleFeaturizer(Featurizer):
    """Featurize edges based on interaction distance."""

    def __init__(self, encoder, min, max, n_intervals, angle_number):
        """Initialize view angle featurizer.

        Parameters
        ----------
        encoder: OneHotEncoder
            Initialized object of class OneHotEncoder.
        min: int
            Minimum value of atomic number
        max: int
            Maximum value of atomic number
        n_intervals: int
            Number of intervals
        angle_number: int
            Angle to be featurized (1 or 2)
        """
        # Initialize variables
        self.min = min
        self.max = max
        self.n_intervals = n_intervals
        self.angle_num = angle_number

        # Fit encoder
        self.encoder = encoder
        self.encoder.fit(self.min, self.max, self.n_intervals)

    def featurize_graph(self, atoms_graph):
        """Featurize an AtomsGraph.

        Parameters
        ----------
        graph: AtomsGraph
            A graph of a collection of bulk, surface, or adsorbate atoms.
        """
        # Get atomic numbers
        inter_dist_dict = nx.get_edge_attributes(
            atoms_graph.graph, f"theta_{self.angle_num}"
        )
        inter_dist_arr = np.array(list(inter_dist_dict.values()))

        # Create node feature matrix
        self._feat_tensor = self.encoder.transform(inter_dist_arr)

        # Create list of edge indices
        self._edge_indices = torch.LongTensor(list(atoms_graph.graph.edges())).transpose(0, 1)

    @property
    def feat_tensor(self):
        """Return the featurized node tensor.

        Returns
        -------
        feat_tensor: torch.Tensor
            Featurized tensor having shape (N, M) where N = number of atoms and
            M = n_intervals provided to the encoder
        """
        return self._feat_tensor

    @property
    def edge_indices(self):
        """Return list of edge indices.

        Returns
        -------
        edge_indices: torch.Tensor
            Tensor with edge indices
        """
        return self._edge_indices

    @staticmethod
    def name():
        """Return the name of the featurizer."""
        return "interaction_distance"


class BulkBondDistanceFeaturizer(BondDistanceFeaturizer):
    """Featurize bulk bond distances.

    Child class of BondDistanceFeaturizer with suitable min, max, and n_interval
    values initialized for bulk atoms. The values are: min = 0, max = 4,
    n_intervals = 4.
    """

    def __init__(self, encoder, min=0, max=4, n_intervals=4):
        super().__init__(encoder, min=min, max=max, n_intervals=n_intervals)

    @staticmethod
    def name():
        """Return the name of the featurizer."""
        return "bulk_bond_distance"


class SurfaceBondDistanceFeaturizer(BondDistanceFeaturizer):
    """Featurize surface bond distances.

    Child class of BondDistanceFeaturizer with suitable min, max, and n_interval
    values initialized for surface atoms. The values are: min = 0, max = 4,
    n_intervals = 4.
    """

    def __init__(self, encoder, min=0, max=4, n_intervals=4):
        super().__init__(encoder, min=min, max=max, n_intervals=n_intervals)

    @staticmethod
    def name():
        """Return the name of the featurizer."""
        return "surface_bond_distance"


class AdsorbateBondDistanceFeaturizer(BondDistanceFeaturizer):
    """Featurize adsorbate bond distances.

    Child class of BondDistanceFeaturizer with suitable min, max, and n_interval
    values initialized for adsorbate atoms. The values are: min = 0, max = 4,
    n_intervals = 8.
    """

    def __init__(self, encoder, min=0, max=4, n_intervals=8):
        super().__init__(encoder, min=min, max=max, n_intervals=n_intervals)

    @staticmethod
    def name():
        """Return the name of the featurizer."""
        return "adsorbate_bond_distance"
    
class AdsorbateInteractionFeaturizer(InteractionFeaturizer):
    """Featurize adsorbate interaction distances.

    Child class of BondDistanceFeaturizer with suitable min, max, and n_interval
    values initialized for adsorbate atoms. The values are: min = 0, max = 4,
    n_intervals = 16.
    """

    def __init__(self, encoder, min=0, max=5, n_intervals=10):
        super().__init__(encoder, min=min, max=max, n_intervals=n_intervals)

    @staticmethod
    def name():
        """Return the name of the featurizer."""
        return "adsorbate_interaction"

class Angle1Featurizer(AngleFeaturizer):
    """Featurize adsorbate view angle 1.

    Child class of AngleFeaturizer with suitable min, max, and n_interval
    values initialized for adsorbate atoms. The values are: min = -180, max = 180,
    n_intervals = 18.
    """

    def __init__(self, encoder, min=-180, max=180, n_intervals=6, angle_number=1):
        super().__init__(
            encoder, min=min, max=max, n_intervals=n_intervals, angle_number=angle_number
        )

    @staticmethod
    def name():
        """Return the name of the featurizer."""
        return "angle_1"
    
class Angle2Featurizer(AngleFeaturizer):
    """Featurize adsorbate view angle 2.

    Child class of AngleFeaturizer with suitable min, max, and n_interval
    values initialized for adsorbate atoms. The values are: min = -180, max = 180,
    n_intervals = 18.
    """

    def __init__(self, encoder, min=-180, max=180, n_intervals=6, angle_number=2):
        super().__init__(
            encoder, min=min, max=max, n_intervals=n_intervals, angle_number=angle_number
        )

    @staticmethod
    def name():
        """Return the name of the featurizer."""
        return "angle_2"

list_of_node_featurizers = [
    AtomNumFeaturizer,
    DBandFeaturizer,
    ValenceFeaturizer,
    CoordinationFeaturizer,
    ReactivityFeaturizer,
    EAFeaturizer
]

list_of_edge_featurizers = [
    BulkBondDistanceFeaturizer,
    SurfaceBondDistanceFeaturizer,
    AdsorbateBondDistanceFeaturizer,
    AdsorbateInteractionFeaturizer,
    Angle1Featurizer,
    Angle2Featurizer
]

if __name__ == "__main__":
    from ase.io import read

    file_path = REPO_PATH / "data" / "S_calcs" / "Au_3_Rh_9_-0-0-S.cif"
    atoms = read(file_path)
    g = AtomsGraph(atoms, select_idx=[24])

    # anf = AtomNumFeaturizer(OneHotEncoder())
    anf = DBandFeaturizer(ScalarEncoder())
    anf.featurize_graph(g)
    print(anf.feat_tensor)

    bdf = BulkBondDistanceFeaturizer(ScalarEncoder())
    bdf.featurize_graph(g)
    print(bdf.feat_tensor)
    print(bdf.edge_indices)
