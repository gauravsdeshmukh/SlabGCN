"""Methods to optimize compositions and make phase diagrams."""

import itertools
import json
import os
from copy import deepcopy
from pathlib import Path, PurePath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.build import molecule, add_adsorbate
from ase.io import read, write
from ase.io.trajectory import TrajectoryWriter, Trajectory
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from ase.data import atomic_numbers, covalent_radii
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor as pym_ase
from sympy.utilities.iterables import multiset_permutations

from .constants import REPO_PATH, VegardsLaw, kB, e
from .data import load_datapoints
from .train import Model
from .utils import partition_structure_by_layers
from .thermodynamics import mu0_H, mu0_H2S, mu_ads, mu_gas
from .plot_utils import make_phase_diagram


# Class to set up optimizer
class Optimizer:
    """Optimizer for a given property predicted using SlabGCN."""

    def __init__(
        self,
        element_list,
        prob_list,
        max_elements,
        seed,
        node_features,
        edge_features,
        layer_cutoffs,
        adsorbate_symbol,
        upper_bounds=None,
        lower_bounds=None,
    ):
        """Initialize optimizer.

        Parameters
        ----------
        element_list: list
            List of elements from which to sample from.
        prob_list: list
            List of probabilites of choosing each of the elements in the
            element_list
        max_elements: int
            Maximum number of elements in alloy.
        seed: int
            Seed for randomization
        node_features: list
            List of node features
        edge_features: list
            List of edge features
        layer_cutoffs: list
            List of layer cutoffs
        adsorbate_symbol: str
            Adsorbate element symbol
        upper_bounds: dict (default = None)
            Dictionary containing elements as keys and composition upper bounds
            (inclusive).
        lower_bounds: dict (default = None)
            Dictionary containing elements as keys and composition lower bounds
            (inclusive).
        """
        self.element_list = np.array(element_list)
        self.prob_list = np.array(prob_list)
        self.max_elements = max_elements
        self.rng = np.random.default_rng(seed)
        self.adsorbate_symbol = adsorbate_symbol

        # Set bounds
        if upper_bounds is not None:
            self.upper_bounds = upper_bounds
        else:
            self.upper_bounds = {}
        if lower_bounds is not None:
            self.lower_bounds = lower_bounds
        else:
            self.lower_bounds = {}

        # Set process_dict
        self.process_dict = {
            "node_features": node_features,
            "edge_features": edge_features,
            "layer_cutoffs": layer_cutoffs,
        }

        # Get current device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def assign_model(self, model):
        """Assign a SlabGCN model to the optimizer.

        The model argument can either be a SlabGCN model instance or path to a
        saved (and pre-trained) model.

        Parameters
        ----------
        model: SlabGCN model or str (path)
            Trained model instance or path to model
        """
        # Check the type of the argument and assign the model to a variable
        if isinstance(model, Model):
            self.model = model
        elif isinstance(model, str) or isinstance(model, PurePath):
            try:
                self.model = Model(model_path=model, load_pretrained=True)
            except FileNotFoundError:
                raise FileNotFoundError("The entered model path is incorrect.")
        else:
            raise ValueError("The model argument should either be a Model")

        # Transfer model to device
        self.model.model.to(self.device)

    def suggest_composition(self, structure):
        """Suggest change in composition of alloy.

        Parameters
        ----------
        structure: Structure
            An initialized object of class Structure.

        Returns
        -------
        dict
            Dictionary with "add" and "remove" keys and elements to be added and
            removed as values.
        """
        # Get list of unique elements in alloy
        n_atoms = structure.n_atoms
        element_comp_dict = structure.element_comp_dict
        # uniq_elements = np.array(list(element_comp_dict.keys()))
        atom_symbols = []
        for e in element_comp_dict:
            atom_symbols.extend([e] * int(np.round(element_comp_dict[e] * n_atoms)))

        # Choose an element to replace it with
        bool_change = False
        while not bool_change:
            # Randomly choose an element from alloy to remove
            remove_idx = self.rng.choice(
                np.arange(len(atom_symbols)), size=1, replace=False
            )
            remove_element = atom_symbols[remove_idx[0]]

            # Randomly choose an element from element list to add
            upd_prob_list = self.prob_list[self.element_list != remove_element]
            upd_prob_list = upd_prob_list / np.sum(upd_prob_list)
            add_element = self.rng.choice(
                self.element_list[self.element_list != remove_element],
                size=1,
                replace=False,
                p=upd_prob_list,
            )
            add_element = add_element[0]

            # Construct test element list
            test_element_list = deepcopy(atom_symbols)
            test_element_list[remove_idx[0]] = add_element

            # Check if the number of elements in alloy is not more than
            # max_element_list
            if not np.unique(test_element_list).shape[0] <= self.max_elements:
                continue

            # Check if composition bounds are satisfied
            test_comp = deepcopy(element_comp_dict)
            test_comp[remove_element] -= 1 / n_atoms
            test_comp[add_element] = test_comp.get(add_element, 0) + 1 / n_atoms
            if not (np.sum(list(test_comp.values())) - 1) < 1e-3:
                raise ValueError("Compositions don't sum to one after switch.")

            bound_check = True
            for e in self.lower_bounds.keys():
                if not test_comp.get(e, 1) >= self.lower_bounds[e]:
                    bound_check = False
            for e in self.upper_bounds.keys():
                if not test_comp.get(e, 0) <= self.upper_bounds[e]:
                    bound_check = False

            if not bound_check:
                continue

            # If all conditions satisfied, control moves here
            bool_change = True

        # Return element choices
        return {"remove": remove_element, "add": add_element}

    def predict_score(self, list_of_structures, scoring_function=lambda x: np.mean(x)):
        """Predict the score for a list of structures.

        Parameters
        ----------
        list_of_structures: list of Structure objects or ase.Atoms objects
            List of structures to be scored by the model
        scoring_function: function
            Scoring function. By default it returns (mean - std) of outputs

        Returns
        -------
        dict
            Dictionary with outputs and score as keys and corresponding values as
            a list and a float.
        """
        if isinstance(list_of_structures[0], Structure):
            list_of_atoms = [s.structure for s in list_of_structures]
        else:
            list_of_atoms = list_of_structures

        # Create datapoints
        datapoints = load_datapoints(
            atoms=list_of_atoms, process_dict=self.process_dict
        )

        # Make prediction
        outputs = self.model.predict(
            dataset=datapoints,
            indices=np.arange(len(datapoints)),
        )

        # Calculate score
        score = scoring_function(outputs["predictions"])

        return {"outputs": outputs["predictions"], "score": score}
    
    def predict_coverage(self, list_of_structures, T, P_ratio, n_surface_configs, kB=kB):
        """Predict the stable coverage from a given list of structures. 

        Parameters
        ----------
        list_of_structures: list of Structure objects or ase.Atoms objects
            List of structures to be scored by the model
        T: float
            Temperature (K)
        P_ratio: float
            Pressure ratio
        n_surface_configs: int
            Number of unique surface configurations considered

        Returns
        -------
        N_ads_stable: float
            Number of adsorbates stable on the surface
        """
        if isinstance(list_of_structures[0], Structure):
            list_of_atoms = [s.structure for s in list_of_structures]
        else:
            list_of_atoms = list_of_structures

        # Get number of adsorbate atoms in each structure
        num_ads_atoms = np.array([
            atoms.get_chemical_symbols().count(self.adsorbate_symbol) for atoms in list_of_atoms
        ])
        num_total_atoms = np.array([len(atoms) for atoms in list_of_atoms])
        num_surf_atoms = (num_total_atoms - num_ads_atoms)  / 2
        # Assume constant number of surface atoms
        num_surf_atoms = num_surf_atoms[0]

        # Create datapoints
        datapoints = load_datapoints(
            atoms=list_of_atoms, process_dict=self.process_dict
        )

        # Make prediction
        outputs = self.model.predict(
            dataset=datapoints,
            indices=np.arange(len(datapoints)),
        )
        predictions = outputs["predictions"]
        preds_stacked = np.vstack(predictions)
        
        # Stratify into adsorption and surface energies
        ads_eng_uncorr = preds_stacked[:, 0].astype(np.float128)
        surf_eng = preds_stacked[:, 1].astype(np.float128)

        # Correct adsorption energies
        ads_eng = ads_eng_uncorr + mu_ads(T) - (mu0_H2S(T) - mu0_H(T) + mu_gas(T, P_ratio))

        # Calculate exponents
        kB_eV = kB / e
        ads_exp = np.exp(- num_ads_atoms * ads_eng / (kB_eV * T))
        surf_exp = np.exp(- num_surf_atoms * surf_eng / (kB_eV * T))

        # Calculate coverage
        numerator = np.sum(
            num_ads_atoms * ads_exp * surf_exp
        )
        denominator = np.sum(
            ads_exp * surf_exp
        )
        # Add term for zero coverage in denominator
        surf_eng_den = np.array_split(
            surf_eng[num_ads_atoms == num_ads_atoms.min()],
            n_surface_configs
        )
        surf_eng_den_mean = np.array(
            [np.mean(arr) for arr in surf_eng_den], dtype=np.float128
        ) 
        surf_exp_den = np.exp(- num_surf_atoms * surf_eng_den_mean / (kB_eV * T))
        denominator += np.sum(surf_exp_den)
        N_ads_stable = numerator / denominator

        return {"N_ads_stable": N_ads_stable,
                "E_ads": ads_eng_uncorr,
                "E_surf": surf_eng}
    
    def predict_coverage_grid(
            self,
            list_of_structures,
            T_range,
            P_ratio_range,
            n_surface_configs,
            kB=kB,
            n_points=100
    ):
        """Predict the stable coverage from a given list of structures. 

        Parameters
        ----------
        list_of_structures: list of Structure objects or ase.Atoms objects
            List of structures to be scored by the model
        T_range: list or np.ndarray
            Range of temperatures (K) (min and max)
        P_ratio_range: list or np.ndarray
            Range of pressure ratios (min and max)
        n_surface_configs: int
            Number of unique surface configurations considered
        n_points: int (default = 50)
            Number of points to add between the min and max of T and P_ratio

        Returns
        -------
        dict
            Dict containing "N_ads_stable_grid" (grid containing
            number of stable adsorbates for each T (X-axis) and
            P_ratio (Y-axis)), "T" (temperature array), "P_ratio" (pressure ratio
            array).
        """
        if isinstance(list_of_structures[0], Structure):
            list_of_atoms = [s.structure for s in list_of_structures]
        else:
            list_of_atoms = list_of_structures

        # Get number of adsorbate atoms in each structure
        num_ads_atoms = np.array([
            atoms.get_chemical_symbols().count(self.adsorbate_symbol) for atoms in list_of_atoms
        ])
        num_total_atoms = np.array([len(atoms) for atoms in list_of_atoms])
        num_surf_atoms = (num_total_atoms - num_ads_atoms)  / 2
        # Assume constant number of surface atoms
        num_surf_atoms = num_surf_atoms[0]

        # Create datapoints
        datapoints = load_datapoints(
            atoms=list_of_atoms, process_dict=self.process_dict
        )

        # Make prediction
        outputs = self.model.predict(
            dataset=datapoints,
            indices=np.arange(len(datapoints)),
        )
        predictions = outputs["predictions"]
        preds_stacked = np.vstack(predictions)
        
        # Stratify into adsorption and surface energies
        ads_eng_uncorr = preds_stacked[:, 0].astype(np.float128)
        surf_eng = preds_stacked[:, 1].astype(np.float128)

        # Create meshgrid
        P_ratio_arr = np.logspace(
            np.amin(np.log10(P_ratio_range)),
            np.log10(np.amax(P_ratio_range)),
            n_points,
            endpoint=True
        )
        T_arr = np.linspace(
            np.amin(T_range), np.amax(T_range), n_points
        )
        P_grid, T_grid = np.meshgrid(P_ratio_arr, T_arr, indexing="ij")
        P_grid = np.flip(P_grid, axis=0)
        N_ads_stable_grid = np.zeros_like(P_grid)

        # For each T and P
        for i in range(P_grid.shape[0]):
            for j in range(P_grid.shape[1]):
                # Set T and P
                P_ratio = P_grid[i, j]
                T = T_grid[i, j]

                # Correct adsorption energies
                ads_eng = ads_eng_uncorr + mu_ads(T) - (mu0_H2S(T) - mu0_H(T) + mu_gas(T, P_ratio))

                # Calculate exponents
                kB_eV = kB / e
                ads_exp = np.exp(- num_ads_atoms * ads_eng / (kB_eV * T))
                surf_exp = np.exp(- num_surf_atoms * surf_eng / (kB_eV * T))

                # Calculate coverage
                numerator = np.sum(
                    num_ads_atoms * ads_exp * surf_exp
                )
                denominator = np.sum(
                    ads_exp * surf_exp
                )
                # Add term for zero coverage in denominator
                surf_eng_den = np.array_split(
                    surf_eng[num_ads_atoms == num_ads_atoms.min()],
                    n_surface_configs
                )
                surf_eng_den_mean = np.array(
                    [np.mean(arr) for arr in surf_eng_den], dtype=np.float128
                )
                surf_exp_den = np.exp(- num_surf_atoms * surf_eng_den_mean / (kB_eV * T))
                denominator += np.sum(surf_exp_den)
                N_ads_stable_grid[i, j] = numerator / denominator

        return {
            "N_ads_stable_grid": N_ads_stable_grid,
            "T": T_grid,
            "P_ratio": P_grid,
            "E_ads": ads_eng_uncorr,
            "E_surf": surf_eng
        }
    
    def generate_optimal_surfaces_with_swaps(self, T, atoms, surf_idx, n_configs, seed):
        """Randomly generate surface configurations for a given alloy structure.

        Parameters
        ----------
        T: float
            Temperature (K)
        atoms: ase.Atoms object
            Alloy structure as an Atoms object
        surf_idx: list or array
            Indices of surface atoms
        n_configs: int
            Number of surface configurations to be generate
        seed: int
            Seed for random number generator.

        Returns
        -------
        surface_configs: list of ase.Atoms objects
            Alloy structures with varying surfaces.

        """
        # Get list of symbols
        uniq_symbols = np.unique(atoms.get_chemical_symbols())
        surf_dict = {symbol: [] for symbol in uniq_symbols}
        for idx in surf_idx:
            surf_dict[atoms[idx].symbol].append(idx)

        # Create random number generator
        rng = np.random.default_rng(seed)

        # Create copy of atoms object
        atoms_copy = deepcopy(atoms)
        surface_configs = [deepcopy(atoms_copy)]
        # Add a dummy adsorbate
        add_adsorbate(atoms_copy, self.adsorbate_symbol, 2.0, (3., 2.))

        # Create modified process dict
        mod_proc_dict = deepcopy(self.process_dict)
        mod_proc_dict["silence"] = True

        # Get constants
        kB_eV = kB / e

        # Shuffle symbols, assign to surface, create atoms object, save, and repeat
        n_mc_steps = 10 * n_configs
        energies = []
        for i in range(n_mc_steps):
            # Randomly choose one index
            idx1 = rng.choice(surf_idx, size=1)

            # Remove indices from surf_idx corresponding to the element that was
            # chosen above. This is to speed up the MCMC. (We don't want to pick
            # atoms corresponding to the same element to swap.)
            rem_sym = atoms_copy[idx1[0]].symbol
            rem_idx = surf_dict[rem_sym] + [idx1[0]]
            choose_idx = [i for i in surf_idx if i not in rem_idx]

            # Choose second index
            idx2 = rng.choice(choose_idx, size=1)

            # Make test structure
            test_atoms = deepcopy(atoms_copy)

            # Make prediction
            datapoints = load_datapoints(
                atoms=test_atoms, process_dict=mod_proc_dict
            )
            pred = self.model.predict(datapoints, indices=[0])["predictions"]
            E_init = pred[0][1]

            # Swap symbols
            sym_0 = deepcopy(atoms_copy[idx1[0]].symbol)
            sym_1 = deepcopy(atoms_copy[idx2[0]].symbol)
            test_atoms[idx1[0]].symbol = sym_1
            test_atoms[idx2[0]].symbol = sym_0

            # Make energy prediction
            datapoints = load_datapoints(
                atoms=test_atoms, process_dict=mod_proc_dict
            )
            pred = self.model.predict(datapoints, indices=[0])["predictions"]
            E_fin = pred[0][1]

            # Accept structure if Boltzmann criterion is satisfied
            rand_no = rng.uniform()
            if rand_no < np.exp( - (E_fin - E_init) / (kB_eV * T)):
                atoms_copy = deepcopy(test_atoms)
                atoms_save = deepcopy(atoms_copy)
                del atoms_save[-1]
                surface_configs.append(atoms_save)
                energies.append(E_fin)

        if len(surface_configs) > n_configs:
            print(np.array(energies) - energies[0])
            low_energy_idx = np.argsort(energies)
            low_energy_configs = [surface_configs[i] for i in low_energy_idx[:n_configs]]
            return low_energy_configs
        else:
            return surface_configs
        
    def generate_optimal_surfaces(self, atoms, surf_idx, n_configs, seed):
        """Generate optimal surface configurations for a given alloy structure.

        Parameters
        ----------
        atoms: ase.Atoms object
            Alloy structure as an Atoms object
        surf_idx: list or array
            Indices of surface atoms
        n_configs: int
            Number of surface configurations to be generate
        seed: int
            Seed for random number generator.

        Returns
        -------
        surface_configs: list of ase.Atoms objects
            Alloy structures with varying surfaces.

        """
        # Get list of symbols
        surf_symbols = [atoms[i].symbol for i in surf_idx]

        # Create random number generator
        rng = np.random.default_rng(seed)

        # Create modified process dict
        mod_proc_dict = deepcopy(self.process_dict)
        mod_proc_dict["silence"] = True

        # Create copy of atoms object
        atoms_copy = deepcopy(atoms)
        # Add a dummy adsorbate
        add_adsorbate(atoms_copy, self.adsorbate_symbol, 2.0, (3., 2.))

        # Shuffle symbols, assign to surface, create atoms object, save, and repeat
        n_mc_steps = 50 * n_configs
        energies = []
        surface_configs = []
        for i in range(n_mc_steps):
            rng.shuffle(surf_symbols)
            for j, idx in enumerate(surf_idx):
                atoms_copy[idx].symbol = surf_symbols[j]

            # Make prediction
            datapoints = load_datapoints(
                atoms=atoms_copy, process_dict=mod_proc_dict
            )
            pred = self.model.predict(datapoints, indices=[0])["predictions"]
            E_surf = pred[0][1]
            
            atoms_save = deepcopy(atoms_copy)
            del atoms_save[-1]
            surface_configs.append(atoms_save)
            energies.append(E_surf)
        
        if len(surface_configs) > n_configs:
            low_energy_idx = np.argsort(energies)
            low_energy_configs = [surface_configs[i] for i in low_energy_idx[:n_configs]]
            return low_energy_configs
        else:
            return surface_configs
        
    def return_stable_ads_configs(self, list_of_structures, top_k=5):
        """Return most stable adsorbate arrangements.

        Parameters
        ----------
        list_of_structures: list of Structure objects or ase.Atoms objects
            List of structures to be scored by the model

        Returns
        -------
        dict
            Most stable structures and their energies.
            
        """
        if isinstance(list_of_structures[0], Structure):
            list_of_atoms = [s.structure for s in list_of_structures]
        else:
            list_of_atoms = list_of_structures

        # Create datapoints
        datapoints = load_datapoints(
            atoms=list_of_atoms, process_dict=self.process_dict
        )

        # Make prediction
        outputs = self.model.predict(
            dataset=datapoints,
            indices=np.arange(len(datapoints)),
        )
        predictions = outputs["predictions"]
        preds_stacked = np.vstack(predictions)

        # Stratify into adsorption and surface energies
        ads_eng = preds_stacked[:, 0]

        if top_k < ads_eng.shape[0]:
            min_ads_idx = np.argsort(ads_eng)[:top_k]
        else:
            min_ads_idx = np.argsort(ads_eng)

        # Return atoms
        most_stable_ads = [list_of_structures[i] for i in min_ads_idx]
        most_stable_ads_engs = [ads_eng[i] for i in min_ads_idx]

        return {"structures": most_stable_ads, "energies": most_stable_ads_engs}


class Structure:
    """Class to define and modify structues in optimization runs.

    It has the following attributes:
    1. structure (Atoms object)
    2. bulk_idx (bulk partition indices)
    3. surf_idx (surface partition indices)
    4. ads_idx (adsorbate partition indices)
    5. n_atoms (number of atoms)
    6. element_comp_dict (dictionary with elements as keys and comps as values)
    7. element_idx_dict (dictionary with elements as keys and indices as values)
    8. adsorbate_dict (dictionary with adsorbates as keys and counts as values)
    9. lattice_constant (lattice constant of alloy)
    10. n_layer_atoms (number of atoms in each layer)
    11. layer_cutoffs (layer cutoffs)
    12. adsorbate_symbol (adsorbate symbol)
    13. bulk_partition, surf_partition, and ads_partition (indices of partitions)

    """

    def __init__(
        self,
        structure,
        n_layer_atoms,
        adsorbate_symbol,
        layer_cutoffs=[],
        surf_partition=None,
        bulk_partition=None,
    ):
        """Set the structure.

        Parameters
        ----------
        structure: Atoms object or path to structure file
            Path to either an ASE Atoms object or path to a file that can be
            read as an Atoms object using ASE.
        layer_cutoffs: list
            Layer cutoffs to split structure into partitions
        n_layer_atoms: int
            Number of atoms in each layer
        adsorbate_symbol: str
            Symbol of the adsorbate in the structure
        surf_partition: int (default = 0)
            The index of the surface partition in which atoms are to be swapped
        bulk_partition: int (default = 1)
            The index of the bulk partition.
        """
        # Set structure
        if isinstance(structure, Atoms):
            self.structure = structure
        elif isinstance(structure, str) or isinstance(structure, PurePath):
            self.structure = read(structure)

        # Check indices
        if bulk_partition is None:
            self.bulk_partition = 0
        else:
            self.bulk_partition = bulk_partition
        if surf_partition is None:
            self.surf_partition = 1
        else:
            self.surf_partition = surf_partition

        # Remove adsorbates
        self.adsorbate_symbol = adsorbate_symbol
        remove_idx = []
        for atom in self.structure:
            if atom.symbol == adsorbate_symbol:
                remove_idx.append(atom.index)
        del self.structure[remove_idx]

        # Get partition indices
        self.layer_cutoffs = layer_cutoffs
        part_idx = partition_structure_by_layers(self.structure, layer_cutoffs)
        self.surf_idx = part_idx[surf_partition]
        self.bulk_idx = part_idx[bulk_partition]

        # Get adsorbates
        # adsorbates = []
        # for idx in self.ads_idx:
        #     adsorbates.append(self.structure[idx].symbol)
        # unique_adsorbates = np.unique(adsorbates)
        # self.adsorbate_dict = {
        #     ads: adsorbates.count(ads) for ads in unique_adsorbates
        # }

        # Get list of elements in alloy
        alloy_symbols = np.array([self.structure[i].symbol for i in self.bulk_idx])
        self.n_atoms = len(alloy_symbols)
        uniq_elements, count_elements = np.unique(alloy_symbols, return_counts=True)
        self.element_comp_dict = dict(zip(uniq_elements, count_elements / self.n_atoms))

        # Get element indices
        self.element_idx_dict = {e: [] for e in uniq_elements}
        for atom in self.structure:
            self.element_idx_dict[atom.symbol].append(atom.index)

        # Get lattice constant
        self.n_layer_atoms = n_layer_atoms
        lat_ratio = np.linalg.norm(self.structure.get_cell()[1, :]) / np.linalg.norm(
            self.structure.get_cell()[0, :]
        )
        n_x = np.sqrt(n_layer_atoms / lat_ratio)
        self.lattice_constant = (
            np.linalg.norm(self.structure.get_cell()[0, :]) * np.sqrt(2) / n_x
        )


def change_slab_lattice_constant(atoms, old_lattice_constant, new_lattice_constant):
    """Change lattice constant of a slab.

    Parameters
    ----------
    atoms: ase.Atoms object
        The slab whose lattice constant is to be changed
    old_lattice_constant: float
        Lattice constant of the slab
    new_lattice_constant: float
        New lattice constant of the slab

    Returns
    -------
    new_atoms: ase.Atoms object
        New slab with new lattice constant
    """
    # Make a copy of the structure
    new_atoms = deepcopy(atoms)

    # Change lattice constant
    lat_ratio = new_lattice_constant / old_lattice_constant
    new_cell = new_atoms.get_cell().copy()
    new_cell[0, :] = new_cell[0, :] * lat_ratio
    new_cell[1, :] = new_cell[1, :] * lat_ratio
    new_atoms.set_cell(new_cell, scale_atoms=True)

    return new_atoms


def change_composition(structure, change_dict, vegards_law=VegardsLaw):
    """Change compoition of structure.

    This method changes the composition by (1) changing the lattice constant
    and (2) switching atoms in the surface and bulk according to change_dict.

    Parameters
    ----------
    structure: Structure
        An object of class Structure whose composition is to be changed.
    change_dict: dict
        Dictionary with "add" and "remove" indicating the elements to be
        added and removed respectively.
    vegards_law: function
        Function that takes in dictionary of elements (with their pure element
        symbols as keys and compositions as values) and returns an alloy
        lattice constant. By default, the VegardsLaw function defined in
        constants.py is used (only works for Pd-Pt-Rh-Au-Cu).

    Returns
    -------
    new_structure: Structure
        An object of class Structure with changed composition and lattice constant.

    """
    # New comp_dict (add and remove elements and reevaluate compositions)
    new_comp_dict = deepcopy(structure.element_comp_dict)
    new_comp_dict[change_dict["remove"]] -= 1 / structure.n_atoms
    new_comp_dict[change_dict["add"]] = (
        new_comp_dict.get(change_dict["add"], 0) + 1 / structure.n_atoms
    )

    # Calculate new lattice constant
    new_lattice_constant = vegards_law(new_comp_dict)

    # Adjust structure to match new lattice constant
    new_atoms = change_slab_lattice_constant(
        structure.structure, structure.lattice_constant, new_lattice_constant
    )

    # Determine indices of atoms to be removed
    bulk_remove_idx = [
        i
        for i in structure.element_idx_dict[change_dict["remove"]]
        if i in structure.bulk_idx
    ]
    surf_remove_idx = [
        i
        for i in structure.element_idx_dict[change_dict["remove"]]
        if i in structure.surf_idx
    ]
    bulk_remove_atom = np.random.choice(bulk_remove_idx, size=1, replace=False)
    surf_remove_atom = np.random.choice(surf_remove_idx, size=1, replace=False)

    # Swap atoms
    new_atoms.symbols[bulk_remove_atom] = change_dict["add"]
    new_atoms.symbols[surf_remove_atom] = change_dict["add"]

    # Create new Structure
    new_structure = Structure(
        structure=new_atoms,
        n_layer_atoms=structure.n_layer_atoms,
        layer_cutoffs=structure.layer_cutoffs,
        surf_partition=structure.surf_partition,
        bulk_partition=structure.bulk_partition,
        adsorbate_symbol=structure.adsorbate_symbol,
    )

    return new_structure


def generate_surface_configurations(atoms, surf_idx, n_configs, seed):
    """Randomly generate surface configurations for a given alloy structure.

    Parameters
    ----------
    atoms: ase.Atoms object
        Alloy structure as an Atoms object
    surf_idx: list or array
        Indices of surface atoms
    n_configs: int
        Number of surface configurations to be generate
    seed: int
        Seed for random number generator.

    Returns
    -------
    surface_configs: list of ase.Atoms objects
        Alloy structures with varying surfaces.

    """
    # Get list of symbols
    surf_symbols = [atoms[i].symbol for i in surf_idx]

    # Create random number generator
    rng = np.random.default_rng(seed)

    # Create copy of atoms object
    atoms_copy = deepcopy(atoms)

    # Shuffle symbols, assign to surface, create atoms object, save, and repeat
    surface_configs = []
    for i in range(n_configs):
        rng.shuffle(surf_symbols)
        for j, idx in enumerate(surf_idx):
            atoms_copy[idx].symbol = surf_symbols[j]
        surface_configs.append(deepcopy(atoms_copy))

    return surface_configs


def generate_adsorbate_configurations(
    atoms,
    adsorbate_symbol,
    adsorbed_atom_symbol=None,
    sites=["hollow", "ontop", "bridge"],
    n_adsorbates=[1]
):
    """Generate adsorption configurations using Pymatgen.

    Parameters
    ----------
    atoms: ase.Atoms object
        Surface slab on which adsorbates are to be placed
    adsorbate_symbol: str
        The symbol of the adsorbate
    adsorbed_atom_symbol: str (default = None)
        The symbol of the atom (in the adsorbat) that bonds with the surface.
        If single atom adsorbate is specified, adsorbed_atom_symbol is automatically
        changed to the adsorbate_symbol
    sites: list or str
        The sites to be considered for adsorption
    n_adsorbates: int or list
        Number of adsorbates to be placed

    """
    # Check type of site
    if isinstance(sites, str):
        sites = [sites]

    # Check number of adsorbates
    if isinstance(n_adsorbates, int):
        n_adsorbates = [n_adsorbates]

    # Create pymatgen structure
    pym_slab = pym_ase.get_structure(atoms)

    # Create adsorbate
    mol = molecule(adsorbate_symbol)

    # Check adsorbed atom
    if adsorbed_atom_symbol is None:
        if len(adsorbate_symbol) == 1:
            adsorbed_atom_symbol = adsorbate_symbol
        else:
            raise ValueError("Enter adsorbed_atom_symbol.")

    # Create Pymatgen Molecule
    _adspos = []
    for atom in mol:
        if atom.symbol == adsorbed_atom_symbol:
            _adspos = atom.position
            break

    for atom in mol:
        atom.position -= _adspos

    bool_array = [atom.position[-1] < -0.001 for atom in mol]

    species_array = [atom.symbol for atom in mol]
    pos_array = [atom.position for atom in mol]
    pym_mol = Molecule(species=species_array, coords=pos_array)

    # If molecule is upside down, reflect it
    if True in bool_array:
        _reflect = SymmOp.reflection([0, 0, 1], [0, 0, 0])
        pym_mol.apply_operation(_reflect)

    # Create ASE molecule object again
    atoms_mol = pym_ase.get_atoms(pym_mol)

    # Generate list of sites
    asf = AdsorbateSiteFinder(pym_slab)
    found_sites_dict = asf.find_adsorption_sites(positions=sites)
    found_sites_list = np.array(found_sites_dict["all"])
    found_sites_range = np.arange(len(found_sites_list))

    # Get covalent radius of adsorbate
    atom_num = atomic_numbers[adsorbed_atom_symbol]
    cov_rad = covalent_radii[atom_num]

    # Get surface cell
    surf_cell = atoms.get_cell().copy()

    # For each number of adsorbates, pick sites and place adsorbate(s) there
    final_ads_structs = []
    for n in n_adsorbates:
        # Generate combinations
        combs = itertools.combinations(found_sites_range, n)

        # Go over each combination and place adsorbates
        for comb in combs:
            # Check if none of the positions overlap
            dist_check = True
            for i in range(len(comb) - 1):
                for j in range(i + 1, len(comb)):
                    # Distance between adsorbates
                    _dist1 = np.linalg.norm(found_sites_list[comb[i]] - found_sites_list[comb[j]])
                    _dist2 = np.linalg.norm(found_sites_list[comb[i]] + surf_cell[0, :] - found_sites_list[comb[j]])
                    _dist3 = np.linalg.norm(found_sites_list[comb[i]] - found_sites_list[comb[j]] - surf_cell[0, :])
                    _dist4 = np.linalg.norm(found_sites_list[comb[i]] + surf_cell[1, :] - found_sites_list[comb[j]]) 
                    _dist5 = np.linalg.norm(found_sites_list[comb[i]] - found_sites_list[comb[j]] - surf_cell[1, :])
                    _dist6 = np.linalg.norm(found_sites_list[comb[i]] + surf_cell[0, :] + surf_cell[1, :] - found_sites_list[comb[j]])
                    _dist7 = np.linalg.norm(found_sites_list[comb[i]] - found_sites_list[comb[j]] - surf_cell[0, :] - surf_cell[1, :])
                    _dist = np.amin([_dist1, _dist2, _dist3, _dist4, _dist5, _dist6, _dist7])                       

                    if _dist < 2 * cov_rad:
                        dist_check = False
                        break
                if not dist_check:
                    break
                
            if not dist_check:
                continue
                

            # Get atoms object
            ads_slab = atoms.copy()
            for k in range(len(comb)):
                # Create adsorbate object(s)
                ads = atoms_mol.copy()

                # Update position
                for atom in ads:
                    atom.position += np.array(found_sites_list[comb[k]])              
                    ads_slab.append(atom) 
                
            # Save structure
            final_ads_structs.append(ads_slab)

    # Remove structures with overlap
    good_ads_structs = []
    for struct in final_ads_structs:
        _pos = struct.positions
        _unq_pos = np.unique(struct.positions.round(decimals=8), axis=0)
        if _pos.shape[0] == _unq_pos.shape[0]:
            good_ads_structs.append(struct)

    # Save structures
    return good_ads_structs

def generate_adsorbate_configurations_metal(
    atoms,
    adsorbate_symbol,
    adsorbed_atom_symbol=None,
    sites=["hollow", "ontop", "bridge"],
    n_adsorbates=[1]
):
    """Generate adsorption configurations for pure metal surfaces using Pymatgen.

    Parameters
    ----------
    atoms: ase.Atoms object
        Surface slab on which adsorbates are to be placed
    adsorbate_symbol: str
        The symbol of the adsorbate
    adsorbed_atom_symbol: str (default = None)
        The symbol of the atom (in the adsorbat) that bonds with the surface.
        If single atom adsorbate is specified, adsorbed_atom_symbol is automatically
        changed to the adsorbate_symbol
    sites: list or str
        The sites to be considered for adsorption
    n_adsorbates: int or list
        Number of adsorbates to be placed

    """
    # Check type of site
    if isinstance(sites, str):
        sites = [sites]

    # Check number of adsorbates
    if isinstance(n_adsorbates, int):
        n_adsorbates = [n_adsorbates]

    # Create pymatgen structure
    pym_slab = pym_ase.get_structure(atoms)

    # Create adsorbate
    mol = molecule(adsorbate_symbol)

    # Check adsorbed atom
    if adsorbed_atom_symbol is None:
        if len(adsorbate_symbol) == 1:
            adsorbed_atom_symbol = adsorbate_symbol
        else:
            raise ValueError("Enter adsorbed_atom_symbol.")

    # Create Pymatgen Molecule
    _adspos = []
    for atom in mol:
        if atom.symbol == adsorbed_atom_symbol:
            _adspos = atom.position
            break

    for atom in mol:
        atom.position -= _adspos

    bool_array = [atom.position[-1] < -0.001 for atom in mol]

    species_array = [atom.symbol for atom in mol]
    pos_array = [atom.position for atom in mol]
    pym_mol = Molecule(species=species_array, coords=pos_array)

    # If molecule is upside down, reflect it
    if True in bool_array:
        _reflect = SymmOp.reflection([0, 0, 1], [0, 0, 0])
        pym_mol.apply_operation(_reflect)

    #Create sites
    list_of_slabs=[pym_slab]
    final_ads_structs=[]
    for n in range(np.amax(n_adsorbates)):
        ads_structs_cov=[]
        for slab in list_of_slabs:
            asf=AdsorbateSiteFinder(slab)
            try:
                pym_ads_structs=asf.generate_adsorption_structures(
                    pym_mol,
                    translate=False,
                    find_args={"positions":sites})
            except (TypeError, ValueError):
                continue

            ads_structs_cov.extend(pym_ads_structs)
            
            if n+1 in n_adsorbates:
                ads_structs=[pym_ase.get_atoms(struct) for struct in pym_ads_structs]
                final_ads_structs.append(ads_structs)

        list_of_slabs=ads_structs_cov

    final_ads_structs=[slab for slab_list in final_ads_structs for slab in slab_list]
    
    #Remove structures with overlap
    good_structs=[]
    for i,struct in enumerate(final_ads_structs):
        _pos=struct.positions
        _unq_pos=np.unique(struct.positions.round(decimals=8),axis=0)
        if _pos.shape[0]==_unq_pos.shape[0]:
            good_structs.append(i)

    good_ads_structs=[final_ads_structs[i] for i in range(len(final_ads_structs)) if i in good_structs]

    unique_structs=[]
    for i,struct in enumerate(good_ads_structs):
        _pym=pym_ase.get_structure(struct)
        bool_match=False
        for s in unique_structs:
            if _pym.matches(pym_ase.get_structure(s)):
                bool_match=True
                break
        if not bool_match: 
            unique_structs.append(struct)


    #Save structures
    return unique_structs


def save_results(dir_path, iteration, structs, coverage, E_ads, E_surf):
    """Save optimization results.

    Parameters
    ----------
    dir_path: str or Path
        Path to the results directory
    iteration: int
        Iteration number
    structs: list
        List of Structure of ase.Atoms objects
    coverage: float
        Predicted stable coverage
    E_ads: np.ndarray
        Adsorption energies
    E_surf: np.ndarray
        Surface energies
    """
    if isinstance(dir_path, str):
        dir_path = Path(dir_path).resolve()
    if isinstance(structs[0], Structure):
        atoms_list = [s.structure for s in structs]
    else:
        atoms_list = structs

    # Make directory
    iter_dir_path = dir_path / str(iteration)
    iter_dir_path.mkdir(exist_ok=True)

    # Make structs directory
    struct_dir_path = iter_dir_path / "structures"
    struct_dir_path.mkdir(exist_ok=True)

    # Make outputs directory
    outputs_dir_path = iter_dir_path / "outputs"
    outputs_dir_path.mkdir(exist_ok=True)

    # Make score directory
    coverage_dir_path = iter_dir_path / "coverage"
    coverage_dir_path.mkdir(exist_ok=True)

    # Save structures
    traj = TrajectoryWriter(struct_dir_path / "structures.traj", mode="w")
    for atoms in atoms_list:
        traj.write(atoms)
    traj.close()

    # Save outputs
    df = pd.DataFrame({"E_ads": E_ads, "E_surf": E_surf})
    df.to_csv(outputs_dir_path / "outputs.csv", header=None, index=None)

    # Save score
    df = pd.DataFrame({"coverage": [coverage]})
    df.to_csv(coverage_dir_path / "coverage.csv", header=None, index=None)


def rev_Boltzmann_criterion(E_f, E_i, T, rand_no):
    """Check if the reverse Boltzmann criterion is satisfied.

    Parameters
    ----------
    E_f: float
        Final energy
    E_i: float
        Initial energy
    T: float
        Temperature
    rand_no: float
        Random number between 0 and 1

    """
    kB = 8.617e-5  # eV/K
    prob = min(np.exp(-(E_i - E_f) / (kB * T)), 1)
    if rand_no < prob:
        return True
    else:
        return False

def predict_phase_diagram(
        config_path,
        phase_dir_path,
        model,
        structure,
        seed,
        method="montecarlo"
):
    """Predict the phase diagram for a given structure.

    The configuration (JSON) file must containg the following definitions.
    1. layer_cutoffs: Cutoffs for bulk, surface, and adsorbate partitions (list)
    2. n_layer_atoms: Number of atoms in each layer (int)
    3. surf_partition: The index of the surface partition (int)
    4. bulk_partition: The index of the bulk partition (int)
    5. ads_partition: The index of the adsorbate partition (int).
    6. adsorbate_symbol: String containing symbol of adsorbate (str)
    7. sites: list of adsorption sites (list)
    8. n_configs: number of surface configurations to be considered (int)
    9. node_features: list of node features (list)
    10. edge_features: list of edge features (list)
    11. T: Range of temperatures (list)
    12. max_coverage: maximum coverage to be considered for adsorption (float)
    13. P_ratio: Range of ratio of pressures of H2S and H2 (list)

    Parameters
    ----------
    config_path: str
        Path to the configuration file (JSON)
    phase_dir_path: str
        Path to results directory. If it does not exist, it is created.
    model: Model or str (path)
        A pre-trained Model instance or path to a pre-trained model
    init_structure: ase.Atoms or str
        Initial structure as an Atoms object or path to a file that can be read
        as an Atoms object.
    seed: int
        Seed for random number generator.
    method: str (default = "montecarlo")
        Method used to sample surfaces. If "montecarlo", then the SlabGCN model
        is used to perform a Monte-Carlo simulation to choose the n_configs
        with the lowest energies. If "random", then n_configs are chosen randomly.
        It is recommended to use "montecarlo".
    
    """
    # First, import the config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Make phase dir path
    # Create path
    phase_dir_path = Path(phase_dir_path)
    phase_dir_path.mkdir(parents=True, exist_ok=True)

    # Create Optimizer
    optimizer = Optimizer(
        element_list=[],
        prob_list=[],
        max_elements=1,
        seed=seed,
        node_features=config["node_features"],
        edge_features=config["edge_features"],
        layer_cutoffs=config["layer_cutoffs"],
        adsorbate_symbol=config["adsorbate_symbol"],
        upper_bounds=config.get("upper_bounds"),
        lower_bounds=config.get("lower_bounds"),
    )

    # Assign model
    optimizer.assign_model(model)

    # Initialize the structure
    structure = Structure(
        structure=structure,
        n_layer_atoms=config["n_layer_atoms"],
        layer_cutoffs=config["layer_cutoffs"],
        surf_partition=config["surf_partition"],
        bulk_partition=config["bulk_partition"],
        adsorbate_symbol=config["adsorbate_symbol"],
    )

    # Generate surface configurations
    if method.lower() == "random":
        surf_structs = generate_surface_configurations(
            structure.structure,
            surf_idx=structure.surf_idx,
            n_configs=config["n_configs"],
            seed=seed,
        )
    elif method.lower() == "montecarlo":
        surf_structs = optimizer.generate_optimal_surfaces(
            atoms=structure.structure,
            surf_idx=structure.surf_idx,
            n_configs=config["n_configs"],
            seed=seed,
        )
    else:
        raise ValueError("method has to be either 'montecarlo' or 'random'.")

    # Generate adsorbate configurations
    max_coverage = config["max_coverage"]
    n_layer_atoms = config["n_layer_atoms"]
    max_layer_atoms = int(np.floor(max_coverage * n_layer_atoms))
    n_adsorbates = np.arange(1, max_layer_atoms + 1, 1)
    ads_structs = []
    
    # Get the right adsorption function
    if len(np.unique(structure.structure.get_chemical_symbols())) == 1:
        ads_func = generate_adsorbate_configurations_metal
    else:
        ads_func = generate_adsorbate_configurations

    for struct in surf_structs:
        ads = ads_func(
            atoms=struct,
            adsorbate_symbol=config["adsorbate_symbol"],
            sites=config["sites"],
            n_adsorbates=n_adsorbates
        )
        ads_structs.extend(ads)

    # Predict coverage grid
    cov_dict = optimizer.predict_coverage_grid(
        ads_structs,
        config["T"],
        config["P_ratio"],
        config["n_configs"]
    )
    coverage_grid = cov_dict["N_ads_stable_grid"] / n_layer_atoms

    # Dump matrix
    np.savetxt(os.path.join(phase_dir_path, "coverage.txt"), coverage_grid)

    # Make phase diagram
    make_phase_diagram(
        coverage_grid,
        cov_dict["T"],
        cov_dict["P_ratio"],
        max_coverage=max(max_coverage, 0.5),
        save_path=phase_dir_path / "phase_diagram.png"
    )


def run_optimization(config_path, results_dir_path, model, init_structure, seed):
    """Run the optimization workflow.

    The configuration (JSON) file must contain the following definitions.
    1. layer_cutoffs: Cutoffs for bulk, surface, and adsorbate partitions (list)
    2. n_layer_atoms: Number of atoms in each layer (int)
    3. surf_partition: The index of the surface partition (int)
    4. bulk_partition: The index of the bulk partition (int)
    5. ads_partition: The index of the adsorbate partition (int).
    6. element_list: List of elements from which to sample from (list).
    7. max_elements: Maximum number of elements in alloy (int).
    9. upper_bounds: Dictionary containing elements as keys and composition
                    upper bounds (inclusive) (dict)
    9. lower_bounds: Dictionary containing elements as keys and composition lower
                    bounds (inclusive) (dict)
    10. adsorbate_symbol: String containing symbol of adsorbate (str)
    11. sites: list of adsorption sites (list)
    12. n_configs: number of surface configurations to be considered (int)
    13. node_features: list of node features (list)
    14. edge_features: list of edge features (list)
    15. n_steps: number of optimization steps (int)
    16. T: temperature for Boltzmann criterion (float)
    17. max_coverage: maximum coverage to be considered for adsorption (float)
    18. P_ratio: ratio of pressures of H2S and H2 (float)
    19. tol: tolerance for coverage optimization (float) (optional: default is 0.01)
    20. prob_list: list of probabilitie of choosing each of the elements in the pool (list)

    Parameters
    ----------
    config_path: str
        Path to the configuration file (JSON)
    results_dir_path: str
        Path to results directory. If it does not exist, it is created.
    model: Model or str (path)
        A pre-trained Model instance or path to a pre-trained model
    init_structure: ase.Atoms or str
        Initial structure as an Atoms object or path to a file that can be read
        as an Atoms object.
    seed: int
        Seed for random number generator.
    """
    # First, import the config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Make results dir path
    # Create path
    results_dir_path = Path(results_dir_path)
    results_dir_path.mkdir(parents=True, exist_ok=True)

    # Create Optimizer
    optimizer = Optimizer(
        element_list=config["element_list"],
        prob_list=config["prob_list"],
        max_elements=config["max_elements"],
        seed=seed,
        node_features=config["node_features"],
        edge_features=config["edge_features"],
        layer_cutoffs=config["layer_cutoffs"],
        adsorbate_symbol=config["adsorbate_symbol"],
        upper_bounds=config.get("upper_bounds"),
        lower_bounds=config.get("lower_bounds"),
    )

    # Assign model
    optimizer.assign_model(model)

    # Initialize the structure
    structure = Structure(
        structure=init_structure,
        n_layer_atoms=config["n_layer_atoms"],
        layer_cutoffs=config["layer_cutoffs"],
        surf_partition=config["surf_partition"],
        bulk_partition=config["bulk_partition"],
        adsorbate_symbol=config["adsorbate_symbol"],
    )

    # Generate surface configurations
    # surf_structs = generate_surface_configurations(
    #     structure.structure,
    #     surf_idx=structure.surf_idx,
    #     n_configs=config["n_configs"],
    #     seed=seed,
    # )
    surf_structs = optimizer.generate_optimal_surfaces(
        atoms=structure.structure,
        surf_idx=structure.surf_idx,
        n_configs=config["n_configs"],
        seed=seed
    )

    # Generate adsorbate configurations
    max_coverage = config["max_coverage"]
    n_layer_atoms = config["n_layer_atoms"]
    max_layer_atoms = int(np.floor(max_coverage * n_layer_atoms))
    n_adsorbates = np.arange(1, max_layer_atoms + 1, 1)
    ads_structs = []
    for struct in surf_structs:
        ads = generate_adsorbate_configurations(
            atoms=struct,
            adsorbate_symbol=config["adsorbate_symbol"],
            sites=config["sites"],
            n_adsorbates=n_adsorbates
        )
        ads_structs.extend(ads)

    # Get initial score
    cov_dict = optimizer.predict_coverage_grid(
        ads_structs,
        config["T"],
        config["P_ratio"],
        config["n_configs"]
    )
    #coverage_stable = np.linalg.norm(cov_dict["N_ads_stable_grid"] / n_layer_atoms)
    coverage_stable = np.mean(cov_dict["N_ads_stable_grid"] / n_layer_atoms)
    E_ads = cov_dict["E_ads"]
    E_surf = cov_dict["E_surf"]
    save_results(results_dir_path, 0, ads_structs, coverage_stable, E_ads, E_surf)

    # Iterate over steps
    n_steps = config["n_steps"]
    tol = config.get("tol", 1.0)
    for i in range(1, n_steps + 1):
        # Suggest new composition
        change_dict = optimizer.suggest_composition(structure)

        # Change composition
        new_structure = change_composition(
            structure=structure,
            change_dict=change_dict,
        )

        # Generate surface configurations
        # new_surf_structs = generate_surface_configurations(
        #     atoms=new_structure.structure,
        #     surf_idx=new_structure.surf_idx,
        #     n_configs=config["n_configs"],
        #     seed=seed,
        # )      
        new_surf_structs = optimizer.generate_optimal_surfaces(
            atoms=new_structure.structure,
            surf_idx=new_structure.surf_idx,
            n_configs=config["n_configs"],
            seed=seed
        )

        # Generate adsorbate configurations
        new_ads_structs = []
        for struct in new_surf_structs:
            new_ads = generate_adsorbate_configurations(
                atoms=struct,
                adsorbate_symbol=config["adsorbate_symbol"],
                sites=config["sites"],
                n_adsorbates=n_adsorbates
            )
            new_ads_structs.extend(new_ads)

        # Get score
        new_cov_dict = optimizer.predict_coverage_grid(
            new_ads_structs,
            config["T"],
            config["P_ratio"],
            config["n_configs"]
        )
        #new_coverage_stable = np.linalg.norm(new_cov_dict["N_ads_stable_grid"] / n_layer_atoms)
        new_coverage_stable = np.mean(new_cov_dict["N_ads_stable_grid"] / n_layer_atoms)
        # If score is higher than previous, change current structure
        if (new_coverage_stable < coverage_stable) or (
          np.abs(new_coverage_stable - coverage_stable) < tol
        ):
            # Get energies
            E_ads_new = new_cov_dict["E_ads"]
            E_surf_new = new_cov_dict["E_surf"]

            # Save results
            save_results(
                results_dir_path,
                iteration=i,
                structs=new_ads_structs,
                coverage=new_coverage_stable,
                E_ads=E_ads_new,
                E_surf=E_surf_new
            )
            coverage_stable = deepcopy(new_coverage_stable)
            structure = deepcopy(new_structure)

def predict_adsorbate_pattern(
        config_path,
        structure,
        model,
        n_adsorbates,
        seed,
        method,
        top_k=5
):
    """Predict the most stable adsorbate arrangements for the given coverage.

    The composition is inferred from the given structure and n_configs number of
    surface configurations are generated. Then, adsorbates are placed on the
    configurations based on the given coverage. Adsorption energies
    are predicted by the given model and the most stable adsorbate arrangements
    are returned.

    The configuration (JSON) file must contain the following definitions.
    1. layer_cutoffs: Cutoffs for bulk, surface, and adsorbate partitions (list)
    2. n_layer_atoms: Number of atoms in each layer (int)
    3. surf_partition: The index of the surface partition (int)
    4. bulk_partition: The index of the bulk partition (int)
    5. ads_partition: The index of the adsorbate partition (int).
    6. adsorbate_symbol: String containing symbol of adsorbate (str)
    7. sites: list of adsorption sites (list)
    8. n_configs: number of surface configurations to be considered (int)
    9. node_features: list of node features (list)
    10. edge_features: list of edge features (list)
    
    Parameters
    ----------
    config_path: str
        Path to the configuration file (JSON)
    structure: ase.Atoms or str
        Structure as an Atoms object or path to a file that can be read
        as an Atoms object.
    model: Model or str (path)
        A pre-trained Model instance or path to a pre-trained model
    n_adsorbates: int
        Number of adsorbates (proxy for coverage)
    seed: int
        Seed for random number generator.
    method: str
        Method for generating optimal surfaces ("montecarlo" or "random")
    top_k: int
        The number of stable configurations to return.

    Returns
    -------
    most_stable_ads: list of ase.Atoms
        List of Atoms objects representing the most stable arrangements
    """
    # First, import the config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Create Optimizer
    optimizer = Optimizer(
        element_list=[],
        prob_list=[],
        max_elements=1,
        seed=seed,
        node_features=config["node_features"],
        edge_features=config["edge_features"],
        layer_cutoffs=config["layer_cutoffs"],
        adsorbate_symbol=config["adsorbate_symbol"],
        upper_bounds=config.get("upper_bounds"),
        lower_bounds=config.get("lower_bounds"),
    )

    # Assign model
    optimizer.assign_model(model)

    # Initialize the structure
    structure = Structure(
        structure=structure,
        n_layer_atoms=config["n_layer_atoms"],
        layer_cutoffs=config["layer_cutoffs"],
        surf_partition=config["surf_partition"],
        bulk_partition=config["bulk_partition"],
        adsorbate_symbol=config["adsorbate_symbol"],
    )

    # Generate surface configs
    if method.lower() == "random":
        surf_structs = generate_surface_configurations(
            structure.structure,
            surf_idx=structure.surf_idx,
            n_configs=config["n_configs"],
            seed=seed,
        )
    elif method.lower() == "montecarlo":
        surf_structs = optimizer.generate_optimal_surfaces(
            atoms=structure.structure,
            surf_idx=structure.surf_idx,
            n_configs=config["n_configs"],
            seed=seed,
        )
    else:
        raise ValueError("method has to be either 'montecarlo' or 'random'.")

    # Generate adsorbate configurations
    ads_structs = []
    for struct in surf_structs:
        ads = generate_adsorbate_configurations(
            atoms=struct,
            adsorbate_symbol=config["adsorbate_symbol"],
            sites=config["sites"],
            n_adsorbates=n_adsorbates
        )
        ads_structs.extend(ads)

    # Predict structures
    most_stable_ads_dict = optimizer.return_stable_ads_configs(ads_structs, top_k=top_k)

    return most_stable_ads_dict


def make_plots(results_dir_path):
    """Make plots of results for the optimization run.

    The plots are stored under the "plots" directory in results_dir_path.

    Parameters
    ----------
    results_dir_path: str or PurePath
        Path to the optimization results directory.

    """
    # Make plots directory
    plots_dir_path = Path(results_dir_path) / "plots"
    plots_dir_path.mkdir(exist_ok=True)

    # Collect scores vs steps
    steps = []
    scores = []
    for root, dirs, files in os.walk(results_dir_path):
        for d in dirs:
            if d not in "plots":
                # Append step
                steps.append(int(d))

                # Go to the steps directory
                score_csv = os.path.join(results_dir_path, d, "coverage", "coverage.csv")
                with open(score_csv, "r") as f:
                    scores.append(float(f.read()))
        break

    # Make arrays
    sort_order = np.argsort(steps)
    steps_arr = np.array(steps)[sort_order]
    scores_arr = np.array(scores)[sort_order]

    # Plot of score against step
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=500)
    ax.plot(steps_arr, scores_arr, color="black", marker="o", linestyle="--")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig_path = plots_dir_path / "coverage_vs_step.png"
    fig.savefig(fig_path)


if __name__ == "__main__":
    seed = 0
    unit_cell = "2x2" #"3x3"
    constraint = "PtCu" #"opt_PdPtCu" #"PdPtCuRh"s

    model_path = REPO_PATH / "trained_models" / "S_all_highcov" / "models" / "best.pt"
    #model_path = REPO_PATH / "trained_models" / "S_binary_highcov" / "models" / "best.pt"
    init_struct_path = REPO_PATH / "data" / "S_calcs" / "Pt_6_Cu_6_-0-1-S.cif"
    #init_struct_path = REPO_PATH / "data" / "S_calcs_3x3_all" / "Pt_9_Cu_9_-0-1-S.cif"
    #init_struct_path = REPO_PATH / "data" / "S_calcs_ternary" / "Pd_3_Pt_3_Rh_6_-0-0-S.cif"
    #init_struct_path = REPO_PATH / "data" / "S_misc" / "opt_PdPtRhCu.cif"
    # config_path = REPO_PATH / "config.json"
    # save_path = REPO_PATH / "opt_results" / f"{seed}-{unit_cell}-{constraint}"
    # run_optimization(
    #     config_path=config_path,
    #     results_dir_path=save_path,
    #     model=model_path,
    #     init_structure=init_struct_path,
    #     seed=seed,
    # )
    # make_plots(results_dir_path=save_path)

    # Phase diagram
    config_path = REPO_PATH / "config_phase.json"
    save_path = REPO_PATH / "phase_results" / f"{seed}-{unit_cell}-{constraint}"
    predict_phase_diagram(
        config_path,
        save_path,
        model_path,
        init_struct_path,
        seed,
        method="montecarlo"
    )
