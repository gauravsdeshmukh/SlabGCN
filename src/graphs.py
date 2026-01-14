"""Classes to create bulk, surface, and adsorbate graphs."""

from copy import deepcopy
import torch
import networkx as nx
import numpy as np
from ase.neighborlist import (
    NewPrimitiveNeighborList,
    build_neighbor_list,
    natural_cutoffs,
)


class AtomsGraph:
    """Create graph representation of a collection of atoms."""

    def __init__(
            self,
            atoms,
            select_idx,
            max_atoms=None,
            adsorbate_list=["H", "C", "N", "O", "S"],
            include_neighbors=True
        ):
        """Initialize variables of the class.

        Parameters
        ----------
        atoms: ase.Atoms object
            Atoms object containing all the atoms in the slab
        select_idx: list or np.ndarray
            List of indices of atoms that are to be included in the graph
        neighbor_list: ase.neighborlist.NeighborList object
            Neighbor list that defines bonds between atoms
        max_atoms: int (default = None)
            The maximum number of atoms in the graph. If it is not None, graphs
            that have fewer nodes than max_atoms are padded with 0s to ensure
            that the total number of nodes is equal to max_atoms.
        adsorbate_list: list (default = ["H", "C", "N", "O", "S"])
            List of atoms to be considered adsorbates
        """
        # Save parameters
        self.atoms = atoms
        self.select_idx = select_idx
        self.max_atoms = max_atoms
        self.adsorbate_list = adsorbate_list
        self.include_neighbors = include_neighbors

        # Create graph
        self.create_graph()

    def create_graph(self):
        """Create a graph from an atoms object and neighbor_list."""
        # Create NetworkX Multigraph
        graph = nx.MultiGraph()

        # Iterate over selected atoms and add them as nodes
        self.node_count = 0
        self.map_idx_node = {}
        for atom in self.atoms:
            if atom.index in self.select_idx and atom.index not in list(
                graph.nodes(data="index")
            ):
                graph.add_node(
                    self.node_count,
                    index=atom.index,
                    atomic_number=atom.number,
                    symbol=atom.symbol,
                )
                self.map_idx_node[atom.index] = self.node_count
                self.node_count += 1

        # Create neighbor list of atoms
        self.neighbor_list = build_neighbor_list(
            self.atoms,
            natural_cutoffs(self.atoms),
            bothways=True,
            self_interaction=False,
            primitive=NewPrimitiveNeighborList,
        )

        # Iterate over nodes, identify neighbors, and add edges between them
        node_list = list(graph.nodes())
        bond_tuples = []
        for n in node_list:
            # Get neighbors from neighbor list
            neighbor_idx, neighbor_offsets = self.neighbor_list.get_neighbors(
                graph.nodes[n]["index"]
            )
            # Iterate over neighbors
            for nn, offset in zip(neighbor_idx, neighbor_offsets):
                # Skip if self atom
                if nn == graph.nodes[n]["index"]:
                    continue
                # Save bond
                bond = (graph.nodes[n]["index"], nn)
                rev_bond = tuple(reversed(bond))
                # Check if bond has already been added
                if rev_bond in bond_tuples:
                    continue
                else:
                    bond_tuples.append(bond)
                # If neighbor is not in graph, add it as a node
                node_indices = nx.get_node_attributes(graph, "index")
                if nn not in list(node_indices.values()):
                    if self.include_neighbors:
                        graph.add_node(
                            self.node_count,
                            index=nn,
                            atomic_number=self.atoms[nn].number,
                            symbol=self.atoms[nn].symbol,
                        )
                        self.map_idx_node[nn] = self.node_count
                        self.node_count += 1
                    else:
                        continue
                # Calculate bond distance
                bond_dist = self.calc_minimum_distance(
                    self.atoms[graph.nodes[n]["index"]].position,
                    self.atoms[nn].position,
                    offset,
                )
                if ((graph.nodes[n]["symbol"] in self.adsorbate_list) and\
                    (self.atoms[nn].symbol in self.adsorbate_list)):
                    graph.add_edge(
                        n,
                        self.map_idx_node[nn],
                        bond_distance=1e6,
                        interaction_distance=bond_dist,
                    )
                else:
                    graph.add_edge(
                        n,
                        self.map_idx_node[nn],
                        bond_distance=bond_dist,
                        interaction_distance=1e6,
                    )

        # Add adsorbate interactions
        for i in range(max(graph.nodes()) - 1):
            for j in range(i+1, max(graph.nodes())):
                if (
                    (graph.nodes[i]["symbol"] in self.adsorbate_list) &
                    (graph.nodes[j]["symbol"] in self.adsorbate_list)
                ):
                    # inter_stats = self.calc_distances_and_angles(
                    #     self.atoms.get_positions()[graph.nodes[i]["index"], :],
                    #     self.atoms.get_positions()[graph.nodes[j]["index"], :]
                    # )
                    # for dist, angle_1, angle_2 in inter_stats:
                    #     graph.add_edge(
                    #         i,
                    #         j,
                    #         bond_distance=0.0001,
                    #         interaction_distance=dist,
                    #         theta_1=angle_1,
                    #         theta_2=angle_2
                    #     )
                    min_dist = self.atoms.get_distance(
                        graph.nodes[i]["index"],
                        graph.nodes[j]["index"],
                        mic=True
                    )
                    graph.add_edge(
                        i,
                        j,
                        bond_distance=1e6,
                        interaction_distance=min_dist
                    )

        # Pad graph
        if self.max_atoms is not None:
            graph = self.pad_graph(graph)

        # Add coordination numbers
        for n in graph.nodes():
            graph.nodes[n]["coordination"] = graph.degree[n]

        # Assign graph object
        self.graph = graph

    def pad_graph(self, graph):
        """Pad graph with empty nodes.

        This can be used to make sure that the number of nodes in each graph is
        equal to max_atoms

        Parameters
        ----------
        graph: Networkx.Graph
            A Networkx graph

        Returns
        -------
        padded_graph: Networkx.Graph
            Padded graph
        """
        padded_graph = deepcopy(graph)

        for i in range(self.node_count, self.max_atoms, 1):
            padded_graph.add_node(
                i, index=-1, atomic_number=0, symbol="", position=np.zeros(3)
            )

        return padded_graph

    def calc_minimum_distance(self, pos_1, pos_2, offset):
        """Calculate minimum distance between two atoms.

        Parameters
        ----------
        pos_1: np.ndarray
            Position of first atom in x, y, z coordinates
        pos_2: np.ndarray
            Position of second atom in x, y, z coordinates
        offset: np.ndarray
            Offset returned by the neighbor list in ASE

        Returns
        -------
        min_dist: float
            Minimum distance between adsorbates.
        """
        # First, calculate the distance without offset
        dist_1 = np.linalg.norm(pos_1 - pos_2)

        # Next calculate the distance by applying offset to second position
        dist_2 = np.linalg.norm(pos_1 - (pos_2 + offset @ self.atoms.get_cell()))

        # Get minimum distance
        min_dist = min(dist_1, dist_2)

        return min_dist
    
    def calc_angles(self, pos_1, pos_2, offset):
        """Calculate the angle between two atoms.

        Parameters
        ----------
        pos_1: np.ndarray
            Position of first atom in x, y, z coordinates
        pos_2: np.ndarray
            Position of second atom in x, y, z coordinates
        offset: np.ndarray
            Offset returned by the neighbor list in ASE

        Returns
        -------
        tuple
            Tuple containing two angles.
        """
        # First, calculate the distance without offset
        vec_1 = pos_1 - pos_2
        dist_1 = np.linalg.norm(vec_1)

        # Next calculate the distance by applying offset to second position
        vec_2 = pos_1 - (pos_2 + offset @ self.atoms.get_cell())
        dist_2 = np.linalg.norm(vec_2)

        # Find minimum distance
        min_idx = np.argmin([dist_1, dist_2])

        if min_idx == 0:
            theta = np.arctan2(vec_1[1], vec_1[0]) * 180 / np.pi
            theta_r = np.arctan2(-vec_1[1], -vec_1[0]) * 180 / np.pi
        else:
            theta = np.arctan2(vec_2[1], vec_2[0]) * 180 / np.pi
            theta_r = np.arctan2(-vec_2[1], -vec_2[0]) * 180 / np.pi

        return (theta, theta_r)
    
    def calc_distances_and_angles(self, pos_1, pos_2):
        """Calculate five distances and angles between two adsorbates.

        Parameters
        ----------
        pos_1: np.ndarray
            Position of first atom in x, y, z coordinates
        pos_2: np.ndarray
            Position of second atom in x, y, z coordinates

        Returns
        -------
        list
            List containing tuples of distance, angle_1, angle_2.
        """
        cell = self.atoms.get_cell().copy()
        
        # First distance
        vec_1 = pos_1 - pos_2
        dist_1 = np.linalg.norm(vec_1)
        theta_1 = np.arctan2(vec_1[1], vec_1[0]) * 180 / np.pi
        theta_1r = np.arctan2(-vec_1[1], -vec_1[0]) * 180 / np.pi

        # Second distance
        vec_2 = pos_1 + cell[0, :] - pos_2
        dist_2 = np.linalg.norm(vec_2)
        theta_2 = np.arctan2(vec_2[1], vec_2[0]) * 180 / np.pi
        theta_2r = np.arctan2(-vec_2[1], -vec_2[0]) * 180 / np.pi

        # Third distance
        vec_3 = pos_1 - pos_2 - cell[0, :]
        dist_3 = np.linalg.norm(vec_3)
        theta_3 = np.arctan2(vec_3[1], vec_3[0]) * 180 / np.pi
        theta_3r = np.arctan2(-vec_3[1], -vec_3[0]) * 180 / np.pi

        # Fourth distance
        vec_4 = pos_1 + cell[1, :] - pos_2
        dist_4 = np.linalg.norm(vec_4)
        theta_4 = np.arctan2(vec_4[1], vec_4[0]) * 180 / np.pi
        theta_4r = np.arctan2(-vec_4[1], -vec_4[0]) * 180 / np.pi

        # Fifth distance
        vec_5 = pos_1 - pos_2 - cell[1, :]
        dist_5 = np.linalg.norm(vec_5)
        theta_5 = np.arctan2(vec_5[1], vec_5[0]) * 180 / np.pi
        theta_5r = np.arctan2(-vec_5[1], -vec_5[0]) * 180 / np.pi

        return [
            (dist_1, theta_1, theta_1r),
            (dist_2, theta_2, theta_2r),
            (dist_3, theta_3, theta_3r),
            (dist_4, theta_4, theta_4r),
            (dist_5, theta_5, theta_5r)
        ]

    def plot(self, filename=None):
        """Plot the graph using NetworkX.

        Parameters
        ----------
        filename: str (optional)
            If provided, the plot is saved with the given filename.
        """
        pass


if __name__ == "__main__":
    import os
    from ase.io import read

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    atoms = read(os.path.join(
        parent_dir, "data", "S_calcs_2x2_highcov", "Pd_6_Rh_6_-0-14-S.cif"
        )
    )
    g = AtomsGraph(atoms, select_idx=[24, 25])
    #print(g.map_idx_node)
    print(g.graph.nodes(data="index"))
    print(g.graph.edges(data=True))
 