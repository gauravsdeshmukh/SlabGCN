"""Samplers for training, validation, and testing."""

import abc
from copy import deepcopy

import numpy as np


class Sampler(abc.ABC):
    """Abstract base class for data samplers."""

    def __init__(self):
        """Blank constructor."""
        pass

    @abc.abstractmethod
    def create_samplers(self):
        """Create training, test, and validation samplers.

        This should return a dictionary with "train", "val", "test" as keys and
        indices of datapoints as values.
        """
        pass

    @abc.abstractstaticmethod
    def name():
        """Name of the sampling method."""
        pass


class RandomSampler(Sampler):
    """Perform uniform random sampling on datapoints."""

    def __init__(self, seed, dataset_size):
        """Initialize sampler.

        Parameters
        ----------
        seed: int
            Seed for random sampling.
        dataset_size: int
            Number of points in dataset
        """
        self.seed = seed
        self.dataset_size = dataset_size

    def create_samplers(self, sample_config):
        """Randomly sample training, validation, and test datapoints.

        Parameters
        ----------
        sample_config: dict
            Dictionary with "train", "val", "test" as values and corresponding
            fractions as values (must sum up to 1).

        Returns
        -------
        samples: dict
            Dictionary with indices for train, val, and test points.
        """
        # Create randomizer
        randomizer = np.random.default_rng(self.seed)

        # Create array of indices
        idx_array = np.arange(self.dataset_size)

        # Shuffle array
        randomizer.shuffle(idx_array)

        # Get indices
        if sample_config["train"] < 1.0:
            train_size = int(np.ceil(sample_config["train"] * self.dataset_size))
        train_idx = idx_array[:train_size]
        if sample_config["val"] < 1.0:
            val_size = int(np.floor(sample_config["val"] * self.dataset_size))
        val_idx = idx_array[train_size : train_size + val_size]
        test_idx = idx_array[train_size + val_size :]

        # Create samples
        samples = {"train": train_idx, "val": val_idx, "test": test_idx}

        return samples

    @staticmethod
    def name():
        """Name of the sampling method."""
        return "random"


class RandomPropSampler(Sampler):
    """Perform random sampling proportionate to given property on datapoints.

    The sampling is performed such that the distributions of properties in the
    train, val, and test sets are the same as the distribution in the whole
    dataset.
    """

    def __init__(self, seed, dataset_props):
        """Initialize sampler.

        Parameters
        ----------
        seed: int
            Seed for random sampling.
        dataset_size: list or array
            List of properties for each point in the dataset.
        """
        self.seed = seed
        self.dataset_props = np.array(dataset_props)
        self.dataset_size = self.dataset_props.shape[0]

    def create_samplers(self, sample_config):
        """Randomly sample training, validation, and test datapoints.

        Parameters
        ----------
        sample_config: dict
            Dictionary with "train", "val", "test" as values and corresponding
            fractions as values (must sum up to 1).

        Returns
        -------
        samples: dict
            Dictionary with indices for train, val, and test points.
        """
        # Create randomizer
        randomizer = np.random.default_rng(self.seed)

        # Get sizes
        if sample_config["train"] < 1.0:
            train_size = int(np.ceil(sample_config["train"] * self.dataset_size))
        if sample_config["val"] < 1.0:
            val_size = int(np.floor(sample_config["val"] * self.dataset_size))
        if sample_config["test"] < 1.0:
            test_size = int(np.floor(sample_config["test"] * self.dataset_size))

        if train_size + val_size + test_size != self.dataset_size:
            test_size = self.dataset_size - train_size - val_size

        # Find property distributions
        unique_props = np.unique(self.dataset_props)
        prop_counts = []
        for prop in unique_props:
            prop_counts.append(self.dataset_props[self.dataset_props == prop].shape[0])
        prop_fracs = np.array(prop_counts) / self.dataset_size

        # Calculate subset sizes
        train_sizes = []
        val_sizes = []
        test_sizes = []
        for frac in prop_fracs:
            train_sizes.append(int(np.round(frac * train_size)))
            val_sizes.append(int(np.round(frac * val_size)))
            test_sizes.append(int(np.round(frac * test_size)))

        # Check sizes
        if np.sum(train_sizes) != train_size:
            train_sizes[-1] = train_size - np.sum(train_sizes[:-1])
        if np.sum(val_sizes) != val_size:
            val_sizes[-1] = val_size - np.sum(val_sizes[:-1])
        if np.sum(test_sizes) != test_size:
            test_sizes[-1] = test_size - np.sum(test_sizes[:-1])

        # Distribute points
        train_idx = []
        val_idx = []
        test_idx = []
        for i, prop in enumerate(unique_props):
            prop_arg = np.argwhere(self.dataset_props == prop).flatten()
            prop_idx = deepcopy(prop_arg)
            randomizer.shuffle(prop_idx)
            train_idx.append(prop_idx[: train_sizes[i]])
            val_idx.append(prop_idx[train_sizes[i] : train_sizes[i] + val_sizes[i]])
            test_idx.append(prop_idx[train_sizes[i] + val_sizes[i] :])

        # Flatten arrays
        train_idx = np.concatenate(train_idx)
        val_idx = np.concatenate(val_idx)
        test_idx = np.concatenate(test_idx)

        # Create samples
        samples = {"train": train_idx, "val": val_idx, "test": test_idx}

        return samples

    @staticmethod
    def name():
        """Name of the sampling method."""
        return "proportionate_random"
    
class ConstrainedRandomPropSampler(Sampler):
    """Perform constrained random sampling proportionate to given property on datapoints.

    The sampling is performed such that the distributions of properties in the
    train, val, and test sets are the same as the distribution in the whole
    dataset. Additionally, points with 1 in the dataset_cons array will always
    be placed in the training set.
    """

    def __init__(self, seed, dataset_props, dataset_cons):
        """Initialize sampler.

        Parameters
        ----------
        seed: int
            Seed for random sampling.
        dataset_size: list or array
            List of properties for each point in the dataset.
        dataset_cons: list or array
            Dataset constraints
        """
        self.seed = seed
        self.dataset_props = np.array(dataset_props)
        self.dataset_cons = np.array(dataset_cons)
        self.dataset_size = self.dataset_props.shape[0]

    def create_samplers(self, sample_config):
        """Randomly sample training, validation, and test datapoints.

        Parameters
        ----------
        sample_config: dict
            Dictionary with "train", "val", "test" as values and corresponding
            fractions as values (must sum up to 1).

        Returns
        -------
        samples: dict
            Dictionary with indices for train, val, and test points.
        """
        # Create randomizer
        randomizer = np.random.default_rng(self.seed)

        # Get sizes
        if sample_config["train"] < 1.0:
            train_size = int(np.ceil(sample_config["train"] * self.dataset_size))
        if sample_config["val"] < 1.0:
            val_size = int(np.floor(sample_config["val"] * self.dataset_size))
        if sample_config["test"] < 1.0:
            test_size = int(np.floor(sample_config["test"] * self.dataset_size))

        if train_size + val_size + test_size != self.dataset_size:
            test_size = self.dataset_size - train_size - val_size

        # Find property distributions
        unique_props = np.unique(self.dataset_props)
        prop_counts = []
        for prop in unique_props:
            prop_counts.append(self.dataset_props[self.dataset_props == prop].shape[0])
        prop_fracs = np.array(prop_counts) / self.dataset_size

        # Calculate subset sizes
        train_sizes = []
        val_sizes = []
        test_sizes = []
        for frac in prop_fracs:
            train_sizes.append(int(np.round(frac * train_size)))
            val_sizes.append(int(np.round(frac * val_size)))
            test_sizes.append(int(np.round(frac * test_size)))

        # Check sizes
        if np.sum(train_sizes) != train_size:
            train_sizes[-1] = train_size - np.sum(train_sizes[:-1])
        if np.sum(val_sizes) != val_size:
            val_sizes[-1] = val_size - np.sum(val_sizes[:-1])
        if np.sum(test_sizes) != test_size:
            test_sizes[-1] = test_size - np.sum(test_sizes[:-1])

        # Distribute points
        train_idx = []
        val_idx = []
        test_idx = []
        for i, prop in enumerate(unique_props):
            prop_arg = np.argwhere(self.dataset_props == prop).flatten()
            prop_idx = deepcopy(prop_arg)
            randomizer.shuffle(prop_idx)
            train_idx.append(prop_idx[: train_sizes[i]])
            val_idx.append(prop_idx[train_sizes[i] : train_sizes[i] + val_sizes[i]])
            test_idx.append(prop_idx[train_sizes[i] + val_sizes[i] :])

        # Flatten arrays
        train_idx = np.concatenate(train_idx)
        val_idx = np.concatenate(val_idx)
        test_idx = np.concatenate(test_idx)

        # Make final adjustments according to constraint
        for i in range(self.dataset_cons.shape[0]):
            if self.dataset_cons[i] == 1.:
                # First remove this from either of train, val, or test idxs
                del_train_idx = np.argwhere(train_idx == i)
                del_val_idx = np.argwhere(val_idx == i)
                del_test_idx = np.argwhere(test_idx == i)
                train_idx = np.delete(train_idx, del_train_idx)
                val_idx = np.delete(val_idx, del_val_idx)
                test_idx = np.delete(test_idx, del_test_idx)

                # Now add it to either training
                train_idx = np.insert(train_idx, 0, i)

        # Create samples
        samples = {"train": train_idx, "val": val_idx, "test": test_idx}

        return samples

    @staticmethod
    def name():
        """Name of the sampling method."""
        return "constrained_proportionate_random"


if __name__ == "__main__":
    props = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1]
    rs = RandomPropSampler(0, props)
    samples = rs.create_samplers({"train": 0.5, "val": 0.25, "test": 0.25})
