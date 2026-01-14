"""Train and test the model."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .models import SlabGCN
from .plot_utils import make_parity_plot, make_loss_plot
from .graphs import AtomsGraph
from .utils import partition_structure_by_layers

class Standardizer:
    """Class to standardize targets."""

    def __init__(self):
        """
        Class to standardize outputs.

        Initialize with dummy values.
        """
        self.mean = 0
        self.std = 0.1

    def initialize(self, X):
        """Initialize mean and std based on the given tensor.

        Parameters
        ----------
        X: torch.Tensor
            Tensor of outputs
        """
        self.mean = torch.mean(X, dim=0)
        self.std = torch.std(X, dim=0)

    def standardize(self, X):
        """
        Convert a non-standardized output to a standardized output.

        Parameters
        ----------
        X: torch.Tensor
            Tensor of non-standardized outputs

        Returns
        -------
        Z: torch.Tensor
            Tensor of standardized outputs

        """
        Z = (X - self.mean) / (self.std)
        return Z

    def restore(self, Z):
        """
        Restore a standardized output to the non-standardized output.

        Parameters
        ----------
        Z: torch.Tensor
            Tensor of standardized outputs

        Returns
        -------
        X: torch.Tensor
            Tensor of non-standardized outputs

        """
        X = self.mean + Z * self.std
        return X

    def get_state(self):
        """
        Return dictionary of the state of the Standardizer.

        Returns
        -------
        dict
            Dictionary with the mean and std of the outputs

        """
        return {"mean": self.mean, "std": self.std}

    def set_state(self, state):
        """
        Load a dictionary containing the state of the Standardizer.

        Parameters
        ----------
        state : dict
            Dictionary containing mean and std
        """
        self.mean = state["mean"]
        self.std = state["std"]


class Model:
    """Wrapper class for a SlabGCN model that allows training and prediction."""

    def __init__(
        self,
        global_config=None,
        partition_configs=None,
        model_path=None,
        load_pretrained=None,
    ):
        """Initialize or load pretrained model.

        If load_pretrained is True, only model_path needs to be supplied. By default,
        the best model is loaded. If load_pretrained is False, all of global_config,
        partition_configs, and model_path need to be specified.
        """
        if load_pretrained:
            if model_path is None:
                raise ValueError("Enter model_path to load a pretrained model.")
            else:
                load_dict = Model.load_from_path(model_path)

                # Set configs
                global_config = load_dict["global_config"]
                partition_configs = load_dict["partition_configs"]

                # Initialize
                model_path = Path(model_path).resolve().parents[1]
                self.initialize_model(global_config, partition_configs, model_path)
                self.load(best_status=True)

        else:
            self.initialize_model(global_config, partition_configs, model_path)

    def initialize_model(self, global_config, partition_configs, model_path):
        """Initialize a SlabGCN model.

        Parameters
        ----------
        global_config: dict
            Global configuration dictionary. Should contain the following keys:
            gpu (whether to use GPU, bool), optimizer (name of optimizer, str; can
            either be "adam" or "sgd"), learning_rate (model learning rate, float),
            lr_milestones (milestones when learning rate is to be decresed, list;
            optional)
        partition_configs: List[Dict]
            List of dictionaries containing parameters for the GNN for each
            partition. The number of different GNNs are judged based on the
            size of the list. Each partition config should contain the following
            keys: n_conv (number of convolutional layers, int), n_hidden (number
            of hidden layers, int), conv_size (feature size before convolution, int)
            hidden_size (nodes per hidden layer node, int), dropout (dropout
            probability for hidden layers, float), conv_type (type of convolution
            layer, str; currently only "CGConv" is supported), pool_type
            (type of pooling layer, str; currently "add" and "mean" are supported),
            num_node_features (number of node features, int), num_edge_features
            (number of edge features, int).
        model_path: str
            Path where the model is to be saved
        """
        # Create model
        self.global_config = global_config
        self.partition_configs = partition_configs
        self.model = SlabGCN(partition_configs, global_config)

        # Create model path
        self.make_directory_structure(model_path)

        # Set GPU status
        self.use_gpu = global_config["gpu"]
        if self.use_gpu:
            self.model.cuda()
        # Set loss function
        if global_config["loss_function"] == "mse":
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError(
                "Incorrect loss function. Currently only 'mse' is supported"
            )

        # Set metric function
        if global_config["metric_function"] == "mae":
            self.metric_fn = mean_absolute_error
        elif global_config["metric_function"] == "mse":
            self.metric_fn = mean_squared_error

        # Set optimizer
        if global_config["optimizer"].lower().strip() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=global_config["learning_rate"],
                weight_decay=global_config["learning_rate"] * 1e-3
            )
        elif global_config["optimizer"].lower().strip() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=global_config["learning_rate"],
                weight_decay=global_config["learning_rate"] * 1e-3
            )

        # Set scheduler
        if "lr_milestones" in global_config.keys():
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer, milestones=global_config["lr_milestones"]
            )
        else:
            self.scheduler = None

        # Set standardizer
        self.standardizer = Standardizer()

        # Set number of outputs
        self.n_outputs = global_config["n_outputs"]

    def make_directory_structure(self, model_path):
        """Make directory structure to store models and results."""
        self.model_path = Path(model_path)
        self.model_save_path = self.model_path / "models"
        self.model_results_path = self.model_path / "results"
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.model_results_path.mkdir(parents=True, exist_ok=True)

    def init_standardizer(self, targets):
        """Initialize the Standardizer using training targets (typically).

        Parameters
        ----------
        targets: np.ndarray or torch.Tensor
            Array of training outputs
        """
        self.standardizer.initialize(torch.stack(targets, dim=0))

    def __str__(self):
        model_text = (
            f"---------- SlabGCN model ----------\n"
            + f"Number of trainable parameters: {self.model.get_number_of_parameters()}\n"
            + f"Using GPU: {self.use_gpu}\n"
        )
        return model_text

    def train_epoch(self, dataloader):
        """Train the model for a single epoch.

        Parameters
        ----------
        dataloader: torch_geometric.loader.DataLoader
            Training dataloader
        """
        # Variables to store average stats
        avg_loss = 0
        avg_metric = 0
        count = 0

        # Enable train mode of model
        self.model.train()

        # Go over each batch in the dataloader
        for data_objects in dataloader:
            # Standardize output
            y = data_objects[0].y
            y = y.reshape(-1, self.n_outputs)
            y_std = self.standardizer.standardize(y)

            # Transfer to GPU (if True)
            if self.use_gpu:
                nn_output = y_std.cuda()
                nn_input = [d.cuda() for d in data_objects]
            else:
                nn_output = y_std
                nn_input = data_objects

            # Compute prediction
            pred_dict = self.model(nn_input)

            # Calculate loss
            loss = self.loss_fn(pred_dict["output"], nn_output)
            avg_loss += loss

            # Calculate metric
            y_pred = self.standardizer.restore(pred_dict["output"].cpu().detach())
            metric = self.metric_fn(y, y_pred)
            avg_metric += metric

            # Set zero gradient for all the tensors
            self.optimizer.zero_grad()

            # Perform backward propagation
            loss.backward()

            # Update weights and biases
            self.optimizer.step()

            # Update scheduler if not None
            if self.scheduler is not None:
                self.scheduler.step()

            # Increase count
            count += 1

        # Calculate average loss and metric
        avg_loss = avg_loss / count
        avg_metric = avg_metric / count

        return avg_loss, avg_metric

    def validate(self, dataloader):
        """Validate/test the model.

        Parameters
        ----------
        dataloader: torch_geometric.loader.DataLoader
            Validation/test dataloader
        """
        # Variables to store average stats
        avg_loss = 0
        avg_metric = 0
        count = 0

        # Enable train mode of model
        self.model.eval()

        # Go over each batch in the dataloader
        for data_objects in dataloader:
            # Standardize output
            y = data_objects[0].y
            y = y.reshape(-1, self.n_outputs)
            y_std = self.standardizer.standardize(y)

            # Transfer to GPU (if True)
            if self.use_gpu:
                nn_output = y_std.cuda()
                nn_input = [d.cuda() for d in data_objects]
            else:
                nn_output = y_std
                nn_input = data_objects

            # Compute prediction
            pred_dict = self.model(nn_input)

            # Calculate loss
            loss = self.loss_fn(pred_dict["output"], nn_output)
            avg_loss += loss

            # Calculate metric
            y_pred = self.standardizer.restore(pred_dict["output"].cpu().detach())
            metric = self.metric_fn(y, y_pred)
            avg_metric += metric

            # Increase count
            count += 1

        # Calculate average loss and metric
        avg_loss = avg_loss / count
        avg_metric = avg_metric / count

        return avg_loss, avg_metric

    def predict(self, dataset, indices, return_targets=False):
        """Predict outputs from the model.

        Parameters
        ----------
        dataset: AtomsDataset or AtomsDatapoints
            Validation dataloader
        indices: list or np.ndarray
            List of indices for datapoints for which predictions are to be made
        return_targets: bool (default = False)
            If True, outputs are returned. If False, all targets will be 0.

        Returns
        -------
        prediction_dict: dict
            Dictionary containing "targets", "predictions", and
            "indices" (copy of predict_idx).
        """
        # Create arrays
        n_partitions = len(dataset.get(indices[0]))
        targets = []
        predictions = []

        # Enable eval mode of model
        self.model.eval()

        # Go over each batch in the dataloader
        for i, idx in enumerate(indices):
            # Get data objects
            data_objects = dataset.get(idx)

            # Standardize output
            if return_targets:
                targets.append(data_objects[0].y.cpu().numpy())

            # Transfer to GPU (if True)
            if self.use_gpu:
                nn_input = [d.cuda() for d in data_objects]
            else:
                nn_input = data_objects

            # Compute prediction
            pred_dict = self.model(nn_input)
            pred = self.standardizer.restore(pred_dict["output"].cpu().detach())
            predictions.append(pred.squeeze().numpy())

        # Check if predictions is 1D
        if hasattr(predictions[0], '__len__') and (not isinstance(predictions[0], str)):
            predictions = [np.array([p]) for p in predictions]
        # Flatten subarrays in predictions
        predictions = [p.flatten() for p in predictions]

        predictions_dict = {
            "targets": targets,
            "predictions": predictions,
            "indices": indices,
        }

        return predictions_dict

    def predict_embeddings(self, dataset, indices):
        """Predict outputs from the model.

        Parameters
        ----------
        dataset: AtomsDataset or AtomsDatapoints
            Validation dataloader
        indices: list or np.ndarray
            List of indices for datapoints for which predictions are to be made

        Returns
        -------
        embeddings_dict: dict
            Dictionary containing "embeddings" and "indices" (copy of predict_idx).
        """
        # Create arrays and dicts
        n_partitions = len(dataset.get(indices[0]))
        embeddings_dict = {"indices": indices}
        for i in range(n_partitions):
            embeddings_dict[f"embeddings_{i}"] = []

        # Enable eval mode of model
        self.model.eval()

        # Go over each batch in the dataloader
        for i, idx in enumerate(indices):
            # Get data objects
            data_objects = dataset.get(idx)

            # Transfer to GPU (if True)
            if self.use_gpu:
                nn_input = [d.cuda() for d in data_objects]
            else:
                nn_input = data_objects

            # Compute prediction
            embeds = self.model.get_embeddings(nn_input)
            for i in range(len(embeds)):
                embeddings_dict[f"embeddings_{i}"].append(
                    embeds[i].cpu().detach().numpy()
                )

        return embeddings_dict
    
    def predict_attention(self, dataset, indices):
        """Predict attention weights for the adsorbate graph.

        Parameters
        ----------
        dataset: AtomsDataset or AtomsDatapoints
            Validation dataloader
        indices: list or np.ndarray
            List of indices for datapoints for which predictions are to be made

        Returns
        -------
        attention_dict: dict
            Dictionary containing "embeddings" and "indices".
        
        """
        # Handle attention partition
        n_partitions = len(dataset.get(indices[0]))

        # Create arrays and dicts
        attention_dict = {"indices": indices, "bond_indices": [], "attention": []}

        # Enable eval mode of model
        self.model.eval()

        # Go over each batch in the dataloader
        for i, idx in enumerate(indices):
            # Get data objects
            data_objects = dataset.get(idx)
            
            # # Get atoms object and graph
            # map_idx_node = dataset.get_map(idx)[attention_partition]
            # map_node_idx = dict(
            #     (value, key) for key, value in map_idx_node.items()
            # )

            # Transfer to GPU (if True)
            if self.use_gpu:
                nn_input = [d.cuda() for d in data_objects]
            else:
                nn_input = data_objects

            # Compute prediction
            att_tuple = self.model.get_attention(nn_input)

            # Get edge tensor
            edge_idx = att_tuple[0].cpu().detach().numpy()
            edge_idx_tuples = []
            for i in range(edge_idx.shape[1]):
                edge_idx_tuples.append(edge_idx[:, i])
            attention_dict["bond_indices"].append(edge_idx_tuples)
            
            # Get attention value
            att_value = att_tuple[1].cpu().detach().numpy()
            attention_dict["attention"].append(att_value)

        return attention_dict


    def save(self, epoch, best_status=None):
        """Save the current state of the model as a dictionary.

        The dictionary contains the epoch, model state dict, optimizer state dict,
        standardizer state dict, and configs.

        Parameters
        ----------
        epoch: int
            Current epoch
        best_status: bool
            If True, this is also saved as "best.pt".
        """
        save_dict = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "standardizer_state_dict": self.standardizer.get_state(),
            "global_config": self.global_config,
            "partition_configs": self.partition_configs,
        }
        save_path = self.model_save_path / f"model_{epoch}.pt"
        torch.save(save_dict, save_path)
        if best_status:
            save_path = self.model_save_path / "best.pt"
            torch.save(save_dict, save_path)

    def load(self, epoch=None, best_status=None):
        """Load a model saved at a particular epoch or the best model.

        If best_status is True, epoch is ignored and the best model is loaded.

        Parameters
        ----------
        epoch: int
            Model at this epoch is loaded
        best_status: bool
            If this is True, the best model is loaded
        """
        # Load path
        if best_status:
            load_path = self.model_save_path / "best.pt"
        else:
            load_path = self.model_save_path / f"model_{epoch}.pt"

        # Load the dictionary
        load_dict = torch.load(load_path)

        # Set state dicts
        self.model.load_state_dict(load_dict["model_state_dict"])
        self.standardizer.set_state(load_dict["standardizer_state_dict"])

        # Set configs
        self.global_config = load_dict["global_config"]
        self.partition_configs = load_dict["partition_configs"]

    def train(self, epochs, dataloader_dict, verbose=False):
        """Train a model for the given number of epochs.

        The training is performed with early stopping, i.e., the metric function
        is evaluated at every epoch and the model with the best value for this
        metric is loaded after training for testing.

        Parameters
        ----------
        epochs: int
            Total number of epochs
        dataloader_dict: dict
            Dictionary of train, val, and test dataloaders
        verbose: bool
            If True, progress is printed for every epoch.

        Returns
        -------
        results_dict: Dict[Dict]
            Dictionary of dictionaries. The outer dictionary contains the keys
            "loss" and "metric" and the inner dictionaries contain the keys
            "train", "val", and "test".
        """
        # Create empty lists
        train_losses = []
        train_metrics = []
        val_losses = []
        val_metrics = []

        # Initialize validation loss
        prev_val_metric = 1e9
        best_status = False

        # Train and validate model
        for i in range(epochs):
            # Train
            train_loss, train_metric = self.train_epoch(dataloader_dict["train"])

            # Validate
            val_loss, val_metric = self.validate(dataloader_dict["val"])

            # Check if model is best
            if val_metric < prev_val_metric:
                best_status = True
                prev_val_metric = deepcopy(val_metric)
            else:
                best_status = False

            # Save model
            self.save(i, best_status)

            # Save losses and metrics
            train_losses.append(train_loss.cpu().detach().numpy())
            val_losses.append(val_loss.cpu().detach().numpy())
            train_metrics.append(train_metric)
            val_metrics.append(val_metric)

            # Print, if verbose
            if verbose:
                print(
                    f"Epoch: [{i}]  Training loss: [{train_loss:.3f}]  "
                    + f"Training metric: [{train_metric:.3f}]  "
                    + f"Validation loss: [{val_loss:.3f}]  "
                    + f"Validation metric: [{val_metric:.3f}]"
                )

        # Load the best model
        self.load(best_status=True)

        # Test the model
        if len(dataloader_dict["test"]) > 0:
            test_loss, test_metric = self.validate(dataloader_dict["test"])
        else:
            test_loss = 0.0
            test_metric = 0.0

        loss_dict = {"train": train_losses, "val": val_losses, "test": test_loss}
        metric_dict = {"train": train_metrics, "val": val_metrics, "test": test_metric}

        results_dict = {"loss": loss_dict, "metric": metric_dict}

        return results_dict

    def parity_plot(self, dataset, indices):
        """Make a parity plot for the given points.

        Parameters
        ----------
        dataset: AtomsDataset
            An processed dataset
        indices: list or np.ndarray
            Parity will be evaluated for points at the given indices in the dataset
        """
        # Get targets and predictions
        pred_dict = self.predict(dataset, indices, return_targets=True)

        # Make parity plot
        for i in range(self.n_outputs):
            targets = np.array([t[i] for t in pred_dict["targets"]])
            preds = np.array([p[i] for p in pred_dict["predictions"]])
            make_parity_plot(
                targets,
                preds,
                self.model_results_path / f"parity_plot_{i}.png",
            )

    def loss_plot(self, results_dict):
        """Make a plot showing the evolution of training and validation loss.

        Parameters
        ----------
        results_dict: dict
            Results dictionary that is the output of the train method.
        """
        train_losses = results_dict["loss"]["train"]
        val_losses = results_dict["loss"]["val"]

        # Make loss plot
        make_loss_plot(
            train_losses=train_losses,
            val_losses=val_losses,
            save_path=self.model_results_path / "loss_plot.png"
        )

    @staticmethod
    def load_from_path(model_path):
        """Load a model saved at a particular path.

        If best_status is True, epoch is ignored and the best model is loaded.

        Parameters
        ----------
        model_path: str
            Path where the model is stored
        """
        # Load the dictionary
        load_dict = torch.load(model_path)

        return load_dict
