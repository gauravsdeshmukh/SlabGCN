"""Graph neural network models."""

import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class SlabGCN(nn.Module):
    """Class to customize the graph neural network."""

    def __init__(self, partition_configs, global_config):
        """Initialize the graph neural network.

        Parameters
        ----------
        partition_configs: List[Dict]
            List of dictionaries containing parameters for the GNN for each
            partition. The number of different GNNs are judged based on the
            size of the list. Each partition config should contain the following
            keys: n_conv (number of convolutional layers, int), n_hidden (number
            of hidden layers, int), conv_size (feature size before convolution, int)
            hidden_size (nodes per hidden layer node, int), dropout (dropout
            probability for hidden layers, float), conv_type (type of convolution
            layer, str; currently only "CGConv" is supported), pool_type
            (type of pooling layer, str; currently "add", "mean" nad "max" are supported),
            num_node_features (number of node features, int), num_edge_features
            (number of edge features, int)
        global_config: Dict
            This should contain the following keys: n_hidden (Number of hidden
            layers for the shared FFN, int), hidden_size (Number of nodes per
            hidden layer, int), dropout (Probability of dropping a node, float),
            n_outputs(number of outputs, int).
        """
        super().__init__()

        # Store hyperparameters
        self.n_conv = [config["n_conv"] for config in partition_configs]
        self.n_hidden = global_config["n_hidden"]
        self.hidden_size = global_config["hidden_size"]
        self.conv_size = [config["conv_size"] for config in partition_configs]
        self.conv_type = [config["conv_type"] for config in partition_configs]
        self.pool_type = [config.get("pool_type", "") for config in partition_configs]
        self.dropout = global_config.get("dropout", 0)
        self.num_node_features = [
            config["num_node_features"] for config in partition_configs
        ]
        self.num_edge_features = [
            config["num_edge_features"] for config in partition_configs
        ]
        self.n_partitions = len(partition_configs)
        self.max_conv_size = max(self.conv_size)
        self.device = torch.device("cuda") if global_config["gpu"] else torch.device("cpu")
        self.n_outputs = global_config["n_outputs"]

        # Initialize layers
        # Initial transform
        self.init_transform = nn.ModuleList()
        for i in range(self.n_partitions):
            self.init_transform.append(
                nn.Sequential(
                    nn.Linear(self.num_node_features[i], self.conv_size[i]),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(p=self.dropout)
                )
            )

        # Convolutional layers
        self.init_conv_layers()

        # Pooling layers
        self.pool_layers = nn.ModuleList()
        for i in range(self.n_partitions):
            if self.pool_type[i].lower() in "mean":
                self.pool_layers.append(gnn.aggr.MeanAggregation())
            elif self.pool_type[i].lower() in "add":
                self.pool_layers.append(gnn.aggr.SumAggregation())
            elif self.pool_type[i].lower() in "max":
                self.pool_layers.append(gnn.aggr.MaxAggregation())
            else:
                self.pool_layers.append(gnn.aggr.MeanAggregation())

        # Pool transform
        if self.n_partitions == 1:
            self.pool_transform = nn.Sequential(
                nn.Linear(sum(self.conv_size), self.hidden_size),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(p=self.dropout)
            )
        else:
            # Create transforms to max conv size
            gat_transforms = []
            for i in range(self.n_partitions):
                gat_transform = nn.Sequential(
                    nn.Linear(self.conv_size[i], self.max_conv_size),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(p=self.dropout)
                )
                gat_transforms.append(gat_transform)
            self.gat_transforms = nn.ModuleList(gat_transforms)

            # Create edge matrix
            gat_edge_marix = torch.ones((self.n_partitions, self.n_partitions)) -\
                                torch.eye(self.n_partitions)
            self.gat_edge_indices = torch.argwhere(gat_edge_marix == 1.).transpose(0, 1).to(self.device)

            # Create pool tranform to hidden size
            self.pool_transform = gnn.GATv2Conv(in_channels=self.max_conv_size,
                              out_channels=self.hidden_size,
                              heads=1,
                              concat=False,
                              dropout=self.dropout)
            self.pool_agg = gnn.aggr.MeanAggregation()

        # Hidden layers
        self.hidden_layers = nn.Sequential(
            *(
                [
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(p=self.dropout),
                ]
                * (self.n_hidden - 1)
                + [
                    nn.Linear(self.hidden_size, self.n_outputs),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(p=self.dropout),
                ]
            )
        )

    def init_conv_layers(self):
        """Initialize convolutional layers."""
        self.conv_layers = nn.ModuleList()
        for i in range(self.n_partitions):
            part_conv_layers = []
            for j in range(self.n_conv[i]):
                # TODO Add possibility of changing convolutional layers
                conv_layer = [
                    fetch_conv_layer(
                        name=self.conv_type[i],
                        conv_size=self.conv_size[i],
                        num_edge_features=self.num_edge_features[i],
                        dropout=self.dropout
                    ),
                    nn.Linear(self.conv_size[i], self.conv_size[i]),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(p=self.dropout),
                ]
                part_conv_layers.extend(conv_layer)

            self.conv_layers.append(nn.ModuleList(part_conv_layers))

    def forward(self, data_objects):
        """Foward pass of the network(s).

        Parameters
        ----------
        data_objects: list
            List of data objects, each corresponding to a graph of a partition
            of an atomic structure.

        Returns
        ------
        dict
            Dictionary containing "output" and "contributions".
        """
        # For each data object
        embeddings = []
        for i, data in enumerate(data_objects):
            # Apply initial transform
            conv_data = self.init_transform[i](data.x)

            # Apply convolutional layers
            for layer in self.conv_layers[i]:
                if isinstance(layer, gnn.MessagePassing):
                    conv_data = layer(
                        x=conv_data,
                        edge_index=data.edge_index,
                        edge_attr=data.edge_attr,
                    )
                else:
                    conv_data = layer(conv_data)

            # Apply pooling layer
            pooled_data = self.pool_layers[i](x=conv_data, index=data.batch)
            embeddings.append(pooled_data)

        # Apply pool-to-hidden transform
        if self.n_partitions == 1:
            embedding = torch.cat(embeddings, dim=1)
            hidden_data = self.pool_transform(embedding)
        else:
            # Create pre embeddings
            pre_embeddings = []
            for i in range(self.n_partitions):
                pre_embedding = self.gat_transforms[i](embeddings[i])
                pre_embeddings.append(pre_embedding.unsqueeze(1))

            # Stack to create node tensor
            gat_node_tensor_chunked = torch.cat(pre_embeddings, dim=1)
            gat_node_tensor = gat_node_tensor_chunked.reshape(-1, gat_node_tensor_chunked.shape[-1])
            batch_repeat_factor = int(gat_node_tensor.shape[0] / self.n_partitions)
            gat_edge_indices = self.gat_edge_indices.repeat(1, batch_repeat_factor)
            batch_index = torch.tensor([[i] * self.n_partitions for i in range(batch_repeat_factor)]).to(self.device)
            batch_index = torch.flatten(batch_index).to(self.device)

            # Input to GATConv
            hidden_embedding = self.pool_transform(
                x=gat_node_tensor,
                edge_index=gat_edge_indices,
            )
            hidden_data = self.pool_agg(
               x=hidden_embedding,
               index=batch_index,
            )


        # Apply hidden layers
        output = self.hidden_layers(hidden_data)

        return {"output": output}

    def get_embeddings(self, data_objects):
        """Get the pooled embeddings of each partition.

        Parameters
        ----------
        data_objects: list
            List of data objects, each corresponding to a graph of a partition
            of an atomic structure.

        Returns
        ------
        embeddings: list
            List of embedding tensors.
        """
        # For each data object
        embeddings = []
        for i, data in enumerate(data_objects):
            # Apply initial transform
            conv_data = self.init_transform[i](data.x)

            # Apply convolutional layers
            for layer in self.conv_layers[i]:
                if isinstance(layer, gnn.MessagePassing):
                    conv_data = layer(
                        x=conv_data,
                        edge_index=data.edge_index,
                        edge_attr=data.edge_attr,
                    )
                else:
                    conv_data = layer(conv_data)

            # Apply pooling layer
            pooled_data = self.pool_layers[i](x=conv_data, index=data.batch)
            embeddings.append(pooled_data)

        if self.n_partitions < 100:
            return embeddings
        else:
            # Create pre embeddings
            pre_embeddings = []
            for i in range(self.n_partitions):
                pre_embedding = self.gat_transforms[i](embeddings[i])
                pre_embeddings.append(pre_embedding.unsqueeze(1))

            # Stack to create node tensor
            gat_node_tensor_chunked = torch.cat(pre_embeddings, dim=1)
            gat_node_tensor = gat_node_tensor_chunked.reshape(-1, gat_node_tensor_chunked.shape[-1])
            batch_repeat_factor = int(gat_node_tensor.shape[0] / self.n_partitions)
            gat_edge_indices = self.gat_edge_indices.repeat(1, batch_repeat_factor)
            batch_index = torch.tensor([[i] * self.n_partitions for i in range(batch_repeat_factor)]).to(self.device)
            batch_index = torch.flatten(batch_index).to(self.device)

            # Input to GATConv
            hidden_embedding = self.pool_transform(
                x=gat_node_tensor,
                edge_index=gat_edge_indices,
            )
            hidden_embeddings = [
                hidden_embedding[i, :] for i in range(self.n_partitions)
            ]
            return hidden_embeddings

    def get_number_of_parameters(self, train=True):
        """Get number of parameters in the model.

        If train is True, then only the number of trainable parameters is returned.

        Parameters
        ----------
        train: bool (default = True)
            Whether to return only trainable parameters
        """
        if train:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def freeze_layers(self, freeze_array):
        """Freeze specific layers in the model.

        The freeze_array should contain as many boolean values as the nubmer of
        partitions in the model. A value of "True" freezes the layers in the
        corresponding partition. If "False" is provided, the layers are unfrozen.

        Parameters
        ----------
        freeze_array: list or np.ndarray
            List of values indicating whether each partition should be frozen.
        """
        # Check if the length of freeze_array is equal to number of partitions
        if len(freeze_array) != self.n_partitions:
            raise ValueError("freeze_array should be equal to the number of partitions")

        # Loop over values
        for i, value in enumerate(freeze_array):
            if value is True:
                # Freeze initial layer
                for p in self.init_transform[i].parameters():
                    p.requires_grad = False

                # Freeze convolutional layer
                for p in self.conv_layers[i].parameters():
                    p.requires_grad = False

                # Freeze pooling layer
                for p in self.pool_layers[i].parameters():
                    p.requires_grad = False
            else:
                # Unfreeze initial layer
                for p in self.init_transform[i].parameters():
                    p.requires_grad = True

                # Unfreeze convolutional layer
                for p in self.conv_layers[i].parameters():
                    p.requires_grad = True

                # Unfreeze pooling layer
                for p in self.pool_layers[i].parameters():
                    p.requires_grad = True

    def get_attention(self, data_objects):
        """Get the attention parameter for each edge in the given data objects.

        Parameters
        ----------
        data_object: list
            List of data objects, each corresponding to a graph of a partition
            of an atomic structure.
    
        Returns
        ------
        attention: dict
            List of dicts with the tuple of edge indices as key and the attention
            as value.
        
        """
        # For each data object
        embeddings = []
        for i, data in enumerate(data_objects):
            # Apply initial transform
            conv_data = self.init_transform[i](data.x)

            # Apply convolutional layers
            for layer in self.conv_layers[i]:
                if isinstance(layer, gnn.MessagePassing):
                    conv_data = layer(
                        x=conv_data,
                        edge_index=data.edge_index,
                        edge_attr=data.edge_attr,
                    )
                else:
                    conv_data = layer(conv_data)

            # Apply pooling layer
            pooled_data = self.pool_layers[i](x=conv_data, index=data.batch)
            embeddings.append(pooled_data)

        # Create pre embeddings
        pre_embeddings = []
        for i in range(self.n_partitions):
            pre_embedding = self.gat_transforms[i](embeddings[i])
            pre_embeddings.append(pre_embedding.unsqueeze(1))

        # Stack to create node tensor
        gat_node_tensor_chunked = torch.cat(pre_embeddings, dim=1)
        gat_node_tensor = gat_node_tensor_chunked.reshape(-1, gat_node_tensor_chunked.shape[-1])
        batch_repeat_factor = int(gat_node_tensor.shape[0] / self.n_partitions)
        gat_edge_indices = self.gat_edge_indices.repeat(1, batch_repeat_factor)
        batch_index = torch.tensor([[i] * self.n_partitions for i in range(batch_repeat_factor)]).to(self.device)
        batch_index = torch.flatten(batch_index).to(self.device)

        # Input to GATConv
        _, att_tuple = self.pool_transform(
            x=gat_node_tensor,
            edge_index=gat_edge_indices,
            return_attention_weights=True
        )

        return att_tuple

def fetch_conv_layer(name, conv_size, num_edge_features, dropout):
    """Fetch an intialized convolution layer based on the given inputs.

    Parameters
    ----------
    name: str
        Name of the convolutional layer.
    conv_size: int
        Size of the input to the layer.
    num_edge_features: int
        Dimension of the edge features.
    dropout: float
        Dropout probability

    Returns
    -------
    conv_layer: nn.Module
        Convolution layer
    
    """
    if name.lower() in "cgconv":
        conv_layer = gnn.CGConv(
            channels=conv_size,
            dim=num_edge_features,
            batch_norm=True,
        )
    elif name.lower() in "gatconv":
        conv_layer = gnn.GATv2Conv(
            in_channels=conv_size,
            out_channels=conv_size,
            heads=1,
            concat=False,
            edge_dim=num_edge_features,
            dropout=dropout
        )
    elif name.lower() in "nnconv":
        conv_layer = gnn.NNConv(
            in_channels=conv_size,
            out_channels=conv_size,
            nn=nn.Sequential(
                nn.Linear(num_edge_features, conv_size ** 2),
                nn.LeakyReLU(inplace=True)
            ),
            aggr="add"
        )
    elif name.lower() in "resgatedgraphconv":
        conv_layer = gnn.ResGatedGraphConv(
            in_channels=conv_size,
            out_channels=conv_size,
            edge_dim=num_edge_features,
        )
    
    return conv_layer



if __name__ == "__main__":
    from pathlib import Path

    from ase.io import read

    from constants import REPO_PATH
    from data import AtomsDatapoints

    # Test for one tensor
    # Create datapoins
    data_root_path = Path(REPO_PATH) / "data" / "S_calcs"
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
    data_objects = datapoint.get(0)

    # Get result
    partition_configs = [
        {
            "n_conv": 3,
            "conv_size": 40,
            "num_node_features": data_objects[0].num_node_features,
            "num_edge_features": data_objects[0].num_edge_features,
            "conv_type": "CGConv",
        },
        {
            "n_conv": 3,
            "conv_size": 40,
            "num_node_features": data_objects[1].num_node_features,
            "num_edge_features": data_objects[1].num_edge_features,
            "conv_type": "CGConv",
        },
        {
            "n_conv": 3,
            "conv_size": 40,
            "num_node_features": data_objects[2].num_node_features,
            "num_edge_features": data_objects[2].num_edge_features,
            "conv_type": "CGConv",
        },
    ]
    net = SlabGCN(partition_configs)
    result_dict = net(data_objects)
    print(result_dict)
