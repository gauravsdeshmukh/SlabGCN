#!/usr/bin/env python

import argparse
import os
import json

import numpy as np
import pandas as pd
import torch

from ase.io import read
from sklearn.metrics import r2_score

from src.constants import REPO_PATH
from src.data import load_dataset, load_datapoints
from src.samplers import RandomSampler, RandomPropSampler
from src.train import Model
from src.utils import create_dataloaders, get_composition_string


def train(config, save_path, test, seed, epochs):
    """Train a SlabGCN model.
    
    Parameters
    ----------
    config: str
        Path to training config
    save_path: str
        Path to save directory
    test: bool
        Whether to save results on test split
    seed: int
        Seed for random number generators
    epochs: int
        Number of epochs
    
    """
    # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Create dataset
    dataset_path = config["dataset_path"]
    csv_path = config["csv_path"]
    proc_dict = {
        "layer_cutoffs": config["layer_cutoffs"],
        "node_features": config["node_features"],
        "edge_features": config["edge_features"],
    }

    dataset = load_dataset(
        root=dataset_path,
        prop_csv=csv_path,
        process_dict=proc_dict,
        load_in_memory=True
    )

    # Create sampler
    sample_config = {
        "train": config["ratios"]["train"],
        "val": config["ratios"]["val"],
        "test": config["ratios"]["test"]
    }
    dataset_size = len(dataset)
    sampler = RandomSampler(seed, dataset_size)
    sample_idx = sampler.create_samplers(sample_config)

    # Create dataloaders
    dataloader_dict = create_dataloaders(
        dataset,
        sample_idx,
        batch_size=config["batch_size"]
    )

    # Model definition
    # Create model
    global_config = {
        "gpu": config["use_GPU"],
        "loss_function": "mse",
        "metric_function": "mae",
        "learning_rate": 0.01,
        "optimizer": "adam",
        "lr_milestones": [50],
        "n_hidden": config["n_hidden"],
        "hidden_size": config["hidden_size"],
        "dropout": config["dropout"],
        "n_outputs": config["n_outputs"]
    }
    partition_configs = []
    for i in range(len(config["n_conv"])):
        part_dict = {
            "n_conv": config["n_conv"][i],
            "conv_size": config["conv_size"][i],
            "num_node_features": dataset[0][i].num_node_features,
            "num_edge_features": dataset[0][i].num_edge_features,
            "conv_type": config["conv_type"][i],
        }
        partition_configs.append(part_dict)

    # Training
    model_path = save_path
    model = Model(global_config, partition_configs, model_path)
    print(model)
    model.init_standardizer([dataset[i][0].y for i in sample_idx["train"]])
    results_dict = model.train(epochs, dataloader_dict, verbose=True)

    # Dump process dict in model results path
    with open(f"{model.model_results_path}/proc_dict.json", "w") as f:
        json.dump(proc_dict, f)

    # Test
    if test:
        model.load(best_status=True)
        model.parity_plot(dataset, sample_idx["test"])
        model.loss_plot(results_dict)

        pred_dict = model.predict(dataset, sample_idx["test"], return_targets=True)
        df_pred = pd.DataFrame(pred_dict)
        for i in range(global_config["n_outputs"]):
            df_pred[f"predictions_{i}"] = 0
            df_pred[f"targets_{i}"] = 0
            for j in range(df_pred.shape[0]):
                df_pred.loc[j, f"predictions_{i}"] = df_pred.loc[j, "predictions"][i]
                df_pred.loc[j, f"targets_{i}"] = df_pred.loc[j, "targets"][i]
        dataset_atoms = [dataset.get_atoms(i) for i in range(dataset.len())]
        dataset_names = np.array(
            [get_composition_string(dataset_atoms[i]) for i in range(len(dataset_atoms))]
        )
        df_pred["name"] = dataset_names[sample_idx["test"]]
        df_pred.to_csv(model.model_results_path / "test.csv")

def predict(model_path, structures, results_dir_path):
    """Make predictions on given structures using a SlabGCN model.
    
    Parameters
    ----------
    model_path: str
        Path where the trained model is stored.
    structures: list of str
        List of paths to structure files (cifs, POSCARS, anything that can
        be read by ASE)
    results_dir_path: str or path
        Path to the directory where results are to be stored.
    
    """
    # Read proc dict
    proc_dict_path = os.path.join(model_path, "results", "proc_dict.json")
    with open(proc_dict_path, "r") as f:
        proc_dict = json.load(f)
    
    # Read structures and create dataset
    list_of_atoms = []
    for s in structures:
        atoms = read(s)
        list_of_atoms.append(atoms)
    
    dataset = load_datapoints(
        list_of_atoms,
        proc_dict
    )
    dataset_indices = np.arange(0, len(dataset), 1)

    # Load model
    best_model_path = os.path.join(model_path, "models", "best.pt")
    model = Model(model_path=best_model_path, load_pretrained=True)
    print(model)

    # Make predictions
    pred_dict = model.predict(dataset, dataset_indices, return_targets=False)
    pred_dict.pop("targets", [])
    df_pred = pd.DataFrame(pred_dict)
    for i in range(model.global_config["n_outputs"]):
        df_pred[f"predictions_{i}"] = 0
        for j in range(df_pred.shape[0]):
            df_pred.loc[j, f"predictions_{i}"] = df_pred.loc[j, "predictions"][i]
    df_pred.rename(columns={"indices": "index"}, inplace=True)

    # Names of structures
    dataset_names = np.array(
        [get_composition_string(list_of_atoms[i]) for i in range(len(list_of_atoms))]
    )
    df_pred["name"] = dataset_names
    
    # Save
    if not os.path.exists(results_dir_path):
        os.makedirs(results_dir_path)
    df_pred.to_csv(os.path.join(results_dir_path, "pred.csv"))



if __name__ == "__main__":
    #### Create Argument Parser ####
    parser = argparse.ArgumentParser(
        prog="SlabGCN",
        description="SlabGCN is a deep learning model that predicts properties from atomic structure",
    )

    # Option to train model
    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Specify if model is to be trained. Requires specification of a training config."
    )

    # Option to make predictions
    parser.add_argument(
        "-p",
        "--predict",
        action="store_true",
        help="Specify if model is to be used for prediction. Requires specification of model path."
    )

    # Option to specify training configuration
    parser.add_argument("-c", "--config", action="store")

    # Option to specify model path for prediction
    parser.add_argument("-m", "--model", action="store")

    # Option to specify structure(s)
    parser.add_argument("-st", "--structs", action="extend", nargs="+")

    # Optional arguments
    # Option to specify where to save trained model (current directory by default)
    parser.add_argument("-s", "--save", action="store", default=".")

    # Option to specify whether to save test results (False by default)
    parser.add_argument("-nt", "--no-test", action="store_true", default=False)

    # Option to specify where to save the results (current directory by default)
    parser.add_argument("-r", "--results", action="store", default=".")

    # Option to specify seed (0 by default; only required for training)
    parser.add_argument("-sd", "--seed", action="store", default=0)

    # Option to specify number of epochs (500 by default)
    parser.add_argument("-e", "--epochs", action="store", default=500)

    # Parse arguments
    args = parser.parse_args()

    ### TRAINING
    if args.train:
        # Config path
        config_path = str(args.config)
        # Save path
        if args.save == ".":
            save_path = os.path.join(os.getcwd(), "slabgcn_model")
        else:
            save_path = str(args.save)
        # Check test
        test = not args.no_test
        # Read config
        with open(config_path, "r") as f:
            config = json.load(f)

        # Perform training
        train(config, save_path, test, int(args.seed), int(args.epochs))

    elif args.predict:
        # Results path
        if args.results == ".":
            results_dir_path = os.path.join(os.getcwd(), "slabgcn_results")
        else:
            results_dir_path = str(args.results)
        # Path to structures
        struct_paths = [os.path.join(os.getcwd(), str(s)) for s in args.structs]
        predict(args.model, struct_paths, results_dir_path)
        