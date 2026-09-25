"""Shared helpers for loading config and data."""

import os
import yaml
import pandas as pd


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Could not find data at '{data_path}'. "
            "Run `python data/generate_data.py` to create it, "
            "or point config.yaml's data_path at your own CSV."
        )
    return pd.read_csv(data_path)
