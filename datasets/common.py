import logging
import pathlib

import torch


def get_dataset_logger(log_dir: str | pathlib.Path | None = None) -> logging.Logger:
    logger = logging.getLogger("dataset loading")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    if log_dir is not None and not any(getattr(handler, "_dataset_handler", False) for handler in logger.handlers):
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(pathlib.Path(log_dir).joinpath("dataset.log"), mode="a")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        handler._dataset_handler = True
        logger.addHandler(handler)
    return logger


def pad_or_sample(list_of_lists, max_len, *, dtype=torch.long, seed=None):
    padded = torch.zeros((len(list_of_lists), max_len), dtype=dtype)
    lengths = torch.empty(len(list_of_lists), dtype=torch.long)
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    for i, values in enumerate(list_of_lists):
        length = len(values)
        if length == 0:
            lengths[i] = 0
            continue
        tensor = torch.as_tensor(values, dtype=dtype)
        if length > max_len:
            indices = torch.randperm(length, generator=generator)[:max_len]
            padded[i] = tensor[indices]
            lengths[i] = max_len
        else:
            padded[i, :length] = tensor
            lengths[i] = length
    return padded, lengths


def convert_feature_tensors_to_float32(data, tensor_keys):
    converted = {}
    for key, value in data.items():
        if key in tensor_keys:
            converted[key] = value.type(torch.FloatTensor)
        else:
            converted[key] = value
    return converted


def sample_identifier(item):
    if isinstance(item, dict):
        return item.get("graph") or item.get("face") or item.get("topo") or str(item)
    return str(item)
