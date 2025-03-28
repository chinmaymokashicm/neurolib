from collections.abc import MutableMapping
from typing import Optional
import json

import numpy as np

def flatten_no_compound_key(dictionary: dict):
    """
    Flatten a dictionary with no compound keys.
    """
    items = []
    for key, value in dictionary.items():
        if isinstance(value, MutableMapping):
            items.extend(flatten_no_compound_key(value).items())
        else:
            items.append((key, value))
    return dict(items)

def clean_dict_values(dictionary: dict, keyword: list[str] | str):
    """
    Clean dictionary values.
    """
    if isinstance(keyword, str):
        keyword = [keyword]
    for key, value in dictionary.items():
        if isinstance(value, dict):
            clean_dict_values(value, keyword)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    clean_dict_values(item, keyword)
                elif isinstance(item, str):
                    for k in keyword:
                        if k in item:
                            value[i] = item.replace(k, "")
        elif isinstance(value, str):
            for k in keyword:
                if k in value:
                    dictionary[key] = value.replace(k, "")
    return dictionary


def clean_dict_from_numpy(dictionary: dict):
    """
    Clean dictionary from numpy objects.
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            clean_dict_from_numpy(value)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    clean_dict_from_numpy(item)
                elif isinstance(item, np.ndarray):
                    value[i] = item.tolist()
        elif isinstance(value, np.ndarray):
            dictionary[key] = value.tolist()
    return dictionary

def clean_string_bids_entity(string: str) -> str:
    """
    BIDS entities should not contain special characters. This function removes special characters from a string.
    """
    return "".join(e for e in string if e.isalnum())


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, dict):
            return {k: self.default(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.default(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)