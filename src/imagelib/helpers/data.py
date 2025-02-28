from collections.abc import MutableMapping
from typing import Optional

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