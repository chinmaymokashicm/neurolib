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