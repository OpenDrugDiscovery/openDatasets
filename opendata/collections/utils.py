import sys
import types
import functools
import pandas as pd
from collections.abc import MutableMapping
from typing import Sequence, Optional, Any


def flatten(seq: Sequence[Optional[Any]]):
    return [item for subseq in seq for item in subseq]


def flatten_dict(d: MutableMapping, sep: str= '.') -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict

def is_callable(func):
    """Check if the object is callable."""
    FUNCTYPES = (types.FunctionType, types.MethodType, functools.partial)
    return func and (isinstance(func, FUNCTYPES) or callable(func))

