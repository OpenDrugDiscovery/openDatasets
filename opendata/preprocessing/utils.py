import os
import fsspec
import numpy as np
import pickle as pkl
import datamol as dm
from collections import Counter
from numpy.lib.stride_tricks import as_strided as strided 



def rolling_window(arr, window_size, pad_with=np.nan):
    a_ext = np.concatenate(( np.full(window_size-1, pad_with) ,arr))
    n = a_ext.strides[0]
    return strided(a_ext[window_size-1:], shape=(arr.size,window_size), strides=(n,-n))[:, ::-1]


def merge_counters(counters, step=5, verbose=0):
    """Merge a list of counters into a single counter in parallel using a step size."""
    def merge_n_counters(x):
        return sum(x, Counter())

    
    if len(counters) == 0:
        return Counter()
    
    if len(counters) < step:
        return merge_n_counters(counters)
    
    n = len(counters)
    res = dm.parallelized(merge_n_counters, 
                          [counters[i:i+step] for i in range(0, n, step)], 
                          n_jobs=-1, progress=verbose, batch_size=1, )
    
    res = merge_counters(res)

    return res


def get_local_cache():
    fname = os.path.abspath(__file__)
    base ='/'.join(fname.split('/')[:-3])
    cache_dir=os.path.join(base, 'cache')
    return cache_dir


def get_misato_folder():
    fname = os.path.abspath(__file__)
    base ='/'.join(fname.split('/')[:-4])
    d = os.path.join(base, 'misato')
    if not os.path.exists(d):
        d = "gs://opendatasets/misato"
    return d


def get_remote_cache(cname, return_filesystem=False):
    if cname == "small_molecules":
        remote_cache = "gs://opendatasets/small_molecule_collections"
        rfs = fsspec.filesystem("gs")
    elif cname == "peptides":
        remote_cache = "gs://opendatasets/peptides"
        rfs = fsspec.filesystem("gs")
    else:
        raise NotImplementedError()
    
    if return_filesystem:
        return remote_cache, rfs
    return remote_cache 


def caching_wrapper(f, *args, local_path=None, remote_path=None, 
                    remote_filesystem=None, **kwargs):
    local_specified = local_path is not None
    remote_specified = remote_path is not None
    if local_specified and os.path.exists(local_path):
        with fsspec.open(local_path, "rb") as fd:
            res = pkl.load(fd)
    elif remote_specified and remote_filesystem.exists(remote_path):
        with fsspec.open(remote_path, "rb") as fd:
            res = pkl.load(fd)
    else:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        res = f(*args, **kwargs)
        if local_path is not None:
            with fsspec.open(local_path, "wb") as fd:
                pkl.dump(res, fd)

        if remote_path is not None:
            with fsspec.open(remote_path, "wb") as fd:
                pkl.dump(res, fd)
    return res
