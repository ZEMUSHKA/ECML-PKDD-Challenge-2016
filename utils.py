import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import cPickle


def save_pickle(obj, fn):
    with open(fn, "wb") as f:
        cPickle.dump(obj, f)


def read_pickle(fn):
    with open(fn, "rb") as f:
        return cPickle.load(f)


def apply_for_partition(df, group_key, apply_func):
    return df.groupby(group_key).apply(apply_func)


def parallel_groupby_and_apply(df, group_key, apply_func, n_jobs=16, n_partitions=16*32):
    partition_keys = map(lambda x: hash(str(x)) % n_partitions, df[group_key].as_matrix())
    partitions = df.groupby(partition_keys)
    res = Parallel(n_jobs=n_jobs)(
        delayed(apply_for_partition)(df, group_key, apply_func)
        for _, df in partitions
    )
    return pd.concat(res)


def apply_to_rows_for_partition(df, apply_func, partial_params):
    res = []
    for (_, row) in df.iterrows():
        if partial_params:
            res.append(apply_func(*(partial_params + [row])))
        else:
            res.append(apply_func(row))
    return np.vstack(res)


def parallel_apply_to_rows(df, apply_func, n_jobs=16, n_partitions=16, partial_params=()):
    partition_keys = np.random.randint(0, n_partitions, (len(df),))
    partitions = df.groupby(partition_keys)
    res = Parallel(n_jobs=n_jobs)(
        delayed(apply_to_rows_for_partition)(df, apply_func, partial_params)
        for _, df in partitions
    )
    return np.concatenate(res)
