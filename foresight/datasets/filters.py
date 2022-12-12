from foresight.datasets.utils import get_all_splits
from collections import defaultdict


def filter_by_count(dataset, min_count=5, min_count_global=100, min_length=5, max_length=0, token_cnt=None, num_proc=None):
    r''' Filters tokens of a dataset and leaves only the ones with frequencey >= min_count

    Args:
        dataset
        min_count:
            Intra patient count
        min_count_global:
            Whole dataset count
        min_length:
            Examples below will be removed, in other words patients with less than min_length concepts
        max_length:
            Anything longer than this will be trimmed to this
        num_proc
    '''

    if min_count_global is not None and min_count_global > 0:
        if token_cnt is None:
            token_cnt = defaultdict(int)
            for _dataset in get_all_splits(dataset):
                for stream in _dataset['stream']:
                    seen_in_stream = set()
                    for sample in stream:
                        token = sample['token']
                        if token not in seen_in_stream:
                            token_cnt[token] += 1
                            seen_in_stream.add(token)

        # First we filter by global count, ie a concept has to have more than min_count_global appearances in the whole dataset
        dataset = dataset.map(function=lambda example: {'stream': [sample for sample in example['stream'] if token_cnt[sample['token']] >= min_count_global]},
                          load_from_cache_file=False, num_proc=num_proc)

    if min_count is not None:
        # Next, filter by intra-patient count
        dataset = dataset.map(function=lambda example: {'stream': [sample for sample in example['stream'] if sample['cnt'] >= min_count]},
                          load_from_cache_file=False, num_proc=num_proc)

    # Remove short streams
    if min_length > 0:
        dataset = dataset.filter(function=lambda example: len(example['stream']) >=  min_length, num_proc=num_proc, load_from_cache_file=False)

    if max_length > 0:
        dataset = dataset.map(function=lambda example: {'stream': example['stream'][:max_length]},
                          load_from_cache_file=False, num_proc=num_proc)

    return dataset


def filter_by_type(dataset, types_to_keep, num_proc):
    dataset = dataset.map(function=lambda example: {'stream': [stream for stream in example['stream'] if stream['token_type'] in types_to_keep]},
                          load_from_cache_file=False, num_proc=num_proc)

    return dataset
