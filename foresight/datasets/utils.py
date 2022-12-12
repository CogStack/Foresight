import logging
import numpy as np
from medcat.utils.matutils import unitvec
from datetime import datetime
import math
import datasets
import random
import copy

def get_all_splits(dataset):
    all_datasets = []
    if 'train' in dataset:
        all_datasets.append(dataset['train'])
    if 'test' in dataset:
        all_datasets.append(dataset['test'])
    if 'valid' in dataset:
        all_datasets.append(dataset['valid'])
    if isinstance(dataset, datasets.arrow_dataset.Dataset):
        # If we have only one, ie no train/test
        all_datasets.append(dataset)

    return all_datasets

def make_example(token, ent_example, token_type='unk', cnt=10**6, time=None, cntx=None):
    out = {'token': token, 'token_type': token_type, 'cnt': cnt, 'time': time}
    if 'context_representation' in ent_example:
        if cntx is None:
            cntx = [0.0 for i in range(len(ent_example['context_representation']))]

        out['context_representation'] = cntx
    return out

def get_duration_separator(separator, start_time, current_time, bucket_size_seconds):
    d_separator = separator
    for i in [1, 7]:
        if (current_time - start_time) >= bucket_size_seconds * i:
            d_separator = f'{separator[0:-1]}-{i}{separator[-1]}'

    return d_separator


def bucket_concepts(examples, bucket_size_seconds=365*24*60*60, separator='<SEP>', duration_separator=False):
    r''' Will bucket concepts into specified bucket_size.

    Args:
        examples
    '''
    for i in range(len(examples['stream'])):
        stream = examples['stream'][i]

        new_stream = []
        _bucket = []
        _tokens = set()
        start_time = -1
        for ent in stream:
            if start_time == -1:
                start_time = ent['time']

            if ent['time'] - start_time >= bucket_size_seconds:
                # Add to stream
                new_stream.extend(_bucket)
                _bucket = []
                _tokens = set()

                if separator is not None:
                    _separator = separator
                    if duration_separator:
                        # This will have different separator for different time spans
                        _separator = get_duration_separator(separator, start_time, ent['time'], bucket_size_seconds)

                    # A separator is +1 of the last token in the stream
                    new_stream.append(make_example(ent_example=ent, token=_separator, token_type='sep', cnt=10**6, time=new_stream[-1]['time']+1))
                # Change start time to current entity time
                start_time = ent['time']

            if ent['token'] not in _tokens:
                _bucket.append(ent)
                _tokens.add(ent['token'])

        if _bucket:
            new_stream.extend(_bucket)

        examples['stream'][i] = new_stream
        new_stream = []

    return examples

def add_position_ids(examples, separators=set()):
    for i in range(len(examples['stream'])):
        stream = examples['stream'][i]

        old_t = None
        cnt = 0
        for ent in stream:
            ent['position_ids'] = cnt
            if ent['token'] in separators:
                cnt += 1

    return examples

def add_age(examples, pt2dob_timestamp, age_prefix='<AGE>', age_suffix=None, age_normalizer=365.25 * 24 * 60 * 60):
    for i in range(len(examples['stream'])):
        stream = examples['stream'][i]
        last_age_added = -1
        new_stream = []
        for ent in stream:
            if examples['patient_id'][i] in pt2dob_timestamp:
                if pt2dob_timestamp is not None:
                    age = int((ent['time'] - pt2dob_timestamp[examples['patient_id'][i]]) / age_normalizer)

                # Age comes a step before the token that caused the change
                if age >= 0 and last_age_added != age:
                    if age_prefix is not None:
                        new_stream.append(make_example(ent_example=ent, token=age_prefix, token_type='age_prefix', cnt=10**6, time=ent['time']))
                    new_stream.append(make_example(ent_example=ent, token=str(age), token_type='age', cnt=10**6, time=ent['time']))
                    last_age_added = age
                    if age_suffix is not None:
                        new_stream.append(make_example(ent_example=ent, token=age_suffix, token_type='age_suffx', cnt=10**6, time=ent['time']))

            new_stream.append(ent)

        examples['stream'][i] = new_stream
        new_stream = []

    return examples

def add_ttd(examples, pt2dod_timestamp, ttd_prefix='<TTD>', ttd_suffix=None, ttd_normalizer=365.25 * 24 * 60 * 60,
            max_ttd=10, ttd_prob=1, max_nttd=10, duplicate_streams=False):
    all_patient_id = []
    all_stream = []
    for i in range(len(examples['stream'])):
        stream = examples['stream'][i]
        last_ttd_added = -1
        new_stream = []
        new_streams = [new_stream]
        n_added_ttds = 0
        for ent in stream:
            if examples['patient_id'][i] in pt2dod_timestamp:
                if n_added_ttds < max_nttd:
                    if random.random() <=  ttd_prob:
                        ttd = int((pt2dod_timestamp[examples['patient_id'][i]] - ent['time']) / ttd_normalizer) + 1
                        if ttd <= max_ttd:
                            if last_ttd_added != ttd:
                                if duplicate_streams:
                                    # At this point we duplicate the first stream fron new_streams (it is the one without TTD always)
                                    new_stream = copy.deepcopy(new_streams[0])
                                    new_streams.append(new_stream)

                                if ttd_prefix is not None:
                                    new_stream.append(make_example(ent_example=ent, token=ttd_prefix, token_type='ttd_prefix', cnt=10**6, time=ent['time']))
                                new_stream.append(make_example(ent_example=ent, token=str(ttd), token_type='ttd', cnt=10**6, time=ent['time']))

                                last_ttd_added = ttd
                                if ttd_suffix is not None:
                                    new_stream.append(make_example(ent_example=ent, token=ttd_suffix, token_type='ttd_suffix', cnt=10**6, time=ent['time']))
                                n_added_ttds += 1

            # append the entity to each stream
            for new_stream in new_streams: new_stream.append(ent)

        if duplicate_streams and len(new_streams) > 1:
            # Remove the first example as it is the base one without time info
            del new_streams[0]

        for new_stream in new_streams:
            all_stream.append(new_stream)
            all_patient_id.append(examples['patient_id'][i])

    examples['patient_id'] = all_patient_id
    examples['stream'] = all_stream

    return examples

def split_stream(examples, max_seq_len=-1):
    if max_seq_len > 0:
        new_streams = []
        new_patient_ids = []
        for ind, stream in enumerate(examples['stream']):
            nparts = math.ceil(len(stream) / max_seq_len)
            for i in range(nparts):
                new_streams.append(stream[i*max_seq_len:(i+1)*max_seq_len])
                new_patient_ids.append(examples['patient_id'][ind])

        examples['stream'] = new_streams
        examples['patient_id'] = new_patient_ids

    return examples


def cleanup_stream(examples, keep_time=True, keep_type=True, keep_position_ids=True, keep_context_representation=True):
    r''' Leave only Tokens and remove the rest from `stream`

    Args:
        examples
        keep_time:
            If set another value will be added to examples that contains the `time` for each
            entity in stream.
        keep_type:
            Same as above
    '''
    if 'token' in examples['stream'][0][0]:
        if keep_time:
            examples['time'] = [[ent['time'] for ent in stream] for stream in examples['stream']]
        if keep_type:
            examples['token_type'] = [[ent['token_type'] for ent in stream] for stream in examples['stream']]
        if keep_position_ids:
            examples['position_ids'] = [[ent['position_ids'] for ent in stream] for stream in examples['stream']]
        if keep_context_representation:
            examples['context_representation'] = [[ent['context_representation'] for ent in stream] for stream in examples['stream']]

        examples['stream'] = [[ent['token'] for ent in stream] for stream in examples['stream']]

    return examples


def add_to_stream(examples, pt2tkn, last=False, prefix=None, unk_tkn='unk', token_type='unk'):
    r''' Add information to the patient stream based on patient_id.

    Args:
        examples
        pt2tkn
        last
        unk_tkn:
            What token will be added if the patient_id is not in pt2tkn
    '''

    for i in range(len(examples['stream'])):
        ent = examples['stream'][i][0]

        if examples['patient_id'][i] in pt2tkn:
            token = pt2tkn.get(examples['patient_id'][i], unk_tkn)
            t_ind = -1 if last else 0 # If -1 means it is the last token, otherwise the first
            to_append = [make_example(ent_example=ent, token=token, cnt=10**6, time=examples['stream'][i][t_ind]['time'], token_type=token_type)]
            if prefix is not None:
                prefix_token = make_example(ent_example=ent, token=prefix, cnt=10**6,
                                            time=examples['stream'][i][t_ind]['time'], token_type="prefix_" + token_type)
                to_append = [prefix_token] + to_append

            if last:
                # Append as last token
                examples['stream'][i] = examples['stream'][i] + to_append
            else:
                examples['stream'][i] = to_append + examples['stream'][i]

    return examples


def remove_tokens_not_in_tokenizer(examples, tokens_to_keep):
    tokens_to_keep = set(tokens_to_keep)
    for i in range(len(examples['stream'])):
        stream = examples['stream'][i]
        new_stream = []

        for ent in stream:
            tkn = ent['token']

            if tkn in tokens_to_keep:
                new_stream.append(ent)

        examples['stream'][i] = new_stream

    return examples


def remove_parents_from_stream(examples, ch2parents, separator=None, separators=None):
    for i in range(len(examples['stream'])):
        stream = examples['stream'][i]
        parents = set()
        new_stream = []

        for ent in stream:
            tkn = ent['token']

            if (separator is not None and tkn == separator) or (separators is not None and tkn in separators):
                # This means we are removing parents only inside of one bucket
                parents = set()

            if tkn in ch2parents:
                # Add only if not in parents
                if tkn not in parents:
                    new_stream.append(ent)
                # Update parents
                parents.update(ch2parents[tkn])
            else:
                new_stream.append(ent)

        examples['stream'][i] = new_stream

    return examples

def get_embeddings_for_tokens(dataset=None, cdb=None, context_type='medium', normalize=True, extra_tokens=['<PAD>'], types=None, concepts=None):
    r''' Given a stream of tokens get the embeddings from MedCAT and make the required maps.

    Args:
        dataset
        cdb
        context_type
        normalize:
            If True the embedding vectors will be normalized
        tkn2type:
            Dictionary mapping from token to type
        types:
            All posible token types (e.g. [T-11, T-12, ...]
        concepts:
            If provided these concepts will also be appened to the tokens and supported by the tokenizer
    Returns:
        embeddings
        tkn2id
        id2tkn
        id2type
        id2type_detailed
    '''
    embeddings = []
    tkn2id = {}
    id2tkn = {}

    def add_tkn(tkn):
        if tkn in cdb.cui2context_vectors and context_type in cdb.cui2context_vectors[tkn]:
            vec = cdb.cui2context_vectors[tkn][context_type]
        else:
            # Token vector is randomly assigned
            vec = np.random.rand(300)

        id2tkn[len(embeddings)] = tkn
        tkn2id[tkn] = len(embeddings)

        vec = unitvec(vec) if normalize else vec
        embeddings.append(vec)

    datasets = get_all_splits(dataset)
    for _dataset in datasets:
        for stream in _dataset['stream']:
            for tkn in stream:
                tkn = str(tkn)
                if tkn not in tkn2id:
                    add_tkn(tkn)
    # Add concepts if they are provided, this is used to build a general
    #tokenizer with all concepts
    if concepts is not None:
        for concept in concepts:
            tkn = str(concept)
            if tkn not in tkn2id:
                add_tkn(tkn)

    # Add named tokens
    for tkn in extra_tokens:
        if tkn not in tkn2id:
            id2tkn[len(embeddings)] = tkn
            tkn2id[tkn] = len(embeddings)
            if tkn != '<PAD>':
                embeddings.append(np.random.rand(len(embeddings[0])))
            else:
                embeddings.append(np.zeros(len(embeddings[0])))

    # Add type tokens
    for tkn in types:
        if tkn not in tkn2id:
            id2tkn[len(embeddings)] = tkn
            tkn2id[tkn] = len(embeddings)
            embeddings.append(np.random.rand(len(embeddings[0])))

    return embeddings, tkn2id, id2tkn


def stream_to_separate_examples(examples):
    r''' Convert a stream to separate examples that can be used to train
    a next concept predictor unable to handle sequences (e.g. random forset). Use with HF datasets map function.

    '''
    out = {}
    out['input_ids'] = [input_ids[0:i+1] for input_ids in examples['input_ids'] for i in range(len(input_ids) - 1)]
    out['labels'] = [input_ids[i+1] for input_ids in examples['input_ids'] for i in range(len(input_ids) - 1)]
    out['labels_all'] = [input_ids[i+1:] for input_ids in examples['input_ids'] for i in range(len(input_ids) - 1)]
    out['patient_id'] = [patient_id for ind, patient_id in enumerate(examples['patient_id']) for _ in range(len(examples['input_ids'][ind]) - 1)]

    return out
