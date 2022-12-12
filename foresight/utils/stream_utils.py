from collections import defaultdict
import logging

def get_entities_for_doc(docs, doc_id):
    r''' Return entities for the given doc_id from the docs dictionary.

    docs:
        Output of medcat multiprocessing
    doc_id:
        id of the doc in docs
    '''
    ents = docs[doc_id]['entities']
    # Depending on the version of medcat ents can be dict {id: entities, ...} or list of entities
    ents = ents.values() if isinstance(ents, dict) else ents

    return ents

def calculate_counts(docs, doc2pt, pt2cui2cnt, meta_requirements=None):

    if pt2cui2cnt is None:
        pt2cui2cnt = defaultdict(lambda: defaultdict(int))

    # Frequency for each each entity given a patient
    for doc in docs:
        for ent in get_entities_for_doc(docs, doc):
            # Must match all meta meta_anns
            if not meta_requirements or \
               all([ent['meta_anns'][name]['value'] == value for name, value in meta_requirements.items()]):
                cui = ent['cui']
                pt = doc2pt[doc]
                pt2cui2cnt[pt][cui] += 1

    return pt2cui2cnt


def docs2stream(docs, doc2pt, pt2cui2cnt, doc2time=None, meta_requirements={}, entity_type_column='tuis',
                historical_meta=None, historical_meta_value=None, old_pt2stream=None, skip_cuis=None,
                require_time=True):
    r''' Convert the `docs` output of medcat multiprocessing
    to a stream of concepts for each patient.

    Args:
        docs
        doc2pt
        doc2time
        meta_requirements:
            Values for meta_annotaitons that must exist e.g. = {'Presence': True}
    '''
    if old_pt2stream is not None:
        pt2stream = old_pt2stream
    else:
        pt2stream = defaultdict(list)

    have_warned = set()
    for doc in docs:
        for ent in get_entities_for_doc(docs, doc):
            if not meta_requirements or \
               all([ent['meta_anns'][name]['value'] == value for name, value in meta_requirements.items()]):

                cui = ent['cui']
                if skip_cuis is None or cui not in skip_cuis:
                    if doc2time is not None:
                        timestamp = doc2time[doc]
                    elif 'document_timestamp' in ent:
                        timestamp = ent['document_timestamp']
                    else:
                        timestamp = None # Means time is not known, later it will be ignored if necessary

                    if not require_time or timestamp is not None: # Skip all where timestamp is None
                        if historical_meta is not None and timestamp is not None:
                            # If something is historical then make the timestamp less by 1 because it appeared before 
                            #other things in this document. Unles time is None which means time is undefined
                            if ent['meta_anns'][historical_meta]['value'] == historical_meta_value:
                                timestamp = timestamp - 1

                        pt = doc2pt[doc]
                        cnt = pt2cui2cnt[pt][cui]
                        if ent[entity_type_column]: # This can be none in some cases
                            token_type = ent[entity_type_column][0]
                        else:
                            token_type = 'unk'
                            if cui not in have_warned:
                                logging.warning(f"Entity type missing from: {cui}")
                                have_warned.add(cui)
                        pt2stream[pt].append((cui, cnt, timestamp, token_type))

    return pt2stream
