import numpy as np
import pandas as pd
import logging
from foresight.tokenizers.simple_map_tokenizer import SimpleMapTokenizer


def metrics_data2df(metrics_data, tkn2name=None, main='positives', temporality='all'):
    d = metrics_data
    if main == 'positives':
        other = 'negatives'
    else:
        other = 'positives'

    out = sorted([(
        "{:.2f}".format(tp / (tp + d[other][temporality].get(cui, 0))),
        "{:.2f}".format(d['fn_positives'][temporality].get(cui, 0) / 
                        (d['fn_positives'][temporality].get(cui, 0) + d['fn_negatives'][temporality].get(cui, 0) )),
        (tkn2name.get(cui, cui) if tkn2name is not None else cui),
        cui,
        tp,
        d[other][temporality].get(cui, 0),
        d['fn_positives'][temporality].get(cui, 0),
        d['fn_negatives'][temporality].get(cui, 0)
        ) for cui, tp in sorted(d[main][temporality].items(), key=lambda x: x[1], reverse=True)],
        key=lambda x: x[0], reverse=True)

    out = pd.DataFrame(out, columns=['precision', 'recall', 'name', 'cui', main, other, 'fn_positives', 'fn_negatives'])
    out['precision'] = pd.to_numeric(out['precision'])
    out['recall'] = pd.to_numeric(out['recall'])

    return out


def precision(predictions, label_ids, id2tkn, token_type2tokens, type_data, select_token_types={'T-11'}, prediction_scope='one', shifted_labels=False,
        predictions_are_scores=True, old_data=None, topk=1, start=0, time_range=None, time_data=None, ignore_label_status=False,
        min_time_left=None, pure_concept_prediction=False, tokenizer: SimpleMapTokenizer=None):
    r''' Calculate precision for next concept prediction.

    Args:
        predictions:
            Expected shape <batch_size> x <sequence_length> x <vocabulary_size>
        label_ids:
            Expected shape <batch_size> x <sequence_length>
        token_type2tokens:
            Map from a token type to all tokens belonging to it
        type_data:
            token types for each label/example
        select_token_types (Set[str], optional, defaults to `{'cui'}`:
            On what token types to calculate the Precision. Leave empoty to include all token types.
        prediction_scope:
            How much into the future should we look to accept something as correct:
                - `one` has to be the next concept
                - `age` until the next age token
                - `any` whenever
        shifted_labels:
            Are labels == input_ids, or shifted by one to the left
        predictions_are_scores:
            Are predictions scores for each label_id or really label_ids already
        old_data:
            If set it will load old values for tp/fp/positives/negatives and continue ontop of those
        topk:
            How many predicted labels to consider when calculating precision
        start:
            At what point to start - we will look only at the precision of concepts at positions after start
        ignore_label_status:
            If True we do not care is the label at position <i> new/old we just predict the most likley concept and see does it
            match (apper in the next N days). If False candidates are only of the same status new/old as the label.

    Return (Dict[str, ]):
        precision:
            Precision
        tp:
            Number of True positives
        fp:
            Number of False positives
        positives:
            For each label ID a count of positive examples
        negatives
            For each label ID a count of negative examples
    '''
    log = logging.getLogger()
    if predictions_are_scores:
        if type(predictions) == list:
            outputs = [np.argsort(-1 * x, axis=1) for x in predictions]
        else:
            outputs = np.argsort(-1 * predictions, axis=2)
    else:
        outputs = predictions
    tp = {'all': 0, 'new': 0, 'old': 0}
    fp = {'all': 0, 'new': 0, 'old': 0}
    fn = {'all': 0, 'new': 0, 'old': 0}
    # The question could be how is it possible to have two different TP counts,
    #and in fact if we look at the results the values of this two will be different. The reason
    #is that something is considered a TP for precision if at timepoint T until T+time_range there is
    #a concept matching the predicted one. But for Recall something is positive if a concept X was
    #predicted at any point during our moving through the timeline, and negative if it was never predicted.
    tp_for_fn = {'all': 0, 'new': 0, 'old': 0}
    positives = {'all': {}, 'new': {}, 'old': {}}
    negatives = {'all': {}, 'new': {}, 'old': {}}
    fn_positives = {'all': {}, 'new': {}, 'old': {}}
    fn_negatives = {'all': {}, 'new': {}, 'old': {}}
    numerical_errors = []

    # Are the requested token types or numerical and can a numerical error be calculated
    calculate_numerical_error = all([tkn_type in ['age', 'ttd'] for tkn_type in select_token_types])

    # If not shifted_labels label = prediction - 1
    label_position_shift = 0 if shifted_labels else 1
    # If labels are not shifted move the start by one
    start += 0 if shifted_labels else 1

    if old_data:
        tp = old_data['tp']
        fp = old_data['fp']
        fn = old_data['fn']
        tp_for_fn = old_data['tp_for_fn']
        positives = old_data['positives']
        negatives = old_data['negatives']
        fn_positives = old_data['fn_positives']
        fn_negatives = old_data['fn_negatives']
        numerical_errors = old_data['numerical_errors']

    def prediction_end_index(i, lbl, ind):
        r''' Used below to get the end index for different
        prediction scopes
        '''
        if prediction_scope == 'one':
            return i + 1
        elif prediction_scope == 'any':
            return len(lbl)
        elif prediction_scope == 'age':
            end = len(lbl) # Set end to last token in the labels array (for one example)
            _token_types = type_data[ind]
            for j in range(i, len(lbl)):
                type_label = _token_types[j] if j < len(_token_types) else 'unk'
                if type_label == 'age':
                    end = j
                    break
            return end
        elif prediction_scope == 'sep':
            end = len(lbl) # Set end to last token in the labels array (for one example)
            _token_types = type_data[ind]
            for j in range(i, len(lbl)):
                type_label = _token_types[j] if j < len(_token_types) else 'unk'
                if type_label == 'sep':
                    end = j
                    break
            return end
        elif prediction_scope == 'time_range':
            end = len(lbl) # Set end to last token in the labels array (for one example)
            token_time = time_data[ind]
            for j in range(i, len(lbl)):
                if j < len(token_time): # It can be that time is not available for padding tokens
                    if token_time[j] > (token_time[i] + time_range):
                        end = j
                        break
            return end
    for ind, lbl in enumerate(label_ids):
        # This will be used to calcualte FPs from the labels
        fn_lbl = np.ones_like(lbl, dtype=np.int32) * -1
        _token_types = type_data[ind]

        if start < len(lbl):
            for i in range(start, len(lbl)):
                tkn_label = str(id2tkn.get(lbl[i], lbl[i]))
                type_label = _token_types[i] if i < len(_token_types) else 'unk'
                is_new_label = True if lbl[i] not in lbl[0:i] else False

                # Calculate the time difference between current and last token if needed
                enough_time_left = True
                if min_time_left is not None:
                    if i < len(time_data[ind]):
                        t_diff = time_data[ind][-1] - time_data[ind][i]
                        if t_diff < min_time_left:
                            enough_time_left = False
                    else:
                        # Means we do not have timedata for this tokens, most likely they are padding
                        enough_time_left = False
                if type_label in select_token_types and enough_time_left:
                    candidates = []
                    select_tokens = token_type2tokens[type_label]
                    if predictions_are_scores:
                        # We only get the type of canidate we know we need at this position,
                        #as well as the temporality new/old
                        for k in range(len(outputs[ind][i-label_position_shift])):
                            out_id = outputs[ind][i-label_position_shift][k]
                            is_new_out_id = True if out_id not in lbl[0:i] else False
                            if pure_concept_prediction or (id2tkn[out_id] in select_tokens and (ignore_label_status or is_new_out_id == is_new_label)):
                                candidates.append(out_id)

                            if len(candidates) == topk:
                                break
                    else:
                        candidates.append(outputs[ind][i-label_position_shift])
                    is_tp = False
                    is_new = False
                    end = prediction_end_index(i, lbl, ind)
                    tkn_candidate = str(id2tkn.get(candidates[0], candidates[0]))
                    if candidates:
                        # If we have candidates and if the lbl was never predicted until now, then
                        #we set it to 1
                        if fn_lbl[i] == -1:
                            fn_lbl[i] = 1 # Means this token is false negative, never predicted before
                    for candidate in candidates:
                        # Is it a new concept or an existing one, this only makes sense when there
                        #is just one candidate or if ignore_label_status is false, then it makes sense for multi candidates. But,
                        #scores per candidate then do not make sense
                        is_new = True if candidate not in lbl[0:i] else False

                        _candidate = str(id2tkn.get(candidate, candidate))
                        if _candidate in select_tokens:
                            # If predictions are scores we can do topk, if not just do simple label match
                            if candidate in lbl[i:end]:
                                # Update for TP
                                if not is_tp:
                                    is_tp = True
                                    tkn_candidate = _candidate
                                # Update the FN
                                for _i in np.where(lbl[i:end] == candidate)[0]:
                                    # Set the FN to 0 as we've now predicted the token and it is not false negative
                                    #anymore
                                    fn_lbl[i + _i] = 0

                    log.debug("Start/End: %d/%d", i, end)
                    if tokenizer:
                        log.debug(    "    Label[%s]: %s - %s", ('N' if is_new_label else 'O'),
                                  lbl[i], tokenizer.tkn2name[tokenizer.id2tkn[lbl[i]]])
                        for candidate in candidates:
                            log.debug("Candidate[%s]: %s - %s", ('N' if is_new else 'O'), 
                                      candidate, tokenizer.tkn2name[tokenizer.id2tkn[candidate]])
                    log.debug("TP: %s, FN: %s", is_tp, fn_lbl[i])
                    log.debug(" ")


                    temporality = 'new' if is_new else 'old'
                    def count_tkn_candidate(positives, negatives, tkn_candidate, temporality):
                        # Scores per tkn_canidate do not make sense when there are multi-candidates
                        positives[temporality][tkn_candidate] = positives[temporality].get(tkn_candidate, 0) + 1
                        if tkn_candidate not in negatives[temporality]:
                            negatives[temporality][tkn_candidate] = 0

                        # Add for ALL
                        positives['all'][tkn_candidate] = positives['all'].get(tkn_candidate, 0) + 1
                        if tkn_candidate not in negatives['all']:
                            negatives['all'][tkn_candidate] = 0

                    # This is for Recall
                    if fn_lbl[i] == 1:
                        fn['all'] += 1
                        fn[temporality] += 1
                        count_tkn_candidate(positives=fn_negatives, negatives=fn_positives,
                                            tkn_candidate=tkn_candidate, temporality=temporality)
                    elif fn_lbl[i] == 0:
                        tp_for_fn['all'] += 1
                        tp_for_fn[temporality] += 1
                        count_tkn_candidate(positives=fn_positives, negatives=fn_negatives,
                                            tkn_candidate=tkn_candidate, temporality=temporality)
                  
                    # This if for Precision
                    if is_tp:
                        tp['all'] += 1
                        tp[temporality] += 1
                        count_tkn_candidate(positives=positives, negatives=negatives,
                                            tkn_candidate=tkn_candidate, temporality=temporality)
                    else:
                        fp['all'] += 1

                        fp[temporality] += 1
                        count_tkn_candidate(positives=negatives, negatives=positives,
                                            tkn_candidate=tkn_candidate, temporality=temporality)

                    if calculate_numerical_error:
                        # Both have to be of the right type, that is how candidates are setup
                        num_label = int(tkn_label)
                        num_pred = int(tkn_candidate)
                        numerical_error = abs(num_label - num_pred)
                        numerical_errors.append([num_label, num_pred, numerical_error])

    precision = {}
    recall = {}
    for temporality in tp.keys():
        if tp[temporality] > 0:
            precision[temporality] = tp[temporality] / (tp[temporality] + fp[temporality])
        else:
            precision[temporality] = 0

        if tp_for_fn[temporality] > 0:
            recall[temporality] = tp_for_fn[temporality] / (fn[temporality] + tp_for_fn[temporality])
        else:
            recall[temporality] = 0


    metrics_data = {
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tp_for_fn': tp_for_fn,
            'positives': positives,
            'negatives': negatives,
            'fn_positives': fn_positives,
            'fn_negatives': fn_negatives,
            'numerical_errors': numerical_errors,
            'macro_precision': {}
            }

    # Calculate macro precision
    for temporality in tp.keys():
        df = metrics_data2df(metrics_data, tkn2name=None, temporality=temporality)
        metrics_data['macro_precision'][temporality] = np.average(df.precision.values)

    return metrics_data


def precision_on_one(predictions, label_ids, concept_id,
        old_data=None, topk=1, start=0, time_range=None, time_data=None,
        min_time_left=None, shifted_labels=False):
    r''' Calculate precision for only one concept
    '''
    if type(predictions) == list:
        outputs = [np.argsort(-1 * x, axis=1) for x in predictions]
    else:
        outputs = np.argsort(-1 * predictions, axis=2)

    tp = 0
    fp = 0
    fn = 0

    label_position_shift = 0 if shifted_labels else 1

    if old_data:
        tp = old_data['tp']
        fp = old_data['fp']
        fn = old_data['fn']

    def prediction_end_index(i, lbl, ind):
        r''' Used below to get the end index for different
        prediction scopes
        '''
        end = len(lbl) # Set end to last token in the labels array (for one example)
        token_time = time_data[ind]
        for j in range(i, len(lbl)):
            if j < len(token_time): # It can be that time is not available for padding tokens
                if token_time[j] > (token_time[i] + time_range):
                    end = j
                    break
        return end

    for ind, lbl in enumerate(label_ids):
        if start < len(lbl):
            # Patient level TP
            is_tp = False
            for i in range(start, len(lbl)):
                if concept_id not in lbl:
                    # Concept is in the labels

                    # Calculate the timedifference between current and last token if needed
                    enough_time_left = True
                    if min_time_left is not None:
                        if i < len(time_data[ind]):
                            t_diff = time_data[ind][-1] - time_data[ind][i]
                            if t_diff < min_time_left:
                                enough_time_left = False
                        else:
                            # Means we do not have timedata for this tokens, most likely they are padding
                            enough_time_left = False

                    if enough_time_left:
                        # Get top 10
                        candidates = []
                        for k in range(len(outputs[ind][i-label_position_shift])):
                            out_id = outputs[ind][i-label_position_shift][k]
                            candidates.append(out_id)

                            if len(candidates)  == topk:
                                break

                        if concept_id in candidates:
                            # Means the concept is there even though it should not be
                            fp += 1
                else:
                    c_ind = np.where(lbl == concept_id)[0][0]
                    if i <= c_ind:
                        # Concept is in the labels, ie this patient has the concept of interest
                        candidates = []
                        for k in range(len(outputs[ind][i-label_position_shift])):
                            out_id = outputs[ind][i-label_position_shift][k]
                            candidates.append(out_id)

                            if len(candidates)  == topk:
                                break

                        end = prediction_end_index(i, lbl, ind)
                        if concept_id in lbl[i:end]:
                            if concept_id in candidates:
                                is_tp = True
                        else:
                            if concept_id in candidates:
                                fp += 1

            # Finally if the concept ID is in candidates
            if concept_id in lbl:
                if is_tp:
                    # concept_id was found
                    tp += 1
                else:
                    fn += 1

    metrics_data = {
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            }

    return metrics_data


class ComputePrecisionOneHF(object):
    r''' Used for computing precison when working with HF trainer
    '''

    def __init__(self, id2tkn, type_data, token_type2tokens, batch_size=1000, topk=1, return_all_metrics=False, time_range=None, time_data=None,
                 ignore_label_status=False, tokenizer=None, **kwargs):
        self.id2tkn = id2tkn
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.topk = topk
        self.return_all_metrics = return_all_metrics
        self.type_data = type_data
        self.token_type2tokens = token_type2tokens
        self.time_range = time_range
        self.time_data = time_data
        self.ignore_label_status = ignore_label_status
        self.tokenizer = tokenizer

    def __call__(self, p, metrics_data=None):
        # We will do this in batches, because it can be very memory demanding
        metrics_data = metrics_data
        start = 0
        while start < len(p.predictions):
            predictions = p.predictions[start:start+self.batch_size]
            label_ids = p.label_ids[start:start+self.batch_size]
            if self.time_data is not None:
                time_data_batch = self.time_data[start:start+self.batch_size]
            else:
                time_data_batch = None
            type_data_batch = self.type_data[start:start+self.batch_size]

            metrics_data = precision(predictions, label_ids=label_ids, token_type2tokens=self.token_type2tokens,
                                     id2tkn=self.id2tkn, type_data=type_data_batch, old_data=metrics_data,
                                     predictions_are_scores=True, topk=self.topk, time_range=self.time_range,
                                     time_data=time_data_batch, ignore_label_status=self.ignore_label_status, 
                                     tokenizer=self.tokenizer, **self.kwargs)
            start += self.batch_size

        if self.return_all_metrics:
            return {
                'metrics_data': metrics_data, # Return all the metrics data too
            }
        else:
            return {
                'precision': metrics_data['precision']['all'],
                'precision_new': metrics_data['precision']['new'],
                'precision_old': metrics_data['precision']['old'],
                'macro_precision': metrics_data['macro_precision']['all'],
                'macro_precision_new': metrics_data['macro_precision']['new'],
                'macro_precision_old': metrics_data['macro_precision']['old'],
            }

class ComputePrecisionHF(object):
    r''' Used for computing precison when working with HF trainer
    '''

    def __init__(self, id2tkn, type_data, token_type2tokens, batch_size=1000, topk=1, return_all_metrics=False, time_range=None, time_data=None,
                 ignore_label_status=False, concept_id=None, **kwargs):
        self.id2tkn = id2tkn
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.topk = topk
        self.return_all_metrics = return_all_metrics
        self.type_data = type_data
        self.token_type2tokens = token_type2tokens
        self.time_range = time_range
        self.time_data = time_data
        self.ignore_label_status = ignore_label_status
        self.concept_id = concept_id

    def __call__(self, p, metrics_data=None):
        # We will do this in batches, because it can be very memory demanding
        metrics_data = metrics_data
        start = 0
        while start < len(p.predictions):
            predictions = p.predictions[start:start+self.batch_size]
            label_ids = p.label_ids[start:start+self.batch_size]
            if self.time_data is not None:
                time_data_batch = self.time_data[start:start+self.batch_size]
            else:
                time_data_batch = None
            type_data_batch = self.type_data[start:start+self.batch_size]

            if self.concept_id is None:
                metrics_data = precision(predictions, label_ids=label_ids, token_type2tokens=self.token_type2tokens,
                                         id2tkn=self.id2tkn, type_data=type_data_batch, old_data=metrics_data,
                                         predictions_are_scores=True, topk=self.topk, time_range=self.time_range,
                                         time_data=time_data_batch, ignore_label_status=self.ignore_label_status, **self.kwargs)
            else:
                metrics_data = precision_on_one(predictions, label_ids, concept_id=self.concept_id,
                                                old_data=metrics_data, topk=self.topk, time_range=self.time_range,
                                                time_data=time_data_batch, **self.kwargs)

            start += self.batch_size

        if self.return_all_metrics:
            return {
                'metrics_data': metrics_data, # Return all the metrics data too
            }
        else:
            return {
                'precision': metrics_data['precision']['all'],
                'precision_new': metrics_data['precision']['new'],
                'precision_old': metrics_data['precision']['old'],
                'macro_precision': metrics_data['macro_precision']['all'],
                'macro_precision_new': metrics_data['macro_precision']['new'],
                'macro_precision_old': metrics_data['macro_precision']['old'],
                'recall': metrics_data['recall']['all'],
                'recall_new': metrics_data['recall']['new'],
                'recall_old': metrics_data['recall']['old'],
            }
