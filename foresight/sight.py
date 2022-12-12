import torch.nn as nn
import numpy as np
import torch
from foresight.utils.cdb_utils import get_parents_map, get_children_map, get_siblings_map

class Sight(object):
    def __init__(self, tokenizer, device, model, cat):
        self.tokenizer = tokenizer
        self.device = device
        self.model = model
        self.cat = cat

    def _predict(self, stream, create_position_ids=False, skip_oov=False):
        self.model.eval()
        _stream = self.tokenizer(stream, return_tensors=True, device=self.model.device, skip_oov=skip_oov)

        # Create position ids
        if create_position_ids:
            position_ids = []
            time = 0
            for tkn in stream:
                position_ids.append(time)
                if tkn.startswith('<SEP'):
                    time += 1
            _stream['position_ids'] = torch.tensor([position_ids]).to(self.device)

        logits = self.model.forward(**_stream)['logits']
        smax = nn.Softmax(dim=0)
        p = smax(logits[0, -1, :]).detach().cpu().numpy()

        return p


    def next_concepts(self, stream, type_ids=None, n=5, p_new=True, p_old=False, create_position_ids=False, prediction_filters=[], cui_filter=None,
                      skip_oov=False):
        r'''
        stream: Stream of concepts to use as history
        type_ids: What type ids to predict
        n: how many to predict
        p_new: do we want new concepts
        p_old: do we want old concepts
        prediction_filters: list of things to ignore: ['Ignore Siblings', 'Ignore Children', 'Ignore Parents']
        cui_filter: list of cuis for which we will get all children and predictions will be limited to them
        '''
        # Simplification
        cat = self.cat

        id2token = self.tokenizer.id2tkn
        token_type2tokens = self.tokenizer.token_type2tokens
        input_ids = self.tokenizer(stream, skip_oov=skip_oov)['input_ids']

        if type_ids is not None:
            select_tokens = []
            for type_id in type_ids:
                select_tokens.extend(token_type2tokens[type_id])
        else:
            select_tokens = None
        ps = self._predict(stream, create_position_ids=create_position_ids)
        preds = np.argsort(-1 * ps)
        candidates = []
        sep_ids = [self.tokenizer.tkn2id[x] for x in self.tokenizer.token_type2tokens['sep']]

        ignore_cuis = set()
        def update_ignore_cuis(ignore_cuis, prediction_filters, cui):
            if 'Ignore Siblings' in prediction_filters:
                ignore_cuis.update(get_siblings_map([cui], cat.cdb.addl_info['pt2ch'], cat.cdb.addl_info['ch2pt'])[cui])
            if 'Ignore Children' in prediction_filters:
                ignore_cuis.update(get_children_map([cui], cat.cdb.addl_info['pt2ch'])[cui])
            if 'Ignore Parents' in prediction_filters:
                ignore_cuis.update(get_parents_map([cui], cat.cdb.addl_info['pt2ch'], cat.cdb.addl_info['ch2pt'])[cui])

            ignore_cuis.add(cui)

            return ignore_cuis
        for cui in stream:
            if cui in cat.cdb.addl_info['pt2ch']:
                ignore_cuis = update_ignore_cuis(ignore_cuis, prediction_filters, cui)

        use_cuis = set()
        if cui_filter:
            for cui in cui_filter:
                cui = str(cui).strip()
                if cui in cat.cdb.addl_info['pt2ch']:
                    use_cuis.update(get_children_map([cui], cat.cdb.addl_info['pt2ch'], depth=10)[cui])
                    use_cuis.add(cui)

        for pred in preds:
            is_new = True if pred not in input_ids else False
            if (select_tokens is None or id2token[pred] in select_tokens) and \
                    (((p_new and p_new == is_new) or (p_old and p_old != is_new)) or (pred in sep_ids)):
                # More filters
                cui = id2token[pred]
                if cui not in ignore_cuis and (not use_cuis or cui in use_cuis):
                    candidates.append((cui, ps[pred]))

                    if len(candidates) >= n:
                        break
        print(len(candidates))
        return candidates


    def mcq(self, question, options, do_print=False):
        option2p = {}
        ps = self._predict(question)

        for option in options:
            tkn_id = self.tokenizer.tkn2id[option]
            option2p[option] = {'original': ps[tkn_id],
                                'cnt': self.tokenizer.global_token_cnt[option]}

        p_sum = sum([v['original'] for v in option2p.values()])

        for option in options:
            tkn_id = self.tokenizer.tkn2id[option]
            option2p[option]['norm'] = ps[tkn_id] / p_sum

        if do_print:
            for tkn in question:
                print("{:5}: {:20} - {}".format(
                    self.tokenizer.global_token_cnt.get(tkn, 0),
                    self.tokenizer.tkn2name[tkn],
                    tkn))
            print()
            for option in options:
                option_name = self.tokenizer.tkn2name[option]
                print("{:5}: {:50} - {:20}- {:.2f} - {:.2f}".format(
                                                       option2p[option]['cnt'],
                                                       option_name[:50],
                                                       option,
                                                       option2p[option]['original'],
                                                       option2p[option]['norm']))


        return option2p
