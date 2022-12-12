import random
import torch
import os
import dill

class SimpleMapTokenizer(object):
    r''' Not even really a tokenizer, will take a list of tokens and
    covert them to IDs

    Args:
        tkn2id
        pad_id
        max_len
    '''
    def __init__(self, tkn2id=None, pad_id=None, max_len=50, tkn2name=None,
                 token_type2tokens=None, embeddings=None, global_token_cnt=None):
        self.tkn2id = tkn2id
        self.pad_id = pad_id
        self.max_len = max_len
        self.tkn2name = tkn2name
        self.token_type2tokens = token_type2tokens
        self.embeddings = embeddings
        self.global_token_cnt = global_token_cnt
        self.id2tkn = None

        # Create id2tkn 
        if tkn2id is not None:
            self.id2tkn = {v:k for k,v in self.tkn2id.items()}

    def __call__(self, text, return_tensors=False, device='cpu', skip_oov=False):
        r'''

        skip_oov: will skip out of vocabulary words, otherwise error
        '''
        out = {'input_ids': [], 'attention_mask': []}

        if isinstance(text, str):
            out['input_ids'] = out['input_ids'] + [self.tkn2id[tkn] for tkn in text.split("~~") if not skip_oov or tkn in self.tkn2id]
        elif isinstance(text, list):
            # It is pre_tokenized
            out['input_ids'] = [self.tkn2id[tkn] for tkn in text if not skip_oov or tkn in self.tkn2id]

        out['attention_mask'] = [float(x != self.pad_id) for x in out['input_ids']]

        if return_tensors:
            out = {k:torch.tensor([v]).to(device) for k,v in out.items()}

        return out


    def decode(self, token_ids, get_names=True):
        tkns = self.convert_ids2tokens(token_ids, get_names=get_names)
        if type(tkns) != list:
            tkns = [tkns]
        return " ".join(tkns)


    def convert_ids2tokens(self, token_ids, get_names=True):
        if type(token_ids) == torch.Tensor:
            token_ids = token_ids.tolist()
        if type(token_ids) == list and type(token_ids[0]) == torch.Tensor:
            token_ids = [x.tolist() for x in token_ids]

        # Same as decode, but needed for compatibility with ecco
        out = []
        if type(token_ids) != list:
            out = [self.id2tkn[int(token_ids)]]
        else:
            # Convert tokens to IDs
            out = [self.id2tkn[int(id)] for id in token_ids]

        if get_names:
            _out = []
            for tkn in out:
                _out.append(self.tkn2name.get(tkn, tkn))
                #_out.append(" | ")
            out = _out

        return out


    def tokens_to_ids(self, tokens):
        r''' This will skip tokens if they are not in the tkn2id dict
        '''
        out = [self.tkn2id[tkn] for tkn in tokens]

        return out


    def encode(self, examples, trim_to_max_len=['position_ids', 'time', 'token_type']):
        r''' Convert 'stream' in the examples from tokens to IDs, save as 'input_ids'. Use with HF datasets.map
        '''
        examples['input_ids'] = [self.tokens_to_ids(stream)[0:self.max_len] for stream in examples['stream']]
        examples['token_type_id'] = [self.tokens_to_ids(token_type)[0:self.max_len] for token_type in examples['token_type']]

        for key in trim_to_max_len:
            examples[key] = [example[0:self.max_len] for example in examples[key]]

        return examples


    def save(self, path):
        with open(path, 'wb') as f:
            dill.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        tokenizer = cls()
        with open(path, 'rb') as f:
            d = dill.load(f)
            for k in tokenizer.__dict__:
                if k in d:
                    tokenizer.__dict__[k] = d[k]
        return tokenizer
