def reverse_pt2ch(pt2ch):
    ch2pt = {}
    for pt in pt2ch:
        for ch in pt2ch[pt]:
            if ch in ch2pt:
                ch2pt[ch].add(pt)
            else:
                ch2pt[ch] = {pt}
    return ch2pt


def get_parents_map(cuis, pt2ch, ch2pt=None, depth=3):
    r''' Get a map from a concept to all of its parents up to the `depth`, meaning parents of parents and so on.

    Args:
        pt2ch (`Dict`):
            map from parent concept to children (this is
            usually what we have when building a CDB).

        depth (`int`, optional defaults to 3):
            Get only parents, or parents of parents also, or ...
    '''

    # First convert pt2ch into ch2pt
    if ch2pt is None:
        ch2pt = reverse_pt2ch(pt2ch)

    def get_parents(concept, ch2pt, depth):
        parents = set()
        parents.update(ch2pt.get(concept, []))
        if depth > 0:
            for pt in ch2pt.get(concept, []):
                parents.update(get_parents(pt, ch2pt, depth=depth-1))
        return parents

    ch2all_pt = {}
    for cui in cuis:
        ch2all_pt[cui] = get_parents(cui, ch2pt, depth=depth)

    return ch2all_pt


def get_children_map(cuis, pt2ch, depth=3):
    r''' Returns a map from a CUI to all chlidren of it until depth
    '''
    def get_children(concept, pt2ch, depth):
        children = set()
        children.update(pt2ch.get(concept, []))
        if depth > 0:
            for ch in pt2ch.get(concept, []):
                children.update(get_children(ch, pt2ch, depth=depth-1))
        return children

    pt2all_ch = {}
    for cui in cuis:
        pt2all_ch[cui] = get_children(cui, pt2ch, depth=depth)

    return pt2all_ch


def get_siblings_map(cuis, pt2ch, ch2pt=None):
    # First convert pt2ch into ch2pt
    if ch2pt is None:
        ch2pt = reverse_pt2ch(pt2ch)

    cui2sib = {}
    for cui in cuis:
        ps = ch2pt[cui]
        cui2sib[cui] = set()
        for p in ps:
            cui2sib[cui].update(pt2ch[p])

    return cui2sib
