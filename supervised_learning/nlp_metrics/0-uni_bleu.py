#!/usr/bin/env python3
"""script 0"""

import math
from collections import Counter


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence
    """
    sentence_counts = Counter(sentence)

    clipped_counts = {}

    for word in sentence_counts:
        max_ref_count = max(ref.count(word) for ref in references)
        clipped_counts[word] = min(sentence_counts[word], max_ref_count)

    clipped_total = sum(clipped_counts.values())
    total_unigrams = len(sentence)

    precision = clipped_total / total_unigrams

    ref_lens = [len(ref) for ref in references]
    len_s = len(sentence)
    closest_ref_len = min(ref_lens, key=lambda r: (abs(r - len_s), r))

    if len_s > closest_ref_len:
        bp = 1
    else:
        bp = math.exp(1 - closest_ref_len / len_s)

    bleu = bp * precision
    return bleu
