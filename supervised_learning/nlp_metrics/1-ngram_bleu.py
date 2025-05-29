#!/usr/bin/env python3
"""script 1"""

import math
from collections import Counter


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence
    """

    def ngrams(seq, n):
        return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]

    sentence_ngrams = ngrams(sentence, n)
    sentence_counts = Counter(sentence_ngrams)

    max_ref_counts = Counter()
    for ref in references:
        ref_ngrams = Counter(ngrams(ref, n))
        for ngram in ref_ngrams:
            max_ref_counts[ngram] = max(
                max_ref_counts[ngram], ref_ngrams[ngram]
            )

    clipped_counts = {
        ngram: min(count, max_ref_counts.get(ngram, 0))
        for ngram, count in sentence_counts.items()
    }

    clipped_total = sum(clipped_counts.values())
    total_ngrams = max(len(sentence_ngrams), 1)

    precision = clipped_total / total_ngrams

    ref_lens = [len(ref) for ref in references]
    len_s = len(sentence)
    closest_ref_len = min(ref_lens, key=lambda r: (abs(r - len_s), r))

    if len_s > closest_ref_len:
        bp = 1
    else:
        bp = math.exp(1 - closest_ref_len / len_s)

    bleu = bp * precision
    return bleu
