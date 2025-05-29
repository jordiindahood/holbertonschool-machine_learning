#!/usr/bin/env python3
"""Calculate cumulative n-gram BLEU score"""

import math
from collections import Counter


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence
    """

    def ngrams(seq, n):
        return [tuple(seq[i: i + n]) for i in range(len(seq) - n + 1)]

    precisions = []
    for i in range(1, n + 1):
        # Sentence n-grams
        sentence_ngrams = ngrams(sentence, i)
        sentence_counts = Counter(sentence_ngrams)

        # Max n-gram counts from references
        max_ref_counts = Counter()
        for ref in references:
            ref_ngrams = Counter(ngrams(ref, i))
            for ngram in ref_ngrams:
                max_ref_counts[ngram] = max(
                    max_ref_counts[ngram], ref_ngrams[ngram]
                )

        # Clipped counts
        clipped = {
            ngram: min(count, max_ref_counts.get(ngram, 0))
            for ngram, count in sentence_counts.items()
        }

        clipped_total = sum(clipped.values())
        total_ngrams = max(len(sentence_ngrams), 1)
        precision = clipped_total / total_ngrams
        precisions.append(precision)

    # Brevity Penalty
    ref_lens = [len(ref) for ref in references]
    len_s = len(sentence)
    closest_ref_len = min(ref_lens, key=lambda r: (abs(r - len_s), r))

    if len_s > closest_ref_len:
        bp = 1
    else:
        bp = math.exp(1 - closest_ref_len / len_s)

    # Geometric mean of precisions (avoid log(0) with epsilon)
    smooth_precisions = [max(p, 1e-8) for p in precisions]
    log_precisions = [math.log(p) for p in smooth_precisions]
    bleu = bp * math.exp(sum(log_precisions) / n)

    return bleu
