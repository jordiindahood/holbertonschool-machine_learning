#!/usr/bin/env python3
"""
script 0
"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Dataset class
    """

    def __init__(self):
        """
        init
        """
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en', split='train', as_supervised=True
        )
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True,
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for both the Portuguese
        and English datasets.
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        def iterate_pt():
            """iterate portuguese"""
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def iterate_en():
            """iterate english"""
            for _, en in data:
                yield en.numpy().decode('utf-8')

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            iterate_pt(), vocab_size=2**13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            iterate_en(), vocab_size=2**13
        )
        return tokenizer_pt, tokenizer_en
