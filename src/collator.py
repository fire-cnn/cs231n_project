import torch
import pdb

class Gpt2ClassificationCollator(object):
    """ Data Collator for GPT-2 in classification task

    This class takes a tokenizer and a text stream to transform any test and
    labels into tokenized tensors to feed into a GTP2 model. This class expects
    data from a `torch.utils.data.DataLoader` in a `tuple`: `(img, txt, label)`.

    Attributes
    ----------
        None. Class is intended to be used as a function as part of the `torch`
        DataLoader.

    Arguments
    ---------
        - tokenizer obj: Transformer-type tokenizer to transform words into
          numbers
        - map_labels dict: A dictionary with a mapping between label text and a
          numeric value. 
        - max_sequence_length int: A value that tells the class the maximum
          length possible in a tensor. If `None`, the default will be the model
          max (GTP-2 has 1050 as a max)
    """

    def __init__(self, tokenizer, map_labels , max_sequence_length=None):
        self.tokenizer = tokenizer
        self.map_labels = map_labels

        if max_sequence_length is None:
            self.max_sequence_length = self.tokenizer.model_max_length
        else:
            self.max_sequence_length = max_sequence_length

    def __call__(self, sequences):

        img, texts, labels = sequences[0] 

        # Do tokenization
        inputs = self.tokenizer(text = texts,
                                return_tensors="pt",
                                padding=True,
                                max_length=self.max_sequence_length)
        #inputs.update({"labels": torch.tensor(labels)})

        return img, inputs, labels
