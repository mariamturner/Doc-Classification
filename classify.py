import json
import numpy

def get_vocabulary(D):
    """
    Given a list of documents, where each document is represented as
    a list of tokens, return the resulting vocabulary. The vocabulary
    should be a set of tokens which appear more than once in the entire
    document collection plus the "<unk>" token.
    """
    once = set()
    tokens = set(["<unk>"])
    for doc in D:
        for token in doc:
            if token in once:
                tokens.add(token)
            else:
                once.add(token)
    return tokens


class BBoWFeaturizer(object):
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and the vocabulary
        as a set of tokens, compute the binary bag-of-words feature representation.
        This function should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        dict = {}
        for token in doc:
            if token in vocab:
                dict[token] = 1
            else:
                dict["<unk>"] = 1
        return dict

class CBoWFeaturizer(object):
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and the vocabulary
        as a set of tokens, compute the count bag-of-words feature representation.
        This function should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        dict = {}
        for token in doc:
            if token in vocab:
                if token in dict:
                    dict[token] += 1
                else:
                    dict[token] = 1
            else:
                if "<unk>" in dict:
                    dict["<unk>"] += 1
                else:
                    dict["<unk>"] = 1
        return dict

def compute_idf(D, vocab):
    """
    Given a list of documents D and the vocabulary as a set of tokens,
    where each document is represented as a list of tokens, return the IDF scores
    for every token in the vocab. The IDFs should be represented as a dictionary that
    maps from the token to the IDF value. If a token is not present in the
    vocab, it should be mapped to "<unk>".
    """
    doc_set_size = len(D)
    docs_for_token = {}
    idfs = {}

    for i, doc in enumerate(D):
        #doc_set_size += 1
        for token in doc:
            if token in vocab:
                if token in docs_for_token:
                    docs_for_token[token].add(i)
                else:
                    docs_for_token[token] = {i}
            else:
                if "<unk>" in docs_for_token:
                    docs_for_token["<unk>"].add(i)
                else:
                    docs_for_token["<unk>"] = {i}

    for token in docs_for_token:
        # underflow
        idfs[token] = numpy.log(doc_set_size / len(docs_for_token[token]))
    keys = idfs.keys()
    for token in (vocab - set(keys)):
        idfs[token] = 0
    return idfs
    
class TFIDFFeaturizer(object):
    def __init__(self, idf):
        """The idf scores computed via `compute_idf`."""
        self.idf = idf
    
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and
        the vocabulary as a set of tokens, compute
        the TF-IDF feature representation. This function
        should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        dict = {}
        cbow = CBoWFeaturizer()
        tf = cbow.convert_document_to_feature_dictionary(doc, vocab)
        for key, value in self.idf.items():
            if key in tf:
                v = value * tf[key]
                dict[key] = v
        return dict

# You should not need to edit this cell
def load_dataset(file_path):
    D = []
    y = []
    with open(file_path, 'r') as f:
        for line in f:
            instance = json.loads(line)
            D.append(instance['document'])
            y.append(instance['label'])
    return D, y

def convert_to_features(D, featurizer, vocab):
    X = []
    for doc in D:
        X.append(featurizer.convert_document_to_feature_dictionary(doc, vocab))
    return X
