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
    tokens = set()
    for doc in D:
        for token in doc:
            if token in once: tokens.add(token)         
            else: once.add(token)
    tokens.add("<unk>")
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
            if token in vocab: dict[token] = 1
            else: dict["<unk>"] = 1
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
                if token in dict: dict[token] += 1
                else: dict[token] = 1
            else:
                if "<unk>" in dict: dict["<unk>"] += 1
                else: dict["<unk>"] = 1
        return dict

def compute_idf(D, vocab):
    """
    Given a list of documents D and the vocabulary as a set of tokens,
    where each document is represented as a list of tokens, return the IDF scores
    for every token in the vocab. The IDFs should be represented as a dictionary that
    maps from the token to the IDF value. If a token is not present in the
    vocab, it should be mapped to "<unk>".
    """
    docs_for_token = {}
    idfs = {}

    for i, doc in enumerate(D):
        for token in doc:
            if token in vocab:
                if token in docs_for_token: docs_for_token[token].add(i)
                else: docs_for_token[token] = {i}
            else:
                if "<unk>" in docs_for_token: docs_for_token["<unk>"].add(i)
                else: docs_for_token["<unk>"] = {i}

    for token in docs_for_token:
        # underflow
        idfs[token] = numpy.log(len(D) / len(docs_for_token[token]))
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


def train_naive_bayes(X, y, k, vocab):
    """
    Computes the statistics for the Naive Bayes classifier.
    X is a list of feature representations, where each representation
    is a dictionary that maps from the feature name to the value.
    y is a list of integers that represent the labels.
    k is a float which is the smoothing parameters.
    vocab is the set of vocabulary tokens.
    
    Returns two values:
        p_y: A dictionary from the label to the corresponding p(y) score
        p_v_y: A nested dictionary where the outer dictionary's key is
            the label and the innner dictionary maps from a feature
            to the probability p(v|y). For example, `p_v_y[1]["hello"]`
            should be p(v="hello"|y=1).
    """
    
    raise NotImplementedError


def predict_naive_bayes(D, p_y, p_v_y):
    """
    Runs the prediction rule for Naive Bayes. D is a list of documents,
    where each document is a list of tokens.
    p_y and p_v_y are output from `train_naive_bayes`.
    
    Note that any token which is not in p_v_y should be mapped to
    "<unk>". Further, the input dictionaries are probabilities. You
    should convert them to log-probabilities while you compute
    the Naive Bayes prediction rule to prevent underflow errors.
    
    Returns two values:
        predictions: A list of integer labels, one for each document,
            that is the predicted label for each instance.
        confidences: A list of floats, one for each document, that is
            p(y|d) for the corresponding label that is returned.
    """
    raise NotImplementedError


def train_semi_supervised(X_sup, y_sup, D_unsup, X_unsup, D_valid, y_valid, k, vocab, mode):
    """
    Trains the Naive Bayes classifier using the semi-supervised algorithm.
    
    X_sup: A list of the featurized supervised documents.
    y_sup: A list of the corresponding supervised labels.
    D_unsup: The unsupervised documents.
    X_unsup: The unsupervised document representations.
    D_valid: The validation documents.
    y_valid: The validation labels.
    k: The smoothing parameter for Naive Bayes.
    vocab: The vocabulary as a set of tokens.
    mode: either "threshold" or "top-k", depending on which selection
        algorithm should be used.
    
    Returns the final p_y and p_v_y (see `train_naive_bayes`) after the
    algorithm terminates.    
    """
    raise NotImplementedError
