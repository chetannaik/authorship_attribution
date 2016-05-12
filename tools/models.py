""" Defines model classes needed for the task."""

import os
import cPickle as pickle
import numpy as np
import scipy.sparse as sp

from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from tools import util


class LangModel:
    """ Implements Language Model."""
    def __init__(self):
        self.data = None
        self.prob_words = {}
        self.max_order = 5

    def train(self, data):
        """Train language model of orders 0-4."""
        self.data = data
        for order in range(self.max_order):
            self.prob_words.update(self.train_lm(order=order))

    def save(self, file_name):
        """Save language model probabilities."""
        pickle.dump(self.prob_words, open(file_name, "wb"))

    def load(self, file_name):
        """Load language model probabilities."""
        self.prob_words = pickle.load(open(file_name, "rb"))

    def train_lm(self, order=2):
        """Train language model of a order passed as argument."""
        lm = defaultdict(Counter)
        pad = "<bgn> " * order
        data = pad + self.data
        data = data.split()
        for i in xrange(len(data)-order):
            history, char = data[i:i+order], data[i+order]
            history = ' '.join(history).strip()
            lm[history][char] += 1

        def normalize(counter):
            s = float(sum(counter.values()))
            return [(c, cnt/s) for c, cnt in counter.iteritems()]

        out = dict((hist, normalize(chars)) for hist, chars in lm.iteritems())
        return out

    def predict(self, context):
        """Predict the next word using language model. Reapeatedly try
        predicting next word using longest possible context to null context
        until a match is found.
        """
        context = context.split()[-self.max_order:]
        while context:
            key = ' '.join(context).strip()
            if (key in self.prob_words):
                words = map(lambda x: x[0], self.prob_words[key])
                probs = map(lambda x: x[1], self.prob_words[key])
                sample = np.random.choice(words, 1, p=probs)
                return sample[0]
            context = context[1:]
        key = ''
        words = map(lambda x: x[0], self.prob_words[key])
        probs = map(lambda x: x[1], self.prob_words[key])
        sample = np.random.choice(words, 1, p=probs)
        return sample[0]


class AuthorshipAttribution:
    """ Implements authorship attribution models."""
    def __init__(self, data_set):
        self.corpus = []
        self.book_labels = []
        self.author_labels = []
        self.tags = []
        for item in data_set:
            self.corpus.append(item["text"])
            self.book_labels.append(item["book"])
            self.author_labels.append(item["author"])
            self.tags.append(item["pos"])

        self.sample_clf = None
        self.author_clf = None
        self.author_lms = {}

        # word ngram feature generators
        self.word_vector = TfidfVectorizer(analyzer="word", ngram_range=(2, 2),
                                           max_features=2000, binary=False,
                                           decode_error='ignore')
        # char ngram feature generators
        self.char_vector = TfidfVectorizer(analyzer="char", ngram_range=(2, 3),
                                           max_features=2000, binary=False,
                                           decode_error='ignore', min_df=0)
        # POS ngram feature generators
        self.tag_vector = TfidfVectorizer(analyzer="word", ngram_range=(2, 2),
                                          max_features=2000, binary=False,
                                          decode_error='ignore')
        # punctuation frequency feature generators
        self.punct_vector = TfidfVectorizer(analyzer='char',
                                            preprocessor=util.retain_punct,
                                            max_features=2000, binary=False,
                                            use_idf=False,
                                            decode_error='ignore')
        # concatenate features generators
        self.vectorizer = FeatureUnion([("chars", self.char_vector),
                                        ("words", self.word_vector),
                                        ("puncts", self.punct_vector)])

        # generate features
        print "- Generating features"
        X1 = self.vectorizer.fit_transform(self.corpus)
        X2 = self.tag_vector.fit_transform(self.tags)
        # concatenate two feature matrices
        matrix = sp.hstack((X1, X2))
        self.X = matrix.toarray()

    def generate_test_features(self, corpus, classes, tags):
        """Generate feature matrix of the test corpus passes as argument."""
        X1 = self.vectorizer.transform(corpus)
        X2 = self.tag_vector.transform(tags)
        # concatenate two matrices
        matrix = sp.hstack((X1, X2))
        X = matrix.toarray()
        y = np.asarray(classes)
        return (X, y)

    def train_sample_model(self):
        """Train classifier needed to predict the book using sample text."""
        print "- Training book/work model"
        save_location = os.path.join("models", "clf", "sample_model.p")
        # check and load if a saved model is present, if not train a new one
        if os.path.isfile(save_location):
            self.sample_clf = pickle.load(open(save_location, "rb"))
        else:
            X_train, y_train = self.X, np.asarray(self.book_labels)
            model = SVC(kernel='rbf')
            self.sample_clf = model.fit(X_train, y_train)
            pickle.dump(self.sample_clf, open(save_location, "wb"))

    def train_author_model(self):
        """Train classifier needed to predict the author using sample text."""
        print "- Training author model"
        save_location = os.path.join("models", "clf", "author_model.p")
        # check and load if a saved model is present, if not train a new one
        if os.path.isfile(save_location):
            self.author_clf = pickle.load(open(save_location, "rb"))
        else:
            X_train, y_train = self.X, np.asarray(self.author_labels)
            model = LinearSVC(loss='hinge', dual=True)
            self.author_clf = model.fit(X_train, y_train)
            pickle.dump(self.author_clf, open(save_location, "wb"))

    def train_lang_model(self):
        """Train language model needed to predict the next word."""
        print "- Training language model"
        author_data = {}
        for author, book in zip(self.author_labels, self.corpus):
            save_location = os.path.join("models", "lm", author+".p")
            if os.path.isfile(save_location):
                continue
            else:
                if author in author_data:
                    author_data[author] = author_data[author] + book
                else:
                    author_data[author] = book
        print "  - LM for: [",
        for author in set(self.author_labels):
            print author,
            save_location = os.path.join("models", "lm", author+".p")
            if os.path.isfile(save_location):
                lm = LangModel()
                lm.load(save_location)
                self.author_lms[author] = lm
            else:
                lm = LangModel()
                works = author_data[author]
                lm.train(works)
                self.author_lms[author] = lm
                lm.save(save_location)
        print " ]"

    def predict_word(self, context, author=None):
        """Predict next word. This first predicts the author if author is not
        passed as an argument, then predicts the next word using author's
        language model.
        """
        if not author:
            author = self.recognize_author([context])[0]
        # print "- predicting word using {}'s language model.".format(author)
        author_lm = self.author_lms[author]
        return author_lm.predict(context)

    def recognize_sample(self, test_text, test_class=None, test_tags=None):
        """Interface to call the classifier and predict the work using sample
        text.
        """
        if not test_tags:
            text_tags = [util.get_pos_tags(txt) for txt in test_text]
        else:
            text_tags = test_tags
        text_class = test_class

        if not (self.sample_clf):
            self.train_sample_model()

        X_test, y_test = self.generate_test_features(test_text,
                                                     text_class,
                                                     text_tags)
        y_pred = self.sample_clf.predict(X_test)
        return y_pred

    def recognize_author(self, test_text, test_class=None, test_tags=None):
        """Interface to call the classifier and predict the author using sample
        text.
        """
        if not test_tags:
            text_tags = [util.get_pos_tags(txt) for txt in test_text]
        else:
            text_tags = test_tags
        text_class = test_class

        if not (self.author_clf):
            self.train_author_model()

        X_test, y_test = self.generate_test_features(test_text,
                                                     text_class,
                                                     text_tags)
        y_pred = self.author_clf.predict(X_test)
        return y_pred
