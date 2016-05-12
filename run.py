import random
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tools import util, models


def generate_test_corpus(data_set, num=50):
    """Generate corpus for testing."""
    test_corpus = []
    test_books = []
    test_authors = []
    test_tags = []
    test_data_set = [random.choice(data_set) for _ in range(num)]
    for item in test_data_set:
        lower_bound = random.randint(0, len(item["text"])-2)
        upper_bound = random.randint(lower_bound+1, len(item["text"]))
        test_books.append(item["book"])
        test_authors.append(item["author"])
        test_corpus.append(item["text"][lower_bound:upper_bound])
        test_tags.append(item["pos"][lower_bound:upper_bound])
    return (test_corpus, test_books, test_authors, test_tags)


def evaluate(test_corpus, y_test, test_tags, recognize_function):
    """Evaluate the corpus by predicting values using passed function."""
    y_pred = recognize_function(test_corpus, test_tags=test_tags)
    cm = confusion_matrix(y_test, y_pred)
    print "- Confusion matrix"
    print cm, "\n"
    print "Precision:", metrics.precision_score(y_pred, y_test,
                                                average='weighted')
    print "Recall   :", metrics.recall_score(y_pred, y_test,
                                             average='weighted')
    print "F1       :", metrics.f1_score(y_pred, y_test,
                                         average='weighted')


def main():
    # print "- Pre-Processing dataset"
    # util.process_corpus('data/raw')
    data_set = util.read_data('data')
    # limit dataset
    aa = models.AuthorshipAttribution(data_set)
    aa.train_lang_model()
    aa.train_sample_model()
    aa.train_author_model()

    tst_corpus, tst_books, tst_authors, tst_tags = generate_test_corpus(
        data_set, 150)

    print "\n1. Work/Book Classifier Evaluation:"
    evaluate(tst_corpus, tst_books, tst_tags, aa.recognize_sample)

    print "\n\n2. Author Classifier Evaluation:"
    evaluate(tst_corpus, tst_authors, tst_tags, aa.recognize_author)

    print "\n\n3. Predicting next word:"
    print "\nContext:", "He heard"
    print "Next Word:", aa.predict_word("He heard")

    print "\nContext:", "I am"
    print "Next Word:", aa.predict_word("I am")

    print "\nContext:", "I am"
    print "Next Word:", aa.predict_word("I am", author="shakespeare")


if __name__ == '__main__':
    main()
