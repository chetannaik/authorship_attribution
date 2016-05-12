""" Defines utilities needed for the task."""

import os
import nltk
import re
import string
import cPickle as pickle


def valid_file(input_dir, f, ext='.txt'):
    """Check if the file is a valid with with ext as extension."""
    if os.path.isfile(os.path.join(input_dir, f)):
        if f.endswith(ext):
            return True
    return False


def process_corpus(input_dir):
    """Process corpus and save it as pickle files."""
    data_files = [f for f in os.listdir(input_dir) if valid_file(input_dir, f)]
    data_set = []
    for file_name in data_files:
        with open(os.path.join(input_dir, file_name), 'r') as f:
            f_text = f.read()
            f_text = unicode(f_text, errors='replace')
            f_text = f_text.encode('ascii', 'ignore')
            f_text = f_text.replace('\n', ' ').replace('\r', '')
            f_text = re.sub(' +', ' ', f_text)
            author, book = file_name.rstrip('.txt').split('-')
            tags = get_pos_tags(f_text)
            f_dict = {'label': file_name,
                      'author': author,
                      'book': book,
                      'text': f_text,
                      'pos': tags}
            data_set.append(f_dict)
            save_dir = '/'.join(input_dir.split('/')[:-1])
            newf = file_name.replace('.txt', '')
            open(os.path.join(save_dir, newf + '.data'), 'w').write(f_text)
            open(os.path.join(save_dir, newf + '.pos'), 'w').write(tags)
            pickle.dump(f_dict, open(os.path.join(save_dir, newf+".p"), "wb"))
    return data_set


def read_data(input_dir):
    """Load processes pickle files."""
    data_files = [f for f in os.listdir(input_dir) if valid_file(input_dir,
                                                                 f, '.p')]
    data_set = []
    for file_name in data_files:
        f_dict = pickle.load(open(os.path.join(input_dir, file_name), "rb"))
        data_set.append(f_dict)
    return data_set


def get_pos_tags(txt):
    """Return POS tags of the sentence passed."""
    tokens = nltk.word_tokenize(txt)
    return " ".join([tag for (word, tag) in nltk.pos_tag(tokens)])


def retain_punct(txt):
    """Retain only the punctiations in the passed sentence."""
    include = set(string.punctuation)
    puncts = ''.join(ch for ch in txt if ch in include)
    return puncts
