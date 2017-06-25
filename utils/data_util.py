# coding=utf-8
"""util for data processing"""
import os
import re
import codecs
import string
import random
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from utils.pickle_util import load_pickle_object, save_obj_pickle
from cache import RandomSet
from enum import Enum, unique
from collections import Counter
import numpy as np
from itertools import chain


wn_lemmatizer = WordNetLemmatizer()


positive_label = 1
negative_label = 0


# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_BOS = b"_BOS"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _BOS, _EOS, _UNK]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

_WORD_SPLIT = re.compile(b"([.,!?\"';-@#)(])")
_DIGIT_RE = re.compile(br"\d")

@unique
class aksis_data_label(Enum):
    negative_label = 0
    positive_label = 1


def words_gen(filename, bos=None, eos=None):
    """Generator that yield each word in a line.
    Parameters
    ----------
        filename: str
            data file name
        bos: str
            tag, beginning of sentence
        eos: str
            tag, ending of sentence
    """
    with codecs.open(filename, encoding='utf-8', errors='ignore') as f:
        for line in f:
            tokens = tokenize(line)
            tokens = [bos] + tokens if bos is not None else tokens
            tokens = tokens + [eos] if eos is not None else tokens
            for w in tokens:
                w = w.strip().lower()
                if len(w):
                    yield w


def sentence_gen(files):
    """Generator that yield each sentence in a line.
    Parameters
    ----------
        files: list
            data file list
    """
    if not isinstance(files, list):
        files = [files]
    for filename in files:
        with codecs.open(filename, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip().lower()
                if len(line):
                    yield line


def aksis_sentence_gen(filename):
    """Generator that yield each sentence in aksis corpus.
    Parameters
    ----------
        filename: str
            data file name
    """
    for line in sentence_gen(filename):
        line = extract_query_title_from_aksis_data(line)
        if len(line):
            yield line


def stem_tokens(tokens, lemmatizer):
    """lemmatizer
    Parameters
    ----------
        tokens: list
            token for lemmatizer
        lemmatizer: stemming model
            default model is wordnet lemmatizer
    """
    return [lemmatizer.lemmatize(token) for token in tokens]


def tokenize(text, lemmatizer=wn_lemmatizer):
    """tokenize and lemmatize the text"""
    text = clean_html(text)
    tokens = word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    stems = stem_tokens(tokens, lemmatizer)
    return stems


def extract_query_title_from_aksis_data(sentence):
    """extract the query and title from aksis raw data, this function is for building up vocabulary
    Aksis data format: MarketplaceId\tAsin\tKeyword\t Score\tActionType\tDate
    ActioType: 1-KeywordsByAdds, 2-KeywordsBySearches, 3-KeywordsByPurchases, 4-KeywordsByClicks
    """
    sentence = sentence.strip().lower()
    items = re.split(r'\t+', sentence)
    if len(items) == 7 and len(items[2]) and len(items[6]):
        return items[2] + " " + items[6]
    else:
        return str()


def extract_raw_query_title_score_from_aksis_data(sentence):
    """extract the query, title and score from aksis raw data, this function is to generate training data
    score gives a rough idea about specificness of a query. For example query1: "iphone" and query2: "iphone 6s 64GB".
    In both the query customer is looking for iphone but query2 is more specific.
    Query specificity score is number which ranges from 0.0 to 1.0.
    Aksis data format: MarketplaceId\tAsin\tKeyword\t Score\tActionType\tDate
    ActioType: 1-KeywordsByAdds, 2-KeywordsBySearches, 3-KeywordsByPurchases, 4-KeywordsByClicks
    """
    sentence = sentence.strip().lower()
    items = re.split(r'\t+', sentence)
    if len(items) == 7 and len(items[2]) and len(items[3]) and len(items[6]):
        return items[2], items[6], items[3]
    else:
        return None, None, None


def query_title_score_generator_from_aksis_data(files, dropout=-1):
    """Generator that yield query, title, score in aksis corpus"""
    for line in sentence_gen(files):
        query, title, score = extract_raw_query_title_score_from_aksis_data(line)
        if query and title and score:
            if not is_hit(score, dropout):
                continue
            yield query, title


def negative_sampling_train_data_generator(files, neg_number, dropout=-1):
    rs = RandomSet()
    for query, title in query_title_score_generator_from_aksis_data(files, dropout):
        rs.add(title)
        yield query, title, aksis_data_label.positive_label.value
        for neg_title in rs.get_n_items(neg_number):
            yield query, neg_title, aksis_data_label.negative_label.value


def is_hit(score, dropout):
    """sample function to decide whether the data should be trained,
    not sample if dropout less than 0"""
    return dropout < 0 or float(score) > random.uniform(dropout, 1)


def load_vocabulary_from_pickle(path, top_words=40000, special_words=dict()):
    """load vocabulary from pickle
    Parameters
    ----------
        path: str
            corpus path
        top_words: int
            the max words num in the vocabulary
        special_words: dict
         special_words like <unk>, <s>, </s>
    Returns
    -------
        vocabulary
    """
    vocab_pickle = load_pickle_object(path)

    words_num = len(vocab_pickle)
    special_words_num = len(special_words)

    if words_num <= special_words_num:
        raise ValueError("the size of total words must be larger than the size of special_words")

    if top_words <= special_words_num:
        raise ValueError("the value of most_commond_words_num must be larger "
                         "than the size of special_words")

    vocab = dict()
    vocab.update(special_words)
    for word, _ in vocab_pickle:
        if len(vocab) >= top_words:
            break
        if word not in special_words:
            vocab[word] = len(vocab)

    return vocab


def sentence2id(sentence, the_vocab):
    """convert the sentence to the index in vocabulary"""
    words = [the_vocab[w.strip().lower()] if w.strip().lower() in the_vocab else the_vocab[config.unk_word] for w in
             sentence if len(w) > 0]
    return words


def word2id(word, the_vocab):
    """convert the word to the index in vocabulary"""
    word = word.strip().lower()
    return the_vocab[word] if word in the_vocab else the_vocab[config.unk_word]







def clean_html(html):
    """
    Copied from NLTK package.
    Remove HTML markup from the given string.

    Parameters
    ----------
        html: str
            the HTML string to be cleaned
    """

    # First we remove inline JavaScript/CSS:
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
    # Then we remove html comments. This has to be done before removing regular
    # tags since comments can contain '>' characters.
    cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
    # Next we can remove the remaining tags:
    cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
    # Finally, we deal with whitespace
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    return cleaned.strip()


def text_normalize(rawstr):
    tnstring = rawstr.lower()
    tnstring = re.sub("[^a-z0-9':#,$-]", " ", tnstring)
    tnstring = re.sub("\\s+", " ", tnstring).strip()
    return tnstring


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    sentence_normed = text_normalize(sentence)
    # sentence_normed = sentence.lower()
    for space_separated_fragment in sentence_normed.split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def sentence_to_tokens(sentence, vocabulary, tokenizer=None):
    """Convert a string to list of integers representing token-ids.
    For example, a sentence "How are you?" will be tokenized into
    ["How", "are", "you", "?"] and then lowercased.
    If vocabulary is {"how": 1, "are": 2, "you": 4, "?": 7"}
    this function will return [1, 2, 4, 7].
    If a word isn't recognized it is replaced with a UNK_ID.
    Args:
      sentence: the plain text input (How are you?)
      vocabulary: a dictionary mapping tokens to integers.
    Returns:
      a list of integers, the token-ids for the sentence.
    """

    sentence = clean_html(sentence)
    words = tokenizer(sentence) if tokenizer else basic_tokenizer(sentence)

    return [vocabulary.get(w, UNK_ID) for w in words]


def sentence_to_padding_tokens(sentence, vocabulary, max_seq_length, tokenizer=None):
    # get source input sequence and PADDING accordingly
    tokens = sentence_to_tokens(sentence, vocabulary, tokenizer)
    tokens_len = len(tokens)
    if tokens_len > max_seq_length:
        print(
            'Error Deteced!!! Source input seq length is:%s. \n Excced current MAX_SEQ_LENTH of %s. Try to increase limit!!!!' %
            (str(tokens_len), str(max_seq_length)))
        return 0, []

    tokens += [PAD_ID] * (max_seq_length - tokens_len)
    return tokens_len, tokens


def build_words_frequency_counter(vocabulary_data_dir, data_path, tokenizer=None):
    """
    Create vocabulary file (if it does not exist yet) from data file.
    Data file should have one sentence per line.
    Each sentence will be tokenized.
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.
    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
    """
    words_freq_counter_path = os.path.join(vocabulary_data_dir, "words_freq_counter")
    if not os.path.isfile(words_freq_counter_path):
        print("Building words frequency counter %s from data %s" % (words_freq_counter_path, data_path))

        def _word_generator():
            with open(data_path, 'r+') as f:
                for num, line in enumerate(f):
                    if num % 100000 == 0:
                        print("  processing line %d" % counter)
                    try:
                        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                    except Exception, e:
                        print("Tokenize failure: " + line)
                        continue
                    for word in tokens:
                        yield word

        counter = Counter(_word_generator())
        save_obj_pickle(counter, words_freq_counter_path, True)
        print('Vocabulary file created')


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads each sequence to the same length (length of the longest sequence).
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    # Raises
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
        return x


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def trigram_encoding(data, trigram_dict):
    data = re.sub('[^a-z0-9.&\\ ]+', '', data.lower())
    data_seq = data.split()
    if len(data_seq) <= 2:  # remove query that are shorter than 3 words
        raise Exception("trigram_encoding: the length of data should larger than 2 words, error data: %s" % data)

    data_triagrams = list(chain(*[find_ngrams("#" + qw + "#", 3) for qw in data_seq]))
    data_triagrams_index = [trigram_dict[d] if d in trigram_dict else len(trigram_dict) + 1 for d in data_triagrams]
    return data_triagrams_index


def trigram_sentence_to_padding_index(sentence, trigram_dict, maxlen):
    try:
        index = trigram_encoding(sentence, trigram_dict)
    except Exception, e:
        return 0, []
    original_len = len(index)
    index = pad_sequences(np.array([index]), padding='post', truncating='post', maxlen=maxlen)
    return original_len, index[0]