import abc
import six
import re
import string
from functools import lru_cache
from nltk.tokenize import TreebankWordTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob, Word
from config.contractions import CONTRACTION_MAP
from utils.data_util import is_number


@six.add_metaclass(abc.ABCMeta)
class TokenizerBase(object):

    def text_normalization(self, text):
        text = text.strip().lower()
        text = self.contractions(text)
        return text

    @staticmethod
    def contractions(text):
        text_list = text.split()
        return_text_list = [CONTRACTION_MAP.get(w, w) for w in text_list]
        return_text = ' '.join(return_text_list)
        return return_text

    @staticmethod
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

    @abc.abstractmethod
    def tokenize(self, text):
        raise NotImplementedError


class WhitespaceTokenizerHelper(TokenizerBase):
    @lru_cache(maxsize=65535)
    def tokenize(self, text):
        text = text.strip().lower()
        return text.split()


class TextBlobTokenizerHelper(TokenizerBase):
    def __init__(self, tokenizer=TreebankWordTokenizer, **kwargs):
        super(TextBlobTokenizerHelper, self).__init__()
        self.tokenizer = tokenizer()
        self.kwargs = kwargs

    @lru_cache(maxsize=65535)
    def tokenize(self, text):
        text = self.text_normalization(text)
        tb = TextBlob(text, tokenizer=self.tokenizer)
        # tb = tb.correct()
        words = list()
        for w in tb.tokens:
            if is_number(w):
                words.append(self.kwargs.get('num_word'))
            elif w in string.punctuation:
                words.append(self.kwargs.get('punc_word'))
            else:
                words.append(Word(w).lemmatize())
        return words


class NLTKTokenizerHelper(TokenizerBase):
    def __init__(self, tokenizer=word_tokenize, **kwargs):
        super(NLTKTokenizerHelper, self).__init__()
        self.tokenizer = tokenizer()
        self.kwargs = kwargs
        self.lemmatizer = WordNetLemmatizer()

    @lru_cache(maxsize=65535)
    def tokenize(self, text):
        text = self.text_normalization(text)
        tokens = self.tokenizer(text)
        # tb = tb.correct()
        words = list()
        for w in tokens:
            if is_number(w):
                words.append(self.kwargs.get('num_word'))
            elif w in string.punctuation:
                words.append(self.kwargs.get('punc_word'))
            else:
                words.append(self.lemmatizer.lemmatize(w))
        return words
