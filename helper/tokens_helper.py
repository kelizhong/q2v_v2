import re
from functools import lru_cache
from helper.tokenizer_helper import WhitespaceTokenizerHelper


class TokensHelper(object):
    """Convert word to id"""

    def __init__(self, vocabulary, tokenize_fn=None, **kwargs):
        self.vocabulary = vocabulary
        self.tokenize = tokenize_fn or WhitespaceTokenizerHelper().tokenize
        self.kwargs = kwargs

    @staticmethod
    def _find_ngrams(input_list, n):
        return zip(*[input_list[i:] for i in range(n)])

    @staticmethod
    def ngram_tokenize_word(word, ngram):
        word = re.sub('[^a-z0-9#.\'-, ]+', '', word.strip().lower())
        words_list = [''.join(ele) for ele in TokensHelper._find_ngrams('#' + word + '#', ngram)]
        return words_list

    def words_index_lookup(self, words_list, vocabulary):
        if not isinstance(words_list, list):
            words_list = [words_list]
        words_index = [vocabulary[d] if d in vocabulary else self.kwargs.get('unk_token') for d in words_list]
        return words_index

    @lru_cache(maxsize=65536)
    def tokens(self, text, ngram=3, return_data=True):
        text_index = list()
        if text and len(text.strip()) > 0:
            words = self.tokenize(text)
            for word in words:
                if word in self.vocabulary:
                    text_index.extend(self.words_index_lookup(word, self.vocabulary))
                else:
                    words_list = self.ngram_tokenize_word(word, ngram)
                    text_index.extend(self.words_index_lookup(words_list, self.vocabulary))

        result = (text_index, text) if return_data else text_index
        return result
