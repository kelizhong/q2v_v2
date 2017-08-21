import re
import logging
from tqdm import tqdm
from utils.config_decouple import config
from helper.tokens_helper import TokensHelper
from utils.data_util import sentence_gen
from utils.data_util import is_number
from helper.vocabulary_helper import VocabularyHelper

logger = logging.getLogger(__name__)


class QueryPairParser(object):

    def __init__(self, vocab_data_dir=None, tokenize_fn=None):
        """Parser for query pair data(positive and negative data)

        Parameters
        ----------
        vocab_data_dir : {str}, optional
            vocabulary for tokens (the default is None, for None, will load the vocabulary from config)
        """
        vocab_data_dir = vocab_data_dir or config('vocabulary_dir')
        vocab_helper = VocabularyHelper(vocabulary_data_dir=vocab_data_dir)
        vocab = vocab_helper.load_vocabulary()
        self.tokens_helper = TokensHelper(vocabulary=vocab, tokenize_fn=tokenize_fn, unk_token=config('_unk_', section='vocabulary_symbol'))

    def data_text_list_generator(self, files):
        """Generator for the data text, exclude the label/weight, usually for building vocabulary

        Parameters
        ----------
        files : {file list}
            train data file list

        Yields
        ------
        [list]
            Text list with train data text
        """
        for source, target_label_list in self.extract_train_data_generator(files):
            text = list()
            text.append(source)
            text.extend([target for target, _ in target_label_list])
            yield text

    def extract_train_data_generator(self, files):
        for sentence in sentence_gen(files):
            try:
                data = self.extract_train_data(sentence)
                yield data
            except Exception as e:
                logger.error("Failed to extract query pair data %s", sentence, exc_info=True, stack_info=True)

    @staticmethod
    def extract_train_data(text):
        """Extract train data

        Parameters
        ----------
        text : {str}
            one train data with format:
            source\ttarget\tlabel/weight\ttarget\tlabel/weight...

        Returns
        -------
        [tuple]
            source, target_label_list((target, label/weight))
        """
        line = re.sub(r'(?:^\(|\)$)', '', text)
        line = line.strip().lower()
        items = re.split(r'\t+', line)
        # source
        source = items[0]
        # positive and negative data (data, label/weight)
        items = items[1:]
        target_label_list = [(items[i], items[i+1]) for i in range(0, len(items), 2)]
        return source, target_label_list

    def siamese_sequences_to_tokens_generator(self, files, min_words=2):
        """Generator for train data in siamese network

        Parameters
        ----------
        files : {str}
            files for train data
        min_words : {number}, optional
            ignore item with length < min_words (the default is 2)

        Yields
        ------
        [tuple]
            source_tokens, target_list, label_list

        Raises
        ------
        ValueError
            Column size
        """
        samples = None
        for source, target_label_list in tqdm(self.extract_train_data_generator(files)):
            if samples is not None and samples != len(target_label_list):
                raise ValueError("target column size not equal, %s:%s", samples, len(target_label_list))
            else:
                # define target column size
                samples = len(target_label_list)

            source_tokens = self.tokens_helper.tokens(source, return_data=False)
            if source_tokens is None or len(source_tokens) < min_words:
                logger.warning("source %s is none or length is less than %d", source, min_words)
                continue
            target_list = list()
            label_list = list()
            for target, label in target_label_list:
                target_tokens = self.tokens_helper.tokens(target, return_data=False)
                if target_tokens is None or len(target_tokens) < min_words:
                    logger.warning("target %s is none or length is less than %d", target, min_words)
                    break
                if label is None or not is_number(label):
                    logger.warning("target label %s is none, or not a number", label)
                    break
                target_list.append(target_tokens)
                label_list.append(float(label))
            if len(target_list) == samples:
                yield source_tokens, target_list, label_list
            else:
                logger.warning("Failed to tokens target %s, may be the target not meet the select rule(min_words)", str(target_label_list))
