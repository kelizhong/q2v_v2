import re
import logging
from tqdm import tqdm
from utils.config_decouple import config
from helper.tokenizer_helper import TextBlobTokenizerHelper
from helper.tokens_helper import TokensHelper
from utils.data_util import sentence_gen
from utils.data_util import is_number
from helper.vocabulary_helper import VocabularyHelper


class QueryPairParser(object):
    def __init__(self, vocab_data_dir=None):
        self.tokenizer = TextBlobTokenizerHelper()
        vocab_data_dir = vocab_data_dir or config('vocabulary_dir')
        vocab_helper = VocabularyHelper(vocabulary_data_dir=vocab_data_dir)
        vocab = vocab_helper.load_vocabulary()
        self.tokens_helper = TokensHelper(vocabulary=vocab, unk_token=config('_unk_', section='vocabulary_symbol'))

    def data_text_list_generator(self, files):
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
                logging.error("Failed to extract query pair data %s", sentence, exc_info=True, stack_info=True)

    @staticmethod
    def extract_train_data(text):
        line = re.sub(r'(?:^\(|\)$)', '', text)
        line = line.strip().lower()
        items = re.split(r'\t+', line)
        source = items[0]
        items = items[1:]
        target_label_list = [(items[i], items[i+1]) for i in range(0, len(items), 2)]
        return source, target_label_list

    def tokenize_data(self, text):
        items = [] if text is None else text.split("\t")
        tokens_list = list()
        for item in items:
            tokens = self.tokenize(item)
            tokens_list.extend(tokens)
        return tokens_list

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def siamese_sequences_to_tokens_generator(self, files, min_words=2):
        samples = None
        for source, target_label_list in tqdm(self.extract_train_data_generator(files)):
            if samples is not None and samples != len(target_label_list):
                raise ValueError("target column size not equal, %s:%s", samples, len(target_label_list))
            else:
                samples = len(target_label_list)
            source_tokens = self.tokens_helper.tokens(source, return_data=False)
            if source_tokens is None or len(source_tokens) < min_words:
                logging.warning("source %s is none or length is less than %d", source, min_words)
                continue
            target_list = list()
            label_list = list()
            for target, label in target_label_list:
                target_tokens = self.tokens_helper.tokens(target, return_data=False)
                if target_tokens is None or len(target_tokens) < min_words:
                    logging.warning("target %s is none or length is less than %d", target, min_words)
                    break
                if label is None or not is_number(label):
                    logging.warning("target label %s is none, or not a number", label)
                    break
                target_list.append(target_tokens)
                label_list.append(float(label))
            if len(target_list) == samples:
                yield source_tokens, target_list, label_list
            else:
                logging.warning("Failed to tokens target %s, may be the target not meet the select rule(min_words)", str(target_label_list))