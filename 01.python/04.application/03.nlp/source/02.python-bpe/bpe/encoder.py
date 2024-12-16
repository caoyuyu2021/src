# -*- coding: utf-8 -*-
""" 一个学习字节对编码（BPE）的编码器，用于空格分隔的文本。可以进行分词、编码和解码。 """
from collections import Counter

try:
    from typing import Dict, Iterable, Callable, List, Any, Iterator
except ImportError:
    pass

from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm
import toolz
import json

# 定义一些特殊符号，表示词的开始、结束和特殊符号
DEFAULT_EOW = '__eow'  # End of Word
DEFAULT_SOW = '__sow'  # Start of Word
DEFAULT_UNK = '__unk'  # Unknown word
DEFAULT_PAD = '__pad'  # Padding token

class Encoder:
    """ 使用字节对编码（BPE）对空格分隔的文本进行编码。具体内容可参见 https://arxiv.org/abs/1508.07909 """

    def __init__(self, vocab_size=8192, pct_bpe=0.2, word_tokenizer=None,
                 silent=True, ngram_min=2, ngram_max=2, required_tokens=None,
                 strict=False, lowercase=True,
                 EOW=DEFAULT_EOW, SOW=DEFAULT_SOW, UNK=DEFAULT_UNK, PAD=DEFAULT_PAD):
        """ 初始化编码器

        :param vocab_size: 词汇表的总大小
        :param pct_bpe: 分配给字节对编码的词汇比例
        :param word_tokenizer: 用于分词的函数（默认使用 nltk 的 wordpunct_tokenize）
        :param silent: 是否禁用进度条
        :param ngram_min: 字节对编码中最小的 n-gram 长度
        :param ngram_max: 字节对编码中最大的 n-gram 长度
        :param required_tokens: 必须包含的特殊 token（例如 UNK, PAD）
        :param strict: 是否严格模式（在解码时遇到错误时抛出异常）
        :param lowercase: 是否将输入转换为小写
        :param EOW: 词的结束标记
        :param SOW: 词的开始标记
        :param UNK: 未知词标记
        :param PAD: 填充符号
        """
        if vocab_size < 1:
            raise ValueError('vocab size must be greater than 0.')

        # 设置特殊标记
        self.EOW = EOW
        self.SOW = SOW
        self.eow_len = len(EOW)
        self.sow_len = len(SOW)
        self.UNK = UNK
        self.PAD = PAD
        self.required_tokens = list(set(required_tokens or []).union({self.UNK, self.PAD}))
        self.vocab_size = vocab_size
        self.pct_bpe = pct_bpe
        self.word_vocab_size = max([int(vocab_size * (1 - pct_bpe)), len(self.required_tokens or [])])
        self.bpe_vocab_size = vocab_size - self.word_vocab_size
        self.word_tokenizer = word_tokenizer if word_tokenizer is not None else wordpunct_tokenize
        self.custom_tokenizer = word_tokenizer is not None
        self.word_vocab = {}  # 用于存储单词词汇表
        self.bpe_vocab = {}  # 用于存储字节对编码词汇表
        self.inverse_word_vocab = {}  # 用于存储单词词汇表的反向映射
        self.inverse_bpe_vocab = {}  # 用于存储字节对编码词汇表的反向映射
        self._progress_bar = iter if silent else tqdm  # 控制是否显示进度条
        self.ngram_min = ngram_min  # 设置最小 n-gram 长度
        self.ngram_max = ngram_max  # 设置最大 n-gram 长度
        self.strict = strict  # 是否开启严格模式
        self.lowercase = lowercase  # 是否转换为小写

    def mute(self):
        """ 禁用进度条 """
        self._progress_bar = iter

    def unmute(self):
        """ 启用进度条 """
        self._progress_bar = tqdm

    def byte_pair_counts(self, words):
        """ 计算空间分隔的词的字节对出现频率：
            例如： [('T h i s </w>', 4)] -> {'Th': 4, 'hi': 4, 'is': 4}
        """
        for token, count in self._progress_bar(self.count_tokens(words).items()):
            bp_counts = Counter()  # 计数字节对出现次数
            for ngram in token.split(' '):
                bp_counts[ngram] += count
            for ngram_size in range(self.ngram_min, min([self.ngram_max, len(token)]) + 1):
                ngrams = [''.join(ngram) for ngram in toolz.sliding_window(ngram_size, token.split(' '))]
                for ngram in ngrams:
                    bp_counts[''.join(ngram)] += count

            yield bp_counts

    def count_tokens(self, words):
        """ 计算词汇的频次 """
        token_counts = Counter(self._progress_bar(words))
        return {' '.join(token): count for token, count in token_counts.items()}

    def learn_word_vocab(self, sentences):
        """ 从句子中学习词汇表，选择最常见的词汇 """
        word_counts = Counter(word for word in toolz.concat(map(self.word_tokenizer, sentences)))
        for token in set(self.required_tokens or []):
            word_counts[token] = int(2**63)  # 确保特殊 token 始终被选中
        sorted_word_counts = sorted(word_counts.items(), key=lambda p: -p[1])
        return {word: idx for idx, (word, count) in enumerate(sorted_word_counts[:self.word_vocab_size])}

    def learn_bpe_vocab(self, words):
        """ 从剩余的词中学习字节对编码的词汇表 """
        vocab = Counter()  # 计数字节对的频次
        for token in {self.SOW, self.EOW}:
            vocab[token] = int(2**63)  # 给开始和结束符一个极高的频率
        for idx, byte_pair_count in enumerate(self.byte_pair_counts(words)):
            for byte_pair, count in byte_pair_count.items():
                vocab[byte_pair] += count
            if (idx + 1) % 10000 == 0:
                self.trim_vocab(10 * self.bpe_vocab_size, vocab)

        sorted_bpe_counts = sorted(vocab.items(), key=lambda p: -p[1])[:self.bpe_vocab_size]
        return {bp: idx + self.word_vocab_size for idx, (bp, count) in enumerate(sorted_bpe_counts)}

    def fit(self, text):
        """ 从文本中学习词汇表 """
        if self.lowercase:
            _text = [l.lower().strip() for l in text]
        else:
            _text = [l.strip() for l in text]
        # 首先学习单词词汇表
        self.word_vocab = self.learn_word_vocab(_text)

        remaining_words = [word for word in toolz.concat(map(self.word_tokenizer, _text))
                           if word not in self.word_vocab]
        self.bpe_vocab = self.learn_bpe_vocab(remaining_words)

        self.inverse_word_vocab = {idx: token for token, idx in self.word_vocab.items()}
        self.inverse_bpe_vocab = {idx: token for idx, token in self.bpe_vocab.items()}

    @staticmethod
    def trim_vocab(n, vocab):
        """ 修剪词汇表，删除低频率的 token """
        pair_counts = sorted(vocab.items(), key=lambda p: -p[1])
        pairs_to_trim = [pair for pair, count in pair_counts[n:]]
        for pair in pairs_to_trim:
            del vocab[pair]

    def subword_tokenize(self, word):
        """ 使用 BPE 将一个未知的词分解成子词 """
        end_idx = min([len(word), self.ngram_max])
        sw_tokens = [self.SOW]
        start_idx = 0

        while start_idx < len(word):
            subword = word[start_idx:end_idx]
            if subword in self.bpe_vocab:
                sw_tokens.append(subword)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.ngram_max])
            elif len(subword) == 1:
                sw_tokens.append(self.UNK)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.ngram_max])
            else:
                end_idx -= 1

        sw_tokens.append(self.EOW)
        return sw_tokens

    def tokenize(self, sentence):
        """ 将一个句子分解为词和子词的 tokens """
        if self.lowercase:
            word_tokens = self.word_tokenizer(sentence.lower().strip())
        else:
            word_tokens = self.word_tokenizer(sentence.strip())
        tokens = []
        for word_token in word_tokens:
            if word_token in self.word_vocab:
                tokens.append(word_token)
            else:
                tokens.extend(self.subword_tokenize(word_token))

        return tokens

    def transform(self, sentences):
        """ 将句子转换为数字编码 """
        return [[self.word_vocab.get(word, self.UNK) if word in self.word_vocab else self.bpe_vocab.get(word, self.UNK)
                 for word in self.tokenize(sentence)] for sentence in sentences]

    def inverse_transform(self, encoded_sentences):
        """ 将编码的句子转换回原始文本 """
        return [' '.join([self.inverse_word_vocab.get(idx, self.UNK) if idx in self.inverse_word_vocab
                          else self.inverse_bpe_vocab.get(idx, self.UNK) for idx in encoded_sentence])
                for encoded_sentence in encoded_sentences]

    def save(self, filepath):
        """ 保存模型 """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.vocabs_to_dict(), f)

    def load(self, filepath):
        """ 从文件加载模型 """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return self.from_dict(data)

    def vocabs_to_dict(self):
        """ 将词汇表转换为字典格式 """
        return {
            'word_vocab': self.word_vocab,
            'bpe_vocab': self.bpe_vocab,
            'inverse_word_vocab': self.inverse_word_vocab,
            'inverse_bpe_vocab': self.inverse_bpe_vocab
        }

    def from_dict(self, data):
        """ 从字典数据恢复模型 """
        self.word_vocab = data['word_vocab']
        self.bpe_vocab = data['bpe_vocab']
        self.inverse_word_vocab = data['inverse_word_vocab']
        self.inverse_bpe_vocab = data['inverse_bpe_vocab']
        return self
