import numpy as np
import os
import logging
import re

logger = logging.getLogger('CommonUtils')
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)4s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


class CommonUtiler(object):
    def __init__(self):
        pass

    @staticmethod
    def split_sentence(sentence):
        """Tokenize a sentence. All characters are transfered to lower case."""
        # break sentence into a list of words and punctuation
        SPLIT_RE = re.compile(r'(\W+)')
        sentence = [s.strip().lower() for s in SPLIT_RE.split(sentence.strip()) \
                    if len(s.strip()) > 0]
        # remove the '.' from the end of the sentence
        if sentence[-1] != '.':
            return sentence
        else:
            return sentence[:-1]

    @staticmethod
    def add_zero_fname(total_len, num):
        """Add zeros to file names. Useful for S3 files."""
        num_str = str(num)
        if len(num_str) > total_len:
            logger.fatal('Total length is too small to hold the number')
        return '0' * (total_len - len(num_str)) + num_str

    @staticmethod
    def truncate_list(l, num):
        if num == -1:
            num = len(l)
        return l[:min(len(l), num)]

    @staticmethod
    def create_dir_if_not_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def load_vocabulary(vocab_path):
        """Initialize vocabulary from file."""
        assert os.path.exists(vocab_path), 'File does not exists %s' % vocab_path
        rev_vocab = []
        with open(vocab_path) as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab

    @staticmethod
    def encode_sentence(sentence, vocab, flag_add_bos=False):
        """Encode words in a sentence with their index in vocab."""
        assert '<unk>' in vocab
        sentence_encode = []
        for word in sentence:
            if word in vocab:
                sentence_encode.append(vocab[word])
            else:
                sentence_encode.append(vocab['<unk>'])
        if flag_add_bos:
            assert '<bos>' in vocab
            sentence_encode = [vocab['<bos>']] + sentence_encode + [vocab['<bos>']]
        return sentence_encode

    @staticmethod
    def decode_sentence(sentence_encode, vocab, rev_vocab,
                        flag_remove_bos=True):
        """Decode words index of a sentence to words."""
        if flag_remove_bos and sentence_encode[-1] == vocab['<bos>']:
            assert '<bos>' in vocab
            sentence = [rev_vocab[x] for x in sentence_encode[:-1]]
        else:
            sentence = [rev_vocab[x] for x in sentence_encode]
        return sentence

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    @staticmethod
    def coco_val_eval(pred_path, result_path):
        """Evaluate the predicted sentences on MS COCO validation."""
        from external.coco_caption.pycocotools.coco import COCO
        from external.coco_caption.pycocoevalcap.eval import COCOEvalCap
        coco = COCO('./external/coco-caption/annotations/captions_val2014.json')
        cocoRes = coco.loadRes(pred_path)

        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.params['image_id'] = cocoRes.getImgIds()
        cocoEval.evaluate()

        with open(result_path, 'w') as fout:
            for metric, score in cocoEval.eval.items():
                print('%s: %.3f' % (metric, score), file=fout)

    @staticmethod
    def load_config(config_path):
        variables = {}
        exec(compile(open(config_path).read(), config_path, 'exec'), variables)
        return variables['config']
