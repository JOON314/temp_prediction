
import sys
sys.path.append('..')

from dataset import ptb

# Read test data
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, _, _ = ptb.load_data('test')
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]
print(xs)
print(ts)
