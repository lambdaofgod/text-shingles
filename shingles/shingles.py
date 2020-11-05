import mmh3
from nltk import ngrams

from shingles.util import generate_random_seeds, minhash_similarity


def text_similarity(this_text, other_text, shingle_length=5, minhash_size=200, random_seed=5):
    """
    calculate similarity of text text using shingle similarity
    shingle similarity is defined as similarity of minhash obtained by hashing n-grams of length n=shingle_length 
    """
    this_shingles = ShingledText(this_text, random_seed=random_seed, shingle_length=shingle_length, minhash_size=minhash_size)
    other_shingles = ShingledText(other_text, random_seed=random_seed, shingle_length=shingle_length, minhash_size=minhash_size)
    return this_shingles.similarity(other_shingles)


class ShingledText:
    def __init__(self, text, random_seed=5, shingle_length=5, minhash_size=200):
        split_text = text.split()
        if len(split_text) < shingle_length:
            raise ValueError(u'input text is too short for specified shingle length of {}'.format(shingle_length))

        self.minhash = []
        self.shingles = ngrams(split_text, shingle_length)

        for hash_seed in generate_random_seeds(minhash_size, random_seed):
            min_value = float('inf')
            for shingle in ngrams(split_text, shingle_length):
                value = mmh3.hash(' '.join(shingle), hash_seed)
                min_value = min(min_value, value)
            self.minhash.append(min_value)

    def similarity(self, other_shingled_text):
        return minhash_similarity(self.minhash, other_shingled_text.minhash)

