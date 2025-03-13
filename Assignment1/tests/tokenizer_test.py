from transformers import GPT2TokenizerFast
import unittest
import timeit

import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.word_piece_tokenizer.WordPieceTokenizer import WordpieceTokenizer


class TestTokenizer(unittest.TestCase):
    performance = []

    def setUp(self):
        self._bpe_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self._my_tokenizer = WordpieceTokenizer()

    def tokenize_with_both_tokenizer(self, s: str):
        print()
        print(s)

        time_taken = []

        # HuggingFace BPE Tokenizer
        start_time = timeit.default_timer()
        batch = self._bpe_tokenizer([s])
        lib_res = batch.input_ids[0]
        time_taken.append(timeit.default_timer() - start_time)

        # This Tokenizer
        start_time = timeit.default_timer()
        my_res = self._my_tokenizer.tokenize(s)
        time_taken.append(timeit.default_timer() - start_time)

        print(lib_res)
        print(my_res)

        print("\n Performance Results:")
        print(f"BPE tokenizer: {time_taken[0]}")
        print(f"This tokenizer: {time_taken[1]}")
        performance = 1 - time_taken[1] / time_taken[0]
        TestTokenizer.performance.append(performance)
        print(f"This tokenizer is {100 * performance:.2f}% faster")
        print(f"[Average] This tokenizer is {sum(TestTokenizer.performance) / len(TestTokenizer.performance) * 100:.2f}% faster")

        return lib_res, my_res

    def test_normal_sentence(self):
        s = "This is the Hugging Face!"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        #self.assertEqual(lib_res, my_res)

    def test_long_word(self):
        s = "Pneumonoultramicroscopicsilicovolcanoconiosis"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        #self.assertEqual(lib_res, my_res)

    def test_long_word_in_sentence(self):
        s = "wow! Pneumonoultramicroscopicsilicovolcanoconiosis is such a long word!"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        #self.assertEqual(lib_res, my_res)

    def test_long_word_in_sentence_2(self):
        s = "internalization is the best thing in the world!"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        #self.assertEqual(lib_res, my_res)

    def test_random_characters(self):
        s = "sdaw aef asdf w"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        #self.assertEqual(lib_res, my_res)

    def test_hangeul_with_english_words(self):
        s = "abc-와와"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        #self.assertEqual(lib_res, my_res)

    def test_hangeul_words(self):
        s = "와빅 와와빅"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        #self.assertEqual(lib_res, my_res)

    def test_hangeul_words_with_punctuation(self):
        s = "와빅 '와'와빅"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        #self.assertEqual(lib_res, my_res)

    def test_englishwords_with_punctuation(self):
        s = "I'm saying 'running' this morning! Huggingface"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        #self.assertEqual(lib_res, my_res)

    def test_unknown_words(self):
        s = "짱 짱짱"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        #self.assertEqual(lib_res, my_res)

    def test_unknown_words_with_known_words(self):
        s = "you are짱 짱짱bye bye"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        #self.assertEqual(lib_res, my_res)

    def test_random_sentences(self):
        path = os.path.join(str(Path(__file__).resolve().parent), "tests.txt")
        with open(path, 'r') as f:
            sentences = f.read().split('\n')
            for s in sentences:
                lib_res, my_res = self.tokenize_with_both_tokenizer(s)
                #self.assertEqual(lib_res, my_res)
    
    def test_hashtags(self):
        s = "you are #good! ## bye bye"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        #self.assertEqual(lib_res, my_res)

    def test_mask_tokens(self):
        s = "hello [MASK]! how are you?"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        #self.assertEqual(lib_res, my_res)

if __name__ == '__main__':
    unittest.main()