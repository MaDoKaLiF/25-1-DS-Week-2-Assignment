# A Lightweight Word Piece Tokenizer

[![PyPI version shields.io](https://img.shields.io/pypi/v/word-piece-tokenizer.svg)](https://pypi.org/project/word-piece-tokenizer/)

This library is an implementation of a modified version of [Huggingface's Bert Tokenizer](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizer) in pure python.

## Table of Contents

1. [Usage](#usage)
   - [Installing](#installing)
   - [Example](#example)
   - [Running Tests](#running-tests)
1. [Making it Lightweight](#making-it-lightweight)
   - [Optional Features](#optional-features)
   - [Unused Features](#unused-features)
1. [Matching Algorithm](#matching-algorithm)
   - [The Trie](#the-trie)

## Usage

### Installing

Install and update using [pip](https://pip.pypa.io/en/stable/getting-started/)

```shell
pip install word-piece-tokenizer
```

### Example

```python
The tree fell unexpectedly short.
[464, 5509, 3214, 25884, 1790, 13]
['[UNK]', 'tree', 'fell', 'unexpectedly', 'short', '##.']

 Performance Results:
BPE tokenizer: 6.458908319473267e-05
This tokenizer: 9.03010368347168e-06
This tokenizer is 86.02% faster
[Average] This tokenizer is 81.31% faster

you are짱 짱짱bye bye
[5832, 389, 168, 100, 109, 23821, 100, 109, 168, 100, 109, 16390, 33847]
['you', '[UNK]', '[UNK]', 'bye']

 Performance Results:
BPE tokenizer: 0.00020454823970794678
This tokenizer: 1.5280209481716156e-05
This tokenizer is 92.53% faster
[Average] This tokenizer is 81.52% faster


```

### Running Tests

Test the tokenizer against hugging's face implementation:

```bash
pip install transformers
python tests/tokenizer_test.py
```

<br/>

## Making It Lightweight

To make the tokenizer more lightweight and versatile for usage such as embedded systems and browsers, the tokenizer has been stripped of optional and unused features.

### Optional Features

The following features has been enabled by default instead of being configurable:

| Category      | Feature                                                                                                                                                                                 |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tokenizer     | - The tokenizer utilises the pre-trained [bert-based-uncased](https://huggingface.co/bert-base-uncased) vocab list.<br>- Basic tokenization is performed before word piece tokenization |
| Text Cleaning | - Chinese characters are padded with whitespace<br>- Characters are converted to lowercase<br>- Input string is stripped of accent                                                      |

### Unused Features

The following features has been removed from the tokenizer:

- `pad_token`, `mask_token`, and special tokens
- Ability to add new tokens to the tokenizer
- Ability to never split certain strings (`never_split`)
- Unused functions such as `build_inputs_with_special_tokens`, `get_special_tokens_mask`, `get_vocab`, `save_vocabulary`, and more...

<br/>

## Matching Algorithm

The tokenizer's _longest substring token matching_ algorithm is implemented using a `trie` instead of _greedy longest-match-first_

