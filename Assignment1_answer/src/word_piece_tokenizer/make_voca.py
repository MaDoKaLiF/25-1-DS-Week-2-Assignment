import math
from collections import defaultdict, Counter
from pathlib import Path
from utils import whitespace_tokenize

def get_corpus(file_path):
    """파일에서 모든 텍스트 라인을 읽어 리스트로 반환합니다."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines

def get_initial_vocab(corpus):
    """
    초기 어휘 구성:
    각 단어를 문자 단위로 분해한 후 단어 경계를 나타내는 </w>를 붙여 vocabulary를 구성합니다.
    예: "hello" -> "h e l l o </w>"
    """
    vocab = Counter()
    for line in corpus:
        words = whitespace_tokenize(line)
        for word in words:
            tokens = list(word) + ["</w>"]
            token_seq = " ".join(tokens)
            vocab[token_seq] += 1
    return vocab

def get_pair_stats(vocab):
    """
    현재 vocabulary에서 각 단어(토큰 시퀀스) 내 인접한 토큰 쌍의 빈도를 계산합니다.
    """
    pairs = defaultdict(int)
    for token_seq, freq in vocab.items():
        tokens = token_seq.split()
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pairs[pair] += freq
    return pairs

def get_unigram_counts(vocab):
    """
    vocabulary 내의 모든 토큰(단일 문자 또는 병합된 토큰)의 빈도를 계산합니다.
    """
    unigram_counts = Counter()
    for token_seq, freq in vocab.items():
        tokens = token_seq.split()
        for token in tokens:
            unigram_counts[token] += freq
    return unigram_counts

def merge_vocab(pair, vocab):
    """
    주어진 pair(예: ("a", "b"))를 vocabulary의 모든 항목에서 병합합니다.
    "a b" 형태의 bigram을 "ab"로 치환합니다.
    """
    merged_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for token_seq, freq in vocab.items():
        new_seq = token_seq.replace(bigram, replacement)
        merged_vocab[new_seq] = freq
    return merged_vocab

def compute_likelihood_score(pair, pair_freq, unigram_counts, total_tokens):
    """
    likelihood 점수 계산:
    - 관측 빈도: pair_freq
    - 기대 빈도: (freq(token1) * freq(token2)) / total_tokens
    점수는 observed * log(observed/expected)로 계산합니다.
    """
    a, b = pair
    expected = (unigram_counts[a] * unigram_counts[b]) / total_tokens
    if expected == 0:
        return 0
    score = pair_freq * math.log(pair_freq / expected)
    return score

def learn_wordpiece_vocab(file_path, num_merges=1000, target_vocab_size=1000):
    """
    test.txt 파일을 기반으로 likelihood 기반의 점수를 사용해 vocabulary를 학습합니다.
    
    num_merges: 최대 병합 횟수
    target_vocab_size: 원하는 최종 vocabulary 크기 (특수 토큰 포함)
    """
    corpus = get_corpus(file_path)
    vocab = get_initial_vocab(corpus)
    merges = []

    for i in range(num_merges):
        pair_stats = get_pair_stats(vocab)
        unigram_counts = get_unigram_counts(vocab)
        total_tokens = sum(unigram_counts.values())
        
        # 각 후보 쌍에 대해 likelihood 점수를 계산
        scores = {}
        for pair, freq in pair_stats.items():
            score = compute_likelihood_score(pair, freq, unigram_counts, total_tokens)
            scores[pair] = score
        
        if not scores:
            break
        
        best_pair = max(scores, key=scores.get)
        best_score = scores[best_pair]
        # 점수가 음수이거나 변화가 없으면 종료
        if best_score <= 0:
            break
        
        merges.append(best_pair)
        vocab = merge_vocab(best_pair, vocab)
        
        # 현재 vocabulary 크기 확인 (특수 토큰 제외)
        current_vocab = set()
        for token_seq in vocab.keys():
            tokens = token_seq.split()
            for token in tokens:
                token = token.replace("</w>", "")
                current_vocab.add(token)
        if len(current_vocab) >= target_vocab_size:
            break

    return current_vocab, merges

def save_vocab(vocab_set, output_path="vocab.txt"):
    """
    생성된 vocabulary를 파일로 저장합니다.
    특수 토큰([UNK], [CLS], [SEP])을 우선 추가합니다.
    """
    special_tokens = ["[UNK]", "[CLS]", "[SEP]"]
    with open(output_path, "w", encoding="utf-8") as f:
        for token in special_tokens:
            f.write(token + "\n")
        for token in sorted(vocab_set):
            f.write(token + "\n")

if __name__ == "__main__":
    # test.txt 파일을 기반으로 vocabulary 생성
    file_path = Path(__file__).resolve().parent.parent.parent / "tests" / "tests.txt"
    vocab_set, merges = learn_wordpiece_vocab(str(file_path), num_merges=1000, target_vocab_size=1000)
    save_vocab(vocab_set, output_path="src/word_piece_tokenizer/vocab.txt")
    print("Vocabulary 생성 완료. 총 토큰 수:", len(vocab_set))
    print("병합 기록 (일부):", merges[:10])
