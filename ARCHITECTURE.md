# word2vec — Architecture

This repo is Google's original 2013 C implementation of word2vec. It contains five
independent command-line programs, a few shell demos, and two evaluation datasets.
The programs share no headers or libraries. Each `.c` file is a single
self-contained program that keeps its state in globals. The programs talk to each
other only through files on disk.

## Pipeline

```
 raw text ──(normalize: sed/tr/awk in demo scripts)──► plain text, space-separated tokens, one sentence per line
                                                            │
                                         (optional, run 1–2×)│
                                                            ▼
                                                    ┌──────────────┐
                                                    │ word2phrase  │  joins frequent bigrams: "new york" → "new_york"
                                                    └──────┬───────┘
                                                           ▼
                                                    ┌──────────────┐  -save-vocab ──► vocab.txt
                                                    │   word2vec   │◄─ -read-vocab ── vocab.txt
                                                    └──────┬───────┘
                         ┌─────────────────────────────────┼──────────────────────────┐
                         ▼ (-binary 1)                      ▼ (-binary 0)               ▼ (-classes K)
                    vectors.bin                        vectors.txt                 classes.txt  (word → cluster id)
          ┌──────────────┼──────────────────┐
          ▼              ▼                  ▼
     distance       word-analogy     compute-accuracy ◄── questions-words.txt / questions-phrases.txt
  (nearest words)   (a:b :: c:?)      (analogy benchmark)
```

## Files

| File | Role |
|---|---|
| `word2vec.c` | Trainer. Builds the vocabulary, trains CBOW or Skip-gram with hierarchical softmax and/or negative sampling, and writes vectors or k-means word classes. |
| `word2phrase.c` | Phrase detector. Rewrites a corpus so that strongly associated bigrams become single `a_b` tokens. |
| `distance.c` | Interactive nearest-neighbour search by cosine similarity. |
| `word-analogy.c` | Interactive analogy solver (`b - a + c`). |
| `compute-accuracy.c` | Batch analogy evaluation against a question file. |
| `questions-words.txt` | About 19.5k word analogies in 14 sections (5 semantic, then 9 syntactic). |
| `questions-phrases.txt` | Phrase analogies in 5 sections, such as `Boston Boston_Globe ...`. |
| `demo-*.sh` | End-to-end example pipelines (see `BUILD_AND_USAGE.md`). |
| `makefile` | Builds all five binaries with `gcc -O3 -march=native -pthread -lm`. |

---

## `word2vec.c`: the trainer

### Data structures

- **`struct vocab_word`**: `cn` (count), `word`, and for hierarchical softmax `code`
  (Huffman bits), `codelen`, and `point` (indices of inner tree nodes on the path from the root).
- **`vocab[]`**: a dynamic array that is sorted by descending frequency after counting.
  Index **0 is always `</s>`**, the token that stands for a newline (sentence boundary).
- **`vocab_hash[]`**: an open-addressing hash table with linear probing. It holds 30M
  slots (`int`) and maps a word to its `vocab` index. The hash is `h = h*257 + c`.
- **Weight matrices** (row-major, `vocab_size × layer1_size`, `float`, 128-byte aligned):
  - `syn0` holds the input/word vectors. **This is what gets written out.** It is
    initialized uniformly at random in `[-0.5, 0.5] / size`.
  - `syn1` holds the output vectors for the inner nodes of the Huffman tree. It is
    used only with `-hs 1` and is zero-initialized.
  - `syn1neg` holds the output vectors for negative sampling. It is used only when
    `-negative > 0` and is zero-initialized.
- **`expTable`**: a precomputed sigmoid with 1000 entries covering `[-6, 6]`
  (`MAX_EXP`). Dot products outside that range are clipped or skipped.
- **`table`**: the unigram table for negative sampling. It has 1e8 `int` entries
  (about 400 MB), and each word appears in proportion to `count^0.75`.

### Control flow

```
main → parse args (ArgPos) → allocate vocab/hash/expTable → TrainModel
TrainModel
  ├─ ReadVocab()               if -read-vocab   (counts come from the file)
  │  └─ or LearnVocabFromTrainFile()             (one full pass over the corpus)
  │        ReadWord → SearchVocab / AddWordToVocab
  │        ReduceVocab() when the hash is >70% full (drops words with count ≤ min_reduce, then raises min_reduce)
  │        SortVocab()  sorts by count, drops count < min_count, rebuilds the hash
  ├─ SaveVocab()               if -save-vocab
  ├─ return early              if no -output was given (vocab-only mode)
  ├─ InitNet()                 allocates and initializes syn0/syn1/syn1neg, then CreateBinaryTree() (Huffman)
  ├─ InitUnigramTable()        if negative > 0
  ├─ spawn num_threads × TrainModelThread, then join
  └─ write output: vectors (text or binary), or run k-means and write classes
```

### Training loop (`TrainModelThread`)

- **Parallelism.** Each thread `fseek`s to byte offset `file_size / threads * id`
  and processes about `train_words / threads` words per epoch. Threads update the
  shared `syn0`/`syn1`/`syn1neg` and the global `alpha` **without locks**
  (Hogwild-style). This is deliberate: collisions are rare and harmless.
- **Sentences.** The thread reads up to `MAX_SENTENCE_LENGTH` (1000) words, or up to
  `</s>`. Unknown words are skipped.
- **Subsampling.** A word with count `f` is kept with probability
  `(sqrt(f/(s·T)) + 1) · (s·T)/f`, where `s = -sample` and `T = train_words`.
- **Dynamic window.** For each position the thread draws `b ∈ [0, window)` at random,
  so the effective window is `window - b`. This weights near context more heavily.
- **CBOW** (`-cbow 1`) averages the context vectors into `neu1`, predicts the centre
  word, then adds the accumulated gradient `neu1e` to every context word's `syn0` row.
- **Skip-gram** (`-cbow 0`) treats each context word on its own as the input
  (`syn0[l1]`) and predicts the centre word.
- **Output layers**, which can be combined:
  - *Hierarchical softmax* runs a binary logistic regression at each node on the
    centre word's Huffman path: `g = (1 - code[d] - σ(x)) · α`.
  - *Negative sampling* uses one positive (label 1) and `negative` samples drawn
    from `table` (label 0): `g = (label - σ(x)) · α`.
- **Learning rate.** It decays linearly, `α = α₀ · (1 - processed / (iter·T + 1))`,
  with a floor of `α₀ · 1e-4`. It is updated every 10k words, and progress is
  printed at the same time.
- **Epochs.** At EOF, or once the thread has processed its share of words, it
  decrements `local_iter` and seeks back to its start offset.

### Output formats

- **Text** (`-binary 0`): the header line `<vocab_size> <dim>`, followed by one line
  per word, `word v1 v2 ... vD `.
- **Binary** (`-binary 1`): the same header, then for each word `word<space>`
  followed by `D` raw native-endian `float32` values and `\n`.
- **Classes** (`-classes K`): one line per word, `word <cluster_id>`. The clusters come
  from 10 iterations of spherical k-means over `syn0`, with centroids
  L2-normalized and dot-product assignment.

Words are written in descending frequency order, so taking the first N lines of
the output gives you the N most frequent words. `compute-accuracy` depends on this
ordering.

---

## `word2phrase.c`: the phrase detector

1. **Counting pass.** It reads the corpus and counts every unigram and every adjacent
   bigram, stored as the token `a_b`, in a single vocabulary. Bigrams are never
   counted across a newline. `ReduceVocab`/`SortVocab` work the same way as in
   word2vec. The first word of each line starts no bigram. **The hash table has
  500M slots (about 2 GB `calloc`).**
2. **Rewrite pass.** It re-reads the corpus and, for each adjacent pair `(a, b)`,
   computes

   ```
   score = (count(a_b) - min_count) / (count(a) · count(b)) · train_words
   ```

   If `score > threshold` it writes `_` instead of a space before `b`, so the pair
   becomes `a_b`. After a join, `b` cannot start a new phrase (`pb = 0`), so each
   pass produces at most bigrams. Newlines are preserved.

Running it twice, once with a high threshold and once with a lower one, lets
phrases of up to four tokens form, such as `new_york_times`. The demos use 200 and
then 100.

---

## The query tools: `distance.c`, `word-analogy.c`, `compute-accuracy.c`

The three tools share the same loader, copied into each file:

- They read the **binary** vector format only, with vectors of up to 2000 dimensions.
- They L2-normalize every vector on load, which turns a dot product into cosine
  similarity.
- They store words in fixed 50-byte slots (`max_w`), so longer words are truncated.
- They use brute-force linear scans: there is no index, and memory is
  `vocab × dim × 4` bytes.

| Tool | Query | Computation | Output |
|---|---|---|---|
| `distance` | One or more words on a line | Sum of the word vectors, normalized; cosine against every word, excluding the inputs | Top 40 |
| `word-analogy` | Exactly 3 words `a b c` | `vec = b - a + c`, normalized; cosine against every word, excluding a, b, c | Top 40 |
| `compute-accuracy` | stdin: `: section` headers, then `a b c d` lines | Same as the analogy, but top 1 only; correct when the result equals `d` | Accuracy per section and overall |

`compute-accuracy` details:

- It upper-cases everything, so comparisons ignore case.
- An optional second argument `threshold` loads only the **first N (most frequent)**
  words. Questions that use words outside those N are skipped and counted in
  "Questions seen / total".
- It counts sections 1–5 as *semantic* and the rest as *syntactic*. That split
  only makes sense for `questions-words.txt`.

---

## Limits and quirks

| Item | Limit or behaviour |
|---|---|
| File path arguments | Up to 1023 characters. Longer paths are rejected with an error. |
| Token length | 99 bytes in word2vec, 59 in word2phrase, 50 in the query tools. Longer tokens are truncated. |
| Vocabulary | At most about 21M words in word2vec (70% of the 30M hash slots) before pruning starts. |
| Tokenization | Splits on space, tab and newline only. `\r` is ignored. No lower-casing or punctuation handling is done, so the corpus must be pre-normalized (see the demo scripts). |
| Query tool stdin | `distance` and `word-analogy` stop at an `EXIT` line or at end of input. A query line uses at most the first 100 words. |
| Out-of-vocabulary handling | `word-analogy` uses index 0 (`</s>`) as its "not found" sentinel. |
| Portability | On libcs without `fgetc_unlocked`/`fputc_unlocked` (anything that isn't glibc, such as macOS), the code maps them to the POSIX `getc_unlocked`/`putc_unlocked`. |
