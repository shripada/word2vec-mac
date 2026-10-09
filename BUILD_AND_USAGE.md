# word2vec — Build and Usage

See `ARCHITECTURE.md` for how the pieces fit together.

## Quick start

The training data and trained models are not stored in git (see `.gitignore`).
After cloning, download the `text8` corpus (31 MB zip, 100 MB unpacked) into the
repo root:

```sh
make
curl -L -o text8.zip http://mattmahoney.net/dc/text8.zip && unzip -o text8.zip && rm text8.zip
./word2vec -train text8 -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 \
           -hs 0 -sample 1e-4 -threads 8 -binary 1 -iter 15
./distance vectors.bin
```

`./demo-word.sh` does the same steps, and it skips the download when `text8` is
already present.

## 1. Build

You need a C compiler (gcc or clang) and `make`. The code has no other dependencies.

```sh
make            # builds word2vec word2phrase distance word-analogy compute-accuracy
make clean
```

This works on Linux (gcc) and macOS (Apple clang) and should build without warnings.

Optional: `-Ofast` instead of `-O3` can be faster.

## 2. Prepare a corpus

`word2vec` splits tokens on spaces, tabs and newlines and does nothing else.
Normalize the text yourself first. At minimum, lower-case it and strip or separate
punctuation. Put one sentence or document per line, because a newline is treated
as a sentence boundary.

A minimal normalizer, adapted from the demos:

```sh
tr 'A-Z' 'a-z' < raw.txt | sed -e "s/’/'/g" | tr -c "a-z'_ \n" " " > corpus.txt
```

`demo-train-big-model-v1.sh` contains a more thorough `normalize_text()` function,
plus a Wikipedia XML cleaner.

For a quick test corpus, use `text8` (100 MB of cleaned Wikipedia):

```sh
curl -L -o text8.zip http://mattmahoney.net/dc/text8.zip && unzip -o text8.zip && rm text8.zip
```

## 3. (Optional) Detect phrases — `word2phrase`

```sh
./word2phrase -train corpus.txt  -output phrase1.txt -threshold 200 -debug 2
./word2phrase -train phrase1.txt -output phrase2.txt -threshold 100 -debug 2
```

| Option | Default | Meaning |
|---|---|---|
| `-train <file>` | — | Input text |
| `-output <file>` | — | Rewritten text with `a_b` phrase tokens |
| `-min-count <int>` | 5 | Ignore words and bigrams that occur fewer times |
| `-threshold <float>` | 100 | Higher means fewer phrases |
| `-debug <int>` | 2 | Verbosity |

Each pass can only join pairs, so two passes give phrases of up to four words.
Budget about 2 GB of RAM, because the hash table alone takes that much.

## 4. Train vectors — `word2vec`

```sh
./word2vec -train corpus.txt -output vectors.bin -cbow 1 -size 200 -window 8 \
           -negative 25 -hs 0 -sample 1e-4 -threads 8 -binary 1 -iter 15
```

Run `./word2vec` with no arguments to see the full help.

| Option | Default | Meaning |
|---|---|---|
| `-train <file>` | — | Training text |
| `-output <file>` | — | Vectors, or classes when `-classes` is set. If omitted, only vocab work is done. |
| `-size <int>` | 100 | Vector dimensionality |
| `-window <int>` | 5 | Maximum context distance (randomly shrunk per word) |
| `-cbow <int>` | 1 | 1 = CBOW (faster, good for frequent words); 0 = Skip-gram (slower, better for rare words) |
| `-hs <int>` | 0 | 1 = use hierarchical softmax |
| `-negative <int>` | 5 | Number of negative samples (0 = off). Typical values are 5–25. |
| `-sample <float>` | 1e-3 | Down-sampling threshold for frequent words. Typical values are 1e-3 to 1e-5. |
| `-alpha <float>` | 0.05 CBOW / 0.025 SG | Starting learning rate. It decays linearly. |
| `-iter <int>` | 5 | Number of epochs |
| `-min-count <int>` | 5 | Drop rarer words |
| `-threads <int>` | 12 | Worker threads. Set this to your core count. |
| `-binary <int>` | 0 | 1 = binary output, which the query tools **require** |
| `-classes <int>` | 0 | If greater than 0, output k-means cluster ids instead of vectors |
| `-save-vocab <file>` | — | Write a `word count` list |
| `-read-vocab <file>` | — | Reuse a saved vocabulary and skip the counting pass |
| `-debug <int>` | 2 | 2 shows live progress, learning rate and speed |

Some recipes:

```sh
# Skip-gram, text output (easy to load in Python/gensim)
./word2vec -train corpus.txt -output vectors.txt -cbow 0 -size 300 -window 10 -negative 10 -binary 0

# Count the vocabulary once, then reuse it across experiments
./word2vec -train corpus.txt -save-vocab vocab.txt -min-count 5          # no -output → stops after the vocab step
./word2vec -train corpus.txt -read-vocab vocab.txt -output v.bin -binary 1

# 500 word clusters
./word2vec -train corpus.txt -output classes.txt -cbow 1 -size 200 -classes 500
sort -k2 -n classes.txt > classes.sorted.txt
```

Memory is roughly `vocab × size × 4 B` for each of `syn0` and `syn1neg` (and
`syn1` when using `-hs`), plus about 520 MB of fixed tables.

## 5. Use the vectors

All three tools take a **binary** vector file (`-binary 1`).

### Nearest neighbours

```sh
./distance vectors.bin
Enter word or sentence (EXIT to break): france
```

It prints the 40 closest words by cosine similarity. Several words on one line are
summed into a single query.

### Analogies

```sh
./word-analogy vectors.bin
Enter three words (EXIT to break): paris france berlin     # → germany
```

It computes `france - paris + berlin`.

Both tools stop at an `EXIT` line or at end of input, so they work in pipes:

```sh
echo king | ./distance vectors.bin
```

### Accuracy benchmark

```sh
./compute-accuracy vectors.bin 30000 < questions-words.txt     # use only the top-30k words (fast)
./compute-accuracy vectors.bin       < questions-words.txt     # full vocabulary
./compute-accuracy vectors.bin 1000000 < questions-phrases.txt # for phrase models
```

It prints top-1 accuracy per section, total, semantic and syntactic accuracy, and
the share of questions covered by the vocabulary.

### From other languages

The text format is the de facto standard word2vec format. For example, in gensim:
`KeyedVectors.load_word2vec_format("vectors.bin", binary=True)`.

## 6. Demo scripts

Each script runs `make`, downloads data with `curl` if it is missing, trains a
model, and then opens a tool. They need `curl`, `unzip` and `gzip`.

| Script | Data | Does |
|---|---|---|
| `demo-word.sh` | text8 (100 MB) | Trains CBOW-200 and opens `distance` |
| `demo-analogy.sh` | text8 | Trains the same model and opens `word-analogy` |
| `demo-word-accuracy.sh` | text8 | Trains, then runs `compute-accuracy` on the top 30k words |
| `demo-classes.sh` | text8 | Writes 500 k-means classes to `classes.sorted.txt` |
| `demo-phrases.sh` | WMT14 news 2012 (multi-GB) | Normalizes, runs two `word2phrase` passes, trains, opens `distance` |
| `demo-phrase-accuracy.sh` | WMT14 news 2012 | Same pipeline, then `compute-accuracy` on `questions-phrases.txt` |
| `demo-train-big-model-v1.sh` | About 8B words (news, 1B-word benchmark, UMBC, Wikipedia) | Full pipeline, 500-dim vectors, several hours to a day |

The scripts pass `-threads 20`, or 40 for the big model. Lower that to match your
machine.

`demo-train-big-model-v1.sh` builds the tools from this repo and downloads the
corpora into `big-model-data/`. It stops at the first failed step, so a dead
download link can't leave you training on partial data.
