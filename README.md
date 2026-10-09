# word2vec-mac

This is Google's original word2vec C tool (Tomas Mikolov et al., 2013), patched to
build and run on macOS. It also fixes several memory-safety and correctness bugs
and adds documentation.

The original source is at [tmikolov/word2vec](https://github.com/tmikolov/word2vec).
Its README is kept as [`README.txt`](README.txt).

## Documentation

| Document | Contents |
|---|---|
| [BUILD_AND_USAGE.md](BUILD_AND_USAGE.md) | Building, preparing a corpus, every command-line option, using the trained vectors, and the demo scripts |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the five programs fit together and how each one works inside: data structures, training loop, file formats, limits |

## Quick start

```sh
make
curl -L -o text8.zip http://mattmahoney.net/dc/text8.zip && unzip -o text8.zip && rm text8.zip
./word2vec -train text8 -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 \
           -hs 0 -sample 1e-4 -threads 8 -binary 1 -iter 15
./distance vectors.bin            # nearest words, e.g. "king"
./word-analogy vectors.bin        # analogies, e.g. "paris france berlin"
```

The training data (`text8`) and trained models are not stored in the repo. Download
them as shown above.

## What's fixed

### Building and running on macOS

| Problem in the original | Fix |
|---|---|
| `word2vec` and `word2phrase` failed to compile on macOS with `call to undeclared function 'fgetc_unlocked'` | On systems without glibc, the code maps `fgetc_unlocked`/`fputc_unlocked` to the POSIX `getc_unlocked`/`putc_unlocked`. `make` now builds without errors or warnings on macOS and Linux. |
| Every demo script used `wget`, which macOS doesn't ship, and used `gzip` to unpack a `.zip`, which Apple's `gzip` can't do | The scripts download with `curl` and unpack with `unzip`. |

### Crashes and memory-safety bugs

| Problem in the original | Fix |
|---|---|
| File paths longer than 99 characters (59 in `word2phrase`) overflowed a fixed buffer and crashed the program | Paths can be up to 1023 characters. Longer ones are rejected with an error. |
| `distance` and `word-analogy` looped forever at end of input, so they hung when used in a pipe | They exit cleanly at end of input. |
| A query of more than 100 words overflowed a buffer in `distance` and `word-analogy` | Only the first 100 words are used. |
| Vocabulary words of 50+ characters wrote past their slot in all three query tools | Words are truncated inside their slot. |
| Vector files with more than 2000 dimensions overflowed a buffer in the query tools | They are rejected with an error. |
| `compute-accuracy` read tokens with an unbounded `scanf("%s")` | Reads are capped at the buffer size. |
| An output or vocabulary file that couldn't be opened caused a crash | The program reports an error and exits. |

### Correctness bugs

| Problem in the original | Fix |
|---|---|
| `word2phrase` counted bigrams across line breaks, and built the first bigram from an uninitialized buffer | Bigrams are counted only within a line. |
| `word2phrase`'s sort comparator truncated 64-bit count differences to `int`, which can mis-sort the vocabulary on large corpora | It uses the same overflow-safe comparison as `word2vec`. |
| `word2phrase`'s vocabulary pruning freed the wrong entry and could drop a valid word | Pruning removes exactly the words below `-min-count`. |
| On very large corpora, vocabulary pruning could remove the sentence-break token `</s>`, which must stay at index 0 | `</s>` is always kept. |

### Big-model demo script

`demo-train-big-model-v1.sh` didn't work as shipped. It now:

- builds from this repo instead of downloading source from the shut-down `word2vec.googlecode.com`
- keeps its data in `big-model-data/` (the original `word2vec/` folder name clashed with the compiled program)
- saves the UMBC corpus under the filename it later unpacks
- stops at the first failed step instead of training on partial data

## Verification

Every program was built with AddressSanitizer and run through the edge cases above,
with no memory errors reported. The bundled `demo-word-accuracy.sh` ran end to end
on an Apple Silicon Mac: it downloaded `text8`, trained a 200-dimension CBOW model
and scored **53.72 %** on the analogy benchmark, using the 30,000 most frequent words.

## License

Apache License 2.0, as in the original. See [LICENSE](LICENSE).
