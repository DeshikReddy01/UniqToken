The Optimal Foundation is Byte-Fallback Unigram Language Model with Regex Pre-splitting and Subword Regularization.

Raw Text Stream
      │
      ▼
┌──────────────────────────────────────────────────────────┐
│  1. Normalizer (NFKC + Metaspace Replacement)            │
│     - Canonicalizes Unicode (NFKC)                       │
│     - Swaps standard spaces ' ' with ' ' (U+2581)       │
└──────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────┐
│  2. Regex Pre-Tokenizer (Atomic Chunk Slicer)             │
│     - Isolates words: [^\W\d_]+                          │
│     - Isolates single digits: \d                         │
│     - Isolates metaspace markers:  +                     │
│     - Isolates punctuation/symbols: [^\w\s]              │
└──────────────────────────────────────────────────────────┘
      │
      ▼
Output: List of atomic chunk strings ready for the subword model




Pre-tokenized Chunks
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  1. Extract Base Alphabet (Must have 100% coverage)      │
│     - Every unique character in the corpus               │
│     - All 256 fallback byte tokens (<0x00> to <0xFF>)    │
│     - Special tokens (<|pad|>, <|unk|>, <|user|>, etc.)  │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  2. Mine Substring n-grams (up to max_len, e.g., 16)     │
│     - Count frequency of all substrings in chunks        │
│     - Filter by min_frequency (e.g., freq >= 3)          │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  3. Seed Vocabulary Pool                                 │
│     - Top K most frequent subwords (e.g., 64,000)        │
│     - Ready for Unigram likelihood optimization          │
└──────────────────────────────────────────────────────────┘