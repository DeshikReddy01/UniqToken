from __future__ import annotations

import math
from collections import Counter
from typing import List, Set

from seed_builder import SeedVocabularyBuilder
from tokenizer import CustomTokenizer
from unigram_trainer import UnigramModel


class VocabularyAdapter:
    """
    Dynamic Online Vocabulary Adapter (Solves Problem 7: Outdated Vocabulary).

    Expands an existing trained tokenizer's vocabulary with new domain tokens
    (e.g., new tech slang, biomedical terms, code symbols) WITHOUT modifying
    existing token IDs, preserving downstream neural network embedding weights.
    """

    # New tokens get at most 10% of the probability share implied by their
    # domain frequency, biasing segmentation toward already-trained subwords.
    NEW_TOKEN_PROB_SCALE = 0.1

    @staticmethod
    def expand_vocabulary(
        tokenizer: CustomTokenizer,
        new_domain_corpus: List[str],
        num_new_tokens: int = 50,
        min_frequency: int = 2,
        max_ngram_length: int = 16,
        verbose: bool = True,
    ) -> CustomTokenizer:
        """
        Extends the tokenizer with new high-frequency domain subwords.
        """
        if num_new_tokens < 0:
            raise ValueError("num_new_tokens must not be negative")
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least one")
        if max_ngram_length < 1:
            raise ValueError("max_ngram_length must be at least one")
        if num_new_tokens == 0:
            return tokenizer

        normalizer = tokenizer.normalizer
        pre_tokenizer = tokenizer.pre_tokenizer
        old_model = tokenizer.model

        # 1. Pre-tokenize new corpus
        new_chunks: List[str] = []
        for doc in new_domain_corpus:
            norm = normalizer.normalize(doc)
            new_chunks.extend(pre_tokenizer.pre_tokenize(norm))

        chunk_counts = Counter(new_chunks)

        # 2. Mine n-grams from new corpus
        seed_builder = SeedVocabularyBuilder(
            target_vocab_size=num_new_tokens * 2,
            max_ngram_length=max_ngram_length,
            min_frequency=min_frequency,
        )

        existing_tokens: Set[str] = set(old_model.vocab.keys())
        raw_ngrams = seed_builder.mine_ngrams(chunk_counts)
        filtered_candidates = seed_builder.filter_candidates(raw_ngrams, existing_tokens)
        ranked_candidates = seed_builder.rank_candidates(filtered_candidates)

        # 3. Select top new candidates
        new_tokens_to_add = ranked_candidates[:num_new_tokens]

        if verbose:
            print(
                f"[Vocab Adapter] Mined {len(ranked_candidates)} new candidates. Adding top {len(new_tokens_to_add)} tokens."
            )

        if not new_tokens_to_add:
            return tokenizer

        # 4. Assign IDs above the current maximum (existing IDs may be sparse).
        start_id = max(old_model.id_to_token, default=-1) + 1
        new_token_to_id = dict(old_model.token_to_id)
        new_id_to_token = dict(old_model.id_to_token)

        # Compute prior probabilities for new tokens based on occurrence in new corpus
        total_new_freq = sum(filtered_candidates[t] for t in new_tokens_to_add)
        min_existing_prob = max(min((math.exp(lp) for lp in old_model.vocab.values()), default=1e-12), 1e-12)
        new_vocab_probs = {tok: max(math.exp(lp), 1e-12) for tok, lp in old_model.vocab.items()}

        for idx, tok in enumerate(new_tokens_to_add):
            assigned_id = start_id + idx
            new_token_to_id[tok] = assigned_id
            new_id_to_token[assigned_id] = tok
            # Give new token a proportional probability
            token_prob = max(
                filtered_candidates[tok] / max(total_new_freq, 1) * VocabularyAdapter.NEW_TOKEN_PROB_SCALE,
                min_existing_prob,
            )
            new_vocab_probs[tok] = token_prob

        # 5. Re-normalize probability distribution
        total_p = sum(new_vocab_probs.values())
        updated_vocab = {tok: math.log(p / total_p) for tok, p in new_vocab_probs.items()}

        updated_model = UnigramModel(
            vocab=updated_vocab,
            token_to_id=new_token_to_id,
            id_to_token=new_id_to_token,
            special_tokens=list(old_model.special_tokens),
            max_subword_len=max(old_model.max_subword_len, max(map(len, new_tokens_to_add))),
            byte_fallback=old_model.byte_fallback,
            unk_token=old_model.unk_token,
        )

        return CustomTokenizer(
            normalizer=normalizer,
            pre_tokenizer=pre_tokenizer,
            model=updated_model,
        )
