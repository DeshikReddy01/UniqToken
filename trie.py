from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class TrieNode:
    """
    Slots-optimized Node in the Prefix Trie for fast vocabulary matching.
    """

    __slots__ = ("children", "token", "log_p", "token_id", "is_terminal")

    def __init__(
        self,
        children: Optional[Dict[str, TrieNode]] = None,
        token: Optional[str] = None,
        log_p: Optional[float] = None,
        token_id: Optional[int] = None,
        is_terminal: bool = False,
    ) -> None:
        self.children = {} if children is None else children
        self.token = token
        self.log_p = log_p
        self.token_id = token_id
        self.is_terminal = is_terminal


class PrefixTrie:
    """
    Prefix Trie for O(L) single-pass lattice edge mining.

    Eliminates substring slicing and hash table lookups during lattice construction.
    """

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, token: str, log_p: float, token_id: Optional[int] = None) -> None:
        """Inserts a vocabulary token with its log-probability score."""
        node = self.root
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.token = token
        node.log_p = log_p
        node.token_id = token_id
        node.is_terminal = True

    @classmethod
    def from_vocab(cls, vocab: Dict[str, float], token_to_id: Optional[Dict[str, int]] = None) -> PrefixTrie:
        """Constructs a PrefixTrie from a dictionary of token -> log_prob."""
        trie = cls()
        for idx, (token, log_p) in enumerate(vocab.items()):
            t_id = token_to_id.get(token, idx) if token_to_id is not None else idx
            trie.insert(token, log_p, token_id=t_id)
        return trie

    def find_matches(self, text: str, start_idx: int, max_length: int = 16) -> List[Tuple[int, str, float]]:
        """
        Traverses the trie from start_idx in text.
        Returns a list of (end_idx, token_string, log_prob) for all matching prefixes.
        """
        matches: List[Tuple[int, str, float]] = []
        node = self.root
        n = len(text)
        limit = min(start_idx + max_length, n)

        for curr_idx in range(start_idx, limit):
            char = text[curr_idx]
            if char not in node.children:
                break
            node = node.children[char]
            if node.is_terminal and node.token is not None and node.log_p is not None:
                matches.append((curr_idx + 1, node.token, node.log_p))

        return matches
