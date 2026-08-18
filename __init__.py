"""
Caliper: Zero-dependency, high-precision Byte-Fallback Unigram and Multimodal Tokenizer.
"""

from __future__ import annotations

from batch_collator import BatchCollator, BatchEncoding
from bpe_model import BPEModel
from bpe_trainer import BPETrainer
from byte_codec import ByteFallbackEngine
from cem_merger import CrossEntropyMerging
from hf_exporter import HuggingFaceExporter
from indentation_compressor import IndentationCompressor
from multimodal.audio_codec import AudioSegment, ResidualVectorQuantizer
from multimodal.image_patcher import DynamicImagePatcher, ImagePatch
from multimodal.multimodal_tokenizer import (
    ImageElement,
    MultimodalSequence,
    MultimodalTokenizer,
)
from multimodal.neural_codecs import (
    HAS_TORCH,
    NeuralAudioCodec,
    NeuralCodecFacade,
    NeuralVisualCodec,
)
from multimodal.visual_codebook import VisualCodebook
from pre_tokenizer import Normalizer, PreToken, RegexPreTokenizer
from security_shield import SecurityShield
from seed_builder import SeedToken, SeedVocabularyBuilder
from streaming_decoder import StreamingDecoder
from tokenizer import CustomTokenizer, Token
from trie import PrefixTrie, TrieNode
from unigram_lattice import LatticeEdge, UnigramLattice
from unigram_trainer import UnigramModel, UnigramTrainer
from vocab_adapter import VocabularyAdapter

__version__ = "1.0.0"
__all__ = [
    # Core Engine
    "CustomTokenizer",
    "Token",
    "Normalizer",
    "RegexPreTokenizer",
    "PreToken",
    "ByteFallbackEngine",
    "UnigramModel",
    "UnigramTrainer",
    "UnigramLattice",
    "LatticeEdge",
    "PrefixTrie",
    "TrieNode",
    "SeedVocabularyBuilder",
    "SeedToken",
    # Post-Training Optimization
    "CrossEntropyMerging",
    # BPE Engine
    "BPEModel",
    "BPETrainer",
    # Serving & Security
    "SecurityShield",
    "StreamingDecoder",
    "IndentationCompressor",
    "VocabularyAdapter",
    "BatchCollator",
    "BatchEncoding",
    "HuggingFaceExporter",
    # Multimodal Engine
    "MultimodalTokenizer",
    "MultimodalSequence",
    "ImageElement",
    "DynamicImagePatcher",
    "ImagePatch",
    "VisualCodebook",
    "ResidualVectorQuantizer",
    "AudioSegment",
    "NeuralCodecFacade",
    "NeuralVisualCodec",
    "NeuralAudioCodec",
    "HAS_TORCH",
]
