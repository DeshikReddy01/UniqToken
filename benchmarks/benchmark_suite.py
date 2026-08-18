from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))

from bpe_trainer import BPETrainer
from cem_merger import CrossEntropyMerging
from hf_exporter import HuggingFaceExporter
from tokenizer import CustomTokenizer
from unigram_trainer import UnigramTrainer


@dataclass
class BenchmarkMetrics:
    dataset_name: str
    num_chars: int
    num_bytes: int
    num_words: int
    num_tokens: int
    bytes_per_token: float
    tokens_per_word: float  # Fertility
    encode_speed_kbs: float
    encode_speed_tokens_sec: float
    decode_speed_kbs: float
    offset_overhead_ratio: float  # (time_with_offsets / time_without_offsets)


class TokenizerBenchmarkSuite:
    """
    Empirical Benchmarking & Performance Evaluation Suite.

    Evaluates:
    1. Compression Ratio (Bytes / Token) across multilingual scripts.
    2. Morphological Fertility (Tokens / Word).
    3. Throughput (KB/sec & Tokens/sec) on real corpora.
    4. Offset Span Computation Overhead.
    5. Code Indentation Context Compression Savings.
    6. Unigram vs. BPE Head-to-Head Architectural Comparison.
    7. Caliper vs. HuggingFace, SentencePiece, and tiktoken baselines.
    """

    BENCHMARK_CORPORA = {
        "English_Prose": (
            "The architecture of transformer language models relies fundamentally on discrete "
            "tokenization subword vocabularies. Subword tokenization balances the trade-off "
            "between character-level sequence bloat and word-level vocabulary explosion. "
            "Modern systems require robust handling of diverse orthographic conventions, "
            "case normalization, and punctuation isolation."
        )
        * 40,
        "Python_Code": (
            "class DistributedOptimizer:\n"
            "    def __init__(self, params, lr: float = 1e-4):\n"
            "        self.params = list(params)\n"
            "        self.lr = lr\n"
            "        self.state = {}\n\n"
            "    def step(self, closure=None):\n"
            "        loss = None\n"
            "        if closure is not None:\n"
            "            loss = closure()\n"
            "        for p in self.params:\n"
            "            if p.grad is not None:\n"
            "                d_p = p.grad.data\n"
            "                p.data.add_(d_p, alpha=-self.lr)\n"
            "        return loss\n"
        )
        * 30,
        "Indic_Hindi": (
            "प्राकृतिक भाषा प्रसंस्करण और कंप्यूटर विज्ञान में टोकनाइज़र एक अत्यंत महत्वपूर्ण घटक है। "
            "देवनागरी लिपि में मात्राओं और हलंत (विराम) का उचित संयोजन बनाए रखना आवश्यक है ताकि "
            "अक्षरों का विखंडन न हो। भाषा मॉडल की सटीकता सही टोकनीकरण पर निर्भर करती है।"
        )
        * 30,
        "CJK_Japanese": (
            "自然言語処理におけるトークナイザーは、テキストを一連のサブワードに分割する重要な役割を果たします。"
            "日本語のように単語間に空白が存在しない言語では、形態素解析やバイトフォールバック機構が極めて重要です。"
            "正確なアライメントとオフセット追跡が必要です。"
        )
        * 30,
        "Arabic_Script": (
            "تعتبر معالجة اللغات الطبيعية وتجزئة النصوص من أهم ركائز الذكاء الاصطناعي الحديث. "
            "يتطلب التعامل مع اللغة العربية دعماً دقيقاً للحركات وعلامات التشكيل لضمان عدم فقدان المعنى."
        )
        * 30,
        "Arithmetic_Math": (
            "Solve the system of equations: f(x, y) = 3.14159 * x^2 + 2.71828 * y - 42.0. "
            "Given matrices A = [[12, 34], [56, 78]] and B = [[90, 11], [22, 33]], calculate det(A * B). "
            "Indices: 1048576, 2097152, 4194304, 8388608. Verify sum(x_i) for i in range(1000)."
        )
        * 30,
    }

    def __init__(self, tokenizer: Optional[CustomTokenizer] = None):
        if tokenizer is None:
            training_corpus = list(self.BENCHMARK_CORPORA.values())
            self.tokenizer = CustomTokenizer.train_from_corpus(
                corpus=training_corpus,
                target_vocab_size=1000,
                max_ngram_length=12,
                min_frequency=2,
                byte_fallback=True,
                split_digits=True,
                verbose=False,
            )
        else:
            self.tokenizer = tokenizer

    def evaluate_dataset(
        self, name: str, text: str, warmup: int = 2, iterations: int = 5
    ) -> BenchmarkMetrics:
        raw_bytes = text.encode("utf-8")
        num_bytes = len(raw_bytes)
        num_chars = len(text)
        num_words = max(len(text.split()), 1)

        # Warmup
        for _ in range(warmup):
            _ = self.tokenizer.encode_to_ids(text)

        # 1. Encode Speed
        t0 = time.perf_counter()
        token_ids: List[int] = []
        for _ in range(iterations):
            token_ids = self.tokenizer.encode_to_ids(text)
        t_encode = (time.perf_counter() - t0) / iterations

        num_tokens = len(token_ids)
        encode_kbs = (num_bytes / 1024.0) / max(t_encode, 1e-6)
        encode_tokens_sec = num_tokens / max(t_encode, 1e-6)

        # 2. Decode Speed
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = self.tokenizer.decode(token_ids)
        t_decode = (time.perf_counter() - t0) / iterations
        decode_kbs = (num_bytes / 1024.0) / max(t_decode, 1e-6)

        # 3. Offset Mapping Overhead
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = self.tokenizer.encode_with_offsets(text)
        t_offsets = (time.perf_counter() - t0) / iterations
        offset_overhead = t_offsets / max(t_encode, 1e-6)

        bytes_per_token = num_bytes / max(num_tokens, 1)
        tokens_per_word = num_tokens / max(num_words, 1)

        return BenchmarkMetrics(
            dataset_name=name,
            num_chars=num_chars,
            num_bytes=num_bytes,
            num_words=num_words,
            num_tokens=num_tokens,
            bytes_per_token=round(bytes_per_token, 3),
            tokens_per_word=round(tokens_per_word, 3),
            encode_speed_kbs=round(encode_kbs, 2),
            encode_speed_tokens_sec=round(encode_tokens_sec, 1),
            decode_speed_kbs=round(decode_kbs, 2),
            offset_overhead_ratio=round(offset_overhead, 2),
        )

    def run_all_benchmarks(self) -> List[BenchmarkMetrics]:
        results: List[BenchmarkMetrics] = []
        for name, text in self.BENCHMARK_CORPORA.items():
            metrics = self.evaluate_dataset(name, text)
            results.append(metrics)
        return results

    def evaluate_payload_sizes(self) -> List[BenchmarkMetrics]:
        """Measure 1 MiB and 10 MiB payloads without making CI runs expensive."""
        seed = self.BENCHMARK_CORPORA["English_Prose"]
        seed_bytes = len(seed.encode("utf-8"))
        results: List[BenchmarkMetrics] = []
        for size_mib in (1, 10):
            target_bytes = size_mib * 1024 * 1024
            payload = seed * ((target_bytes + seed_bytes - 1) // seed_bytes)
            results.append(
                self.evaluate_dataset(
                    f"English_{size_mib}MiB",
                    payload,
                    warmup=1,
                    iterations=1,
                )
            )
        return results

    def evaluate_indentation_compression(self) -> Dict[str, Any]:
        """
        Measures context token savings with indentation compression enabled during training & inference.
        """
        code_corpus = self.BENCHMARK_CORPORA["Python_Code"]

        # Use identical training data and vocabulary budgets so the comparison
        # measures indentation compression rather than unrelated model quality.
        plain_tok = CustomTokenizer.train_from_corpus(
            corpus=[code_corpus],
            target_vocab_size=500,
            compress_indents=False,
            verbose=False,
        )

        # Train code model with indentation compression enabled.
        code_tok = CustomTokenizer.train_from_corpus(
            corpus=[code_corpus],
            target_vocab_size=500,
            compress_indents=True,
            verbose=False,
        )

        # Plain tokenization (spaces tokenized individually)
        plain_ids = plain_tok.encode_to_ids(code_corpus)

        # CustomTokenizer compresses indentation during normal encoding.
        compressed_ids = code_tok.encode_to_ids(code_corpus)

        token_delta = len(plain_ids) - len(compressed_ids)
        token_reduction = token_delta / len(plain_ids) * 100.0

        return {
            "plain_token_count": len(plain_ids),
            "compressed_token_count": len(compressed_ids),
            "token_delta": token_delta,
            "reduction_percentage": round(token_reduction, 2),
        }

    def evaluate_unigram_vs_bpe(self) -> Dict[str, Any]:
        text = (
            self.BENCHMARK_CORPORA["English_Prose"]
            + self.BENCHMARK_CORPORA["Python_Code"]
        )
        chunks = self.tokenizer.pre_tokenizer.pre_tokenize(
            self.tokenizer.normalizer.normalize(text)
        )

        # Train BPE
        bpe_trainer = BPETrainer(num_merges=100, byte_fallback=True)
        bpe_model = bpe_trainer.train(chunks)

        # Unigram stats
        t0 = time.perf_counter()
        unigram_tokens = self.tokenizer.encode(text)
        t_unigram = time.perf_counter() - t0

        # BPE stats
        t0 = time.perf_counter()
        bpe_tokens: List[str] = []
        for c in chunks:
            bpe_tokens.extend(bpe_model.encode(c))
        t_bpe = time.perf_counter() - t0

        return {
            "unigram_token_count": len(unigram_tokens),
            "bpe_token_count": len(bpe_tokens),
            "unigram_bytes_per_token": round(
                len(text.encode("utf-8")) / len(unigram_tokens), 3
            ),
            "bpe_bytes_per_token": round(
                len(text.encode("utf-8")) / len(bpe_tokens), 3
            ),
            "unigram_encode_sec": round(t_unigram, 4),
            "bpe_encode_sec": round(t_bpe, 4),
        }

    def evaluate_cem(self) -> Dict[str, Any]:
        """
        Cross-Entropy Merging post-training compression gain in the fixed-budget
        scenario. Uses a controlled repetitive-morphology corpus (the embedded
        English/Code corpora are far too small to stress a vocab budget): a
        Unigram model trained at a tight vocab limit leaves common words split
        into subwords; CEM then recovers them as single tokens without
        retraining, cutting the token count.
        """
        base_words = [
            "the quick brown fox jumps over the lazy dog",
            "the quick fox and the lazy dog",
            "jumping foxes are quick and brown",
            "brown dogs are quick",
        ]
        text = " ".join(base_words) * 30
        chunks = self.tokenizer.pre_tokenizer.pre_tokenize(
            self.tokenizer.normalizer.normalize(text)
        )

        constrained = UnigramTrainer(
            target_vocab_size=300,
            max_ngram_length=6,
            min_frequency=2,
            byte_fallback=True,
        ).train(chunks, verbose=False)

        before = sum(len(constrained.encode(c)) for c in chunks)
        cem = CrossEntropyMerging(max_merges=200)
        improved = cem.optimize(constrained, chunks)
        after = sum(len(improved.encode(c)) for c in chunks)

        return {
            "cem_merges_applied": len(cem.merges),
            "vocab_before": len(constrained.vocab),
            "vocab_after": len(improved.vocab),
            "token_count_before": before,
            "token_count_after": after,
            "token_reduction_pct": round((before - after) / max(before, 1) * 100.0, 2),
            "bytes_per_token_before": round(
                len(text.encode("utf-8")) / max(before, 1), 3
            ),
            "bytes_per_token_after": round(
                len(text.encode("utf-8")) / max(after, 1), 3
            ),
        }

    def evaluate_superbpe(self) -> Dict[str, Any]:
        """
        SuperBPE ('space travel') post-training compression gain. Merges tokens
        that span word boundaries (e.g. ``the\u2581quick``) using the same
        fixed-budget stress corpus as the CEM metric, measured end-to-end
        through the tokenizer pipeline.
        """
        base_words = [
            "the quick brown fox jumps over the lazy dog",
            "the quick fox and the lazy dog",
            "jumping foxes are quick and brown",
            "brown dogs are quick",
        ]
        text = " ".join(base_words) * 30
        chunks = self.tokenizer.pre_tokenizer.pre_tokenize(
            self.tokenizer.normalizer.normalize(text)
        )

        constrained = UnigramTrainer(
            target_vocab_size=300,
            max_ngram_length=6,
            min_frequency=2,
            byte_fallback=True,
        ).train(chunks, verbose=False)

        base_tok = CustomTokenizer(
            self.tokenizer.normalizer, self.tokenizer.pre_tokenizer, constrained
        )
        before = len(base_tok.encode(text))

        superbpe = CrossEntropyMerging(max_merges=100, cross_word=True)
        improved = superbpe.optimize(constrained, chunks)
        improved_tok = CustomTokenizer(
            self.tokenizer.normalizer, self.tokenizer.pre_tokenizer, improved
        )
        after = len(improved_tok.encode(text))

        return {
            "superbpe_merges_applied": len(superbpe.merges),
            "vocab_before": len(constrained.vocab),
            "vocab_after": len(improved.vocab),
            "token_count_before": before,
            "token_count_after": after,
            "token_reduction_pct": round((before - after) / max(before, 1) * 100.0, 2),
            "bytes_per_token_before": round(
                len(text.encode("utf-8")) / max(before, 1), 3
            ),
            "bytes_per_token_after": round(
                len(text.encode("utf-8")) / max(after, 1), 3
            ),
        }

    def evaluate_external_baselines(self) -> Dict[str, Any]:
        text = self.BENCHMARK_CORPORA["English_Prose"]
        results: Dict[str, Any] = {}

        # 1. Caliper (Pure Python + Trie)
        t0 = time.perf_counter()
        for _ in range(5):
            caliper_tokens = self.tokenizer.encode_to_ids(text)
        t_caliper = (time.perf_counter() - t0) / 5.0
        results["Caliper (Python+Trie)"] = {
            "tokens": len(caliper_tokens),
            "time_sec": round(t_caliper, 4),
            "tokens_sec": round(len(caliper_tokens) / max(t_caliper, 1e-6), 1),
        }

        # 2. HuggingFace Tokenizers (Rust C-FFI) via our HF Exporter.
        try:
            from tokenizers import Tokenizer
            import json

            with TemporaryDirectory() as tmp_dir:
                hf_json = HuggingFaceExporter.export_to_hf_dict(self.tokenizer)
                path = Path(tmp_dir) / "hf_tok.json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(hf_json, f)

                hf_tok = Tokenizer.from_file(str(path))
                t0 = time.perf_counter()
                for _ in range(5):
                    hf_encoded = hf_tok.encode(text)
                t_hf = (time.perf_counter() - t0) / 5.0

                results["HuggingFace (Rust)"] = {
                    "tokens": len(hf_encoded.ids),
                    "time_sec": round(t_hf, 5),
                    "tokens_sec": round(len(hf_encoded.ids) / max(t_hf, 1e-6), 1),
                }
        except Exception as e:
            results["HuggingFace (Rust)"] = {"error": str(e)}

        # 3. SentencePiece trained on the same benchmark corpus.
        try:
            import sentencepiece as spm

            with TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                corpus_path = tmp_path / "corpus.txt"
                training_lines = [
                    corpus[start : start + 2048]
                    for corpus in self.BENCHMARK_CORPORA.values()
                    for start in range(0, len(corpus), 2048)
                ]
                corpus_path.write_text("\n".join(training_lines), encoding="utf-8")
                model_prefix = tmp_path / "sentencepiece"
                spm.SentencePieceTrainer.train(
                    input=str(corpus_path),
                    model_prefix=str(model_prefix),
                    model_type="unigram",
                    vocab_size=1000,
                    character_coverage=1.0,
                    byte_fallback=True,
                    hard_vocab_limit=False,
                    bos_id=-1,
                    eos_id=-1,
                    pad_id=-1,
                    minloglevel=2,
                )
                sp_processor = spm.SentencePieceProcessor(
                    model_file=str(model_prefix) + ".model"
                )
                t0 = time.perf_counter()
                for _ in range(5):
                    sp_ids = sp_processor.encode(text, out_type=int)
                t_sentencepiece = (time.perf_counter() - t0) / 5.0
                results["SentencePiece (Unigram)"] = {
                    "tokens": len(sp_ids),
                    "time_sec": round(t_sentencepiece, 5),
                    "tokens_sec": round(len(sp_ids) / max(t_sentencepiece, 1e-6), 1),
                }
        except Exception as e:
            results["SentencePiece (Unigram)"] = {"error": str(e)}

        # 4. tiktoken uses a fixed pre-trained vocabulary, so its compression
        # numbers are informational rather than a same-corpus comparison.
        try:
            import tiktoken

            tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
            t0 = time.perf_counter()
            for _ in range(5):
                tiktoken_ids = tiktoken_encoder.encode(text)
            t_tiktoken = (time.perf_counter() - t0) / 5.0
            results["tiktoken (cl100k_base)"] = {
                "tokens": len(tiktoken_ids),
                "time_sec": round(t_tiktoken, 5),
                "tokens_sec": round(len(tiktoken_ids) / max(t_tiktoken, 1e-6), 1),
            }
        except Exception as e:
            results["tiktoken (cl100k_base)"] = {"error": str(e)}

        return results

    def print_summary_report(self, include_large_payloads: bool = False) -> None:
        print("=" * 85)
        print("CALIPER TOKENIZER EMPIRICAL BENCHMARK REPORT")
        print("=" * 85)

        results = self.run_all_benchmarks()

        header = f"{'Dataset':<18} | {'Bytes':<7} | {'Tokens':<7} | {'Bytes/Tok':<10} | {'Fertility':<10} | {'Enc KB/s':<10} | {'Tok/sec':<10} | {'Offset Overhead':<15}"
        print(header)
        print("-" * len(header))

        for r in results:
            print(
                f"{r.dataset_name:<18} | {r.num_bytes:<7} | {r.num_tokens:<7} | "
                f"{r.bytes_per_token:<10} | {r.tokens_per_word:<10} | "
                f"{r.encode_speed_kbs:<10} | {r.encode_speed_tokens_sec:<10} | "
                f"{r.offset_overhead_ratio:<15}x"
            )

        if include_large_payloads:
            print("\n" + "=" * 85)
            print("LARGE-PAYLOAD THROUGHPUT")
            print("=" * 85)
            for r in self.evaluate_payload_sizes():
                print(
                    f"  {r.dataset_name:<16} | {r.num_bytes / (1024 * 1024):>5.2f} MiB | "
                    f"{r.num_tokens} tokens | {r.encode_speed_kbs} KB/s | "
                    f"{r.encode_speed_tokens_sec} tok/s"
                )

        print("\n" + "=" * 85)
        print("CODE INDENTATION COMPRESSION CONTEXT SAVINGS")
        print("=" * 85)
        indent_stats = self.evaluate_indentation_compression()
        print(f"  Plain Code Tokens       : {indent_stats['plain_token_count']}")
        print(f"  Compressed Code Tokens  : {indent_stats['compressed_token_count']}")
        if indent_stats["token_delta"] >= 0:
            print(
                f"  Context Capacity Saved  : {indent_stats['token_delta']} tokens "
                f"({indent_stats['reduction_percentage']}% context reduction)"
            )
        else:
            print(
                f"  Token Increase          : {-indent_stats['token_delta']} tokens "
                f"({-indent_stats['reduction_percentage']}% regression)"
            )

        print("\n" + "=" * 85)
        print("HEAD-TO-HEAD: UNIGRAM VS. BPE ON IDENTICAL DATA")
        print("=" * 85)
        cmp_stats = self.evaluate_unigram_vs_bpe()
        print(
            f"  Unigram Token Count     : {cmp_stats['unigram_token_count']} tokens ({cmp_stats['unigram_bytes_per_token']} bytes/tok)"
        )
        print(
            f"  BPE Token Count         : {cmp_stats['bpe_token_count']} tokens ({cmp_stats['bpe_bytes_per_token']} bytes/tok)"
        )
        print(f"  Unigram Time (Trie DAG) : {cmp_stats['unigram_encode_sec']}s")
        print(f"  BPE Time (Rank Merges)  : {cmp_stats['bpe_encode_sec']}s")

        print("\n" + "=" * 85)
        print("POST-TRAINING CROSS-ENTROPY MERGING (CEM)")
        print("=" * 85)
        cem_stats = self.evaluate_cem()
        print(
            f"  Vocab                   : {cem_stats['vocab_before']} -> {cem_stats['vocab_after']} tokens "
            f"({cem_stats['cem_merges_applied']} merges)"
        )
        print(
            f"  Token Count             : {cem_stats['token_count_before']} -> {cem_stats['token_count_after']} "
            f"({cem_stats['token_reduction_pct']}% fewer)"
        )
        print(
            f"  Compression             : {cem_stats['bytes_per_token_before']} -> {cem_stats['bytes_per_token_after']} bytes/tok"
        )

        print("\n" + "=" * 85)
        print("POST-TRAINING SUPERBPE (SPACE TRAVEL)")
        print("=" * 85)
        sbp_stats = self.evaluate_superbpe()
        print(
            f"  Vocab                   : {sbp_stats['vocab_before']} -> {sbp_stats['vocab_after']} tokens "
            f"({sbp_stats['superbpe_merges_applied']} merges)"
        )
        print(
            f"  Token Count             : {sbp_stats['token_count_before']} -> {sbp_stats['token_count_after']} "
            f"({sbp_stats['token_reduction_pct']}% fewer)"
        )
        print(
            f"  Compression             : {sbp_stats['bytes_per_token_before']} -> {sbp_stats['bytes_per_token_after']} bytes/tok"
        )

        print("\n" + "=" * 85)
        print("COMPARATIVE ENGINE BASELINES")
        print("=" * 85)
        baselines = self.evaluate_external_baselines()
        for engine, stats in baselines.items():
            if "error" in stats:
                print(f"  {engine:<24} : {stats['error']}")
            else:
                print(
                    f"  {engine:<24} : {stats['tokens']} tokens | {stats['tokens_sec']} tok/sec ({stats['time_sec']}s)"
                )
        print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Caliper tokenizer benchmarks.")
    parser.add_argument(
        "--large-payloads",
        action="store_true",
        help="also run the 1 MiB and 10 MiB local throughput workloads",
    )
    args = parser.parse_args()
    suite = TokenizerBenchmarkSuite()
    suite.print_summary_report(include_large_payloads=args.large_payloads)
