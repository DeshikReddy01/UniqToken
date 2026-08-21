"""
Apples-to-Apples Multilingual Tokenizer Benchmark.

Compares Caliper (Unigram), Caliper (SuperBPE), SentencePiece, and BPE
on identical training and held-out evaluation splits at exact equal vocabulary sizes.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bpe_trainer import BPETrainer
from cem_merger import CrossEntropyMerging
from tokenizer import CustomTokenizer

# Standardized 12-Script Multilingual Corpus
MULTILINGUAL_CORPUS: Dict[str, List[str]] = {
    "English": [
        "The transformer architecture relies on subword tokenization to compress sequence length.",
        "Neural language modeling balances vocabulary size against computational embedding cost.",
        "Exact offset alignment is essential for accurate span extraction and structured decoding.",
    ]
    * 50,
    "Hindi": [
        "प्राकृतिक भाषा प्रसंस्करण और कंप्यूटर विज्ञान में टोकनाइज़र एक महत्वपूर्ण घटक है।",
        "देवनागरी लिपि में अक्षरों और मात्राओं का उचित संयोजन विखंडन को रोकता है।",
        "भाषा मॉडल की दक्षता और सटीकता सही सबवर्ड टोकनीकरण पर निर्भर करती है।",
    ]
    * 50,
    "Telugu": [
        "సహజ భాషా ప్రాసెసింగ్ కంప్యూటర్ సైన్స్‌లో అత్యంత ముఖ్యమైన రంగం.",
        "తెలుగు లిపిలో అక్షరాలు మరియు గుణింతాల అమరిక చాలా సంక్లిష్టమైనది.",
        "సరైన టోకనైజర్ ఉపయోగించడం వల్ల భాషా నమూనాల పనితీరు మెరుగుపడుతుంది.",
    ]
    * 50,
    "Tamil": [
        "இயற்கை மொழி செயலாக்கம் கணினி அறிவியலில் ஒரு முக்கிய பகுதியாகும்.",
        "தமிழ் எழுத்துக்களின் துல்லியமான பயன்பாடு மொழி மாதிரிகளுக்கு அவசியமானது.",
        "சரியான டோக்கனைசர் சூழல் சாளரத்தின் திறனை கணிசமாக அதிகரிக்கிறது.",
    ]
    * 50,
    "Bengali": [
        "প্রাকৃতিক ভাষা প্রক্রিয়াকরণ কম্পিউটার বিজ্ঞানের একটি অত্যন্ত গুরুত্বপূর্ণ শাখা।",
        "বাংলা লিপির জটিল গঠন এবং যুক্তাক্ষর সংরক্ষণে সঠিক টোকেনাইজার প্রয়োজন।",
        "ভাষার মডেলের দক্ষতা সাবওয়ার্ড বিভাজনের সঠিকতার উপর নির্ভর করে।",
    ]
    * 50,
    "Arabic": [
        "تعتبر معالجة اللغات الطبيعية وتجزئة النصوص من أهم ركائز الذكاء الاصطناعي الحديث.",
        "يتطلب التعامل مع اللغة العربية دعماً دقيقاً للحركات وعلامات التشكيل لمنع التجزئة.",
        "يعتمد أداء النماذج اللغوية على كفاءة التوكنايزر في ضغط السياق وتقليل الخصوبة.",
    ]
    * 50,
    "Chinese": [
        "自然语言处理和大型语言模型依赖高效的分词技术来提升上下文利用率。",
        "汉字作为表意文字不需要显式空格，子词切分算法必须准确识别词汇边界。",
        "高质量的分词器能够显著降低序列长度并提升推理计算效率。",
    ]
    * 50,
    "Japanese": [
        "自然言語処理におけるトークナイザーは、テキストを一連のサブワードに分割します。",
        "日本語のように単語境界が自明でない言語では、バイトフォールバックが重要です。",
        "正確なアライメントと形態素境界の維持がモデルの性能を左右します。",
    ]
    * 50,
    "Korean": [
        "자연어 처리에서 토크나이저는 텍스트를 최적의 서브워드 시퀀스로 분할합니다.",
        "한국어의 교착어적 특성과 조사를 효과적으로 처리하는 것이 핵심입니다.",
        "문맥 효율성을 극대화하기 위해 정확한 토큰 분할 알고리즘이 필수적입니다.",
    ]
    * 50,
    "Thai": [
        "การประมวลผลภาษาธรรมชาติและการตัดคำเป็นขั้นตอนพื้นฐานที่สำคัญยิ่ง.",
        "ภาษาไทยไม่มีการเว้นวรรคระหว่างคำทำให้อัลกอริทึมการตัดคำมีความท้าทายสูง.",
        "การเลือกใช้โทเคไนเซอร์ที่เหมาะสมช่วยเพิ่มประสิทธิภาพของโมเดลภาษาขนาดใหญ่.",
    ]
    * 50,
    "Russian": [
        "Обработка естественного языка и токенизация лежат в основе современных трансформеров.",
        "Кириллический текст требует сбалансированного словаря для предотвращения фрагментации.",
        "Эффективность контекстного окна напрямую зависит от среднего сжатия байт на токен.",
    ]
    * 50,
    "Spanish": [
        "El procesamiento del lenguaje natural depende de la tokenización precisa de subpalabras.",
        "Los modelos de lenguaje modernos equilibran la fertilidad léxica con el tamaño del vocabulario.",
        "La compresión eficiente del contexto mejora la velocidad de inferencia y reduce el costo computacional.",
    ]
    * 50,
}


@dataclass
class LanguageEvalResult:
    language: str
    tokens: int
    bytes_count: int
    words_count: int
    bytes_per_token: float
    tokens_per_word: float  # Fertility
    tokens_per_byte: float
    bits_per_byte: float
    throughput_tok_sec: float


@dataclass
class EngineBenchmarkSummary:
    engine_name: str
    target_vocab: int
    actual_vocab: int
    total_tokens: int
    total_bytes: int
    overall_bytes_per_token: float
    overall_fertility: float
    overall_bpb: float
    overall_tokens_sec: float
    by_language: Dict[str, LanguageEvalResult]


def build_train_and_val_splits() -> Tuple[List[str], Dict[str, str]]:
    """Splits multilingual corpus into training documents and held-out validation strings."""
    train_docs: List[str] = []
    val_by_lang: Dict[str, str] = {}

    for lang, sentences in MULTILINGUAL_CORPUS.items():
        split_idx = int(len(sentences) * 0.8)
        train_docs.extend(sentences[:split_idx])
        val_by_lang[lang] = " ".join(sentences[split_idx:])

    return train_docs, val_by_lang


def evaluate_tokenizer_on_languages(
    name: str,
    target_vocab: int,
    actual_vocab: int,
    encode_fn: Any,
    val_by_lang: Dict[str, str],
) -> EngineBenchmarkSummary:
    """Evaluates an encoding function across all held-out language splits."""
    by_lang: Dict[str, LanguageEvalResult] = {}
    tot_tokens = 0
    tot_bytes = 0
    tot_words = 0
    tot_time = 0.0

    for lang, text in val_by_lang.items():
        raw_bytes = len(text.encode("utf-8"))
        words = max(len(text.split()), 1)

        t0 = time.perf_counter()
        token_ids = encode_fn(text)
        elapsed = max(time.perf_counter() - t0, 1e-6)

        num_tok = len(token_ids)
        tot_tokens += num_tok
        tot_bytes += raw_bytes
        tot_words += words
        tot_time += elapsed

        bpt = raw_bytes / max(num_tok, 1)
        tpw = num_tok / words
        tpb = num_tok / max(raw_bytes, 1)
        bpb = (math.log(max(actual_vocab, 2)) * num_tok) / (max(raw_bytes, 1) * math.log(2))

        by_lang[lang] = LanguageEvalResult(
            language=lang,
            tokens=num_tok,
            bytes_count=raw_bytes,
            words_count=words,
            bytes_per_token=round(bpt, 3),
            tokens_per_word=round(tpw, 3),
            tokens_per_byte=round(tpb, 3),
            bits_per_byte=round(bpb, 3),
            throughput_tok_sec=round(num_tok / elapsed, 1),
        )

    overall_bpt = tot_bytes / max(tot_tokens, 1)
    overall_fertility = tot_tokens / max(tot_words, 1)
    overall_bpb = (math.log(max(actual_vocab, 2)) * tot_tokens) / (max(tot_bytes, 1) * math.log(2))
    overall_tps = tot_tokens / max(tot_time, 1e-6)

    return EngineBenchmarkSummary(
        engine_name=name,
        target_vocab=target_vocab,
        actual_vocab=actual_vocab,
        total_tokens=tot_tokens,
        total_bytes=tot_bytes,
        overall_bytes_per_token=round(overall_bpt, 3),
        overall_fertility=round(overall_fertility, 3),
        overall_bpb=round(overall_bpb, 3),
        overall_tokens_sec=round(overall_tps, 1),
        by_language=by_lang,
    )


def run_multilingual_benchmark(vocab_size: int = 1000, superbpe_merges: int = 40) -> List[EngineBenchmarkSummary]:
    """Runs equal-vocab benchmark across Caliper Unigram, SuperBPE, SentencePiece, and BPE."""
    train_docs, val_by_lang = build_train_and_val_splits()
    summaries: List[EngineBenchmarkSummary] = []

    print(f"Training on {len(train_docs)} multilingual documents (Target Vocab: {vocab_size})...\n")

    # 1. Caliper (Unigram)
    caliper_unigram = CustomTokenizer.train_from_corpus(
        corpus=train_docs,
        target_vocab_size=vocab_size,
        ranking_strategy="char_savings",
        min_frequency=1,
        verbose=False,
    )
    s_unigram = evaluate_tokenizer_on_languages(
        "Caliper (Unigram)",
        vocab_size,
        caliper_unigram.vocab_size,
        lambda t: caliper_unigram.encode_to_ids(t),
        val_by_lang,
    )
    summaries.append(s_unigram)

    # 2. Caliper (SuperBPE)
    pretok_chunks = []
    for doc in train_docs:
        norm = caliper_unigram.normalizer.normalize(doc)
        pretok_chunks.extend(caliper_unigram.pre_tokenizer.pre_tokenize(norm))
    cem = CrossEntropyMerging(max_merges=superbpe_merges, cross_word=True, verbose=False)
    sbp_model = cem.optimize(caliper_unigram.model, chunks=pretok_chunks)
    caliper_sbp = CustomTokenizer(
        normalizer=caliper_unigram.normalizer,
        pre_tokenizer=caliper_unigram.pre_tokenizer,
        model=sbp_model,
    )
    s_superbpe = evaluate_tokenizer_on_languages(
        "Caliper (SuperBPE)",
        vocab_size,
        caliper_sbp.vocab_size,
        lambda t: caliper_sbp.encode_to_ids(t),
        val_by_lang,
    )
    summaries.append(s_superbpe)

    # 3. SentencePiece (Unigram)
    try:
        import sentencepiece as spm

        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            sp_corpus = tmp / "train.txt"
            sp_corpus.write_text("\n".join(train_docs), encoding="utf-8")
            sp_model_prefix = tmp / "sp_model"

            spm.SentencePieceTrainer.train(
                input=str(sp_corpus),
                model_prefix=str(sp_model_prefix),
                model_type="unigram",
                vocab_size=vocab_size,
                character_coverage=1.0,
                byte_fallback=True,
                hard_vocab_limit=False,
                minloglevel=2,
            )
            sp_proc = spm.SentencePieceProcessor(model_file=str(sp_model_prefix) + ".model")
            s_sp = evaluate_tokenizer_on_languages(
                "SentencePiece (Unigram)",
                vocab_size,
                sp_proc.get_piece_size(),
                lambda t: sp_proc.encode(t, out_type=int),
                val_by_lang,
            )
            summaries.append(s_sp)
    except Exception as e:
        print(f"Warning: SentencePiece baseline skipped ({e})")

    # 4. Standard BPE
    bpe_tr = BPETrainer(target_vocab_size=vocab_size, byte_fallback=True)
    bpe_m = bpe_tr.train(train_docs, verbose=False)
    s_bpe = evaluate_tokenizer_on_languages(
        "Standard BPE",
        vocab_size,
        len(bpe_m.vocab),
        lambda t: [bpe_m.token_to_id.get(tok, 0) for tok in bpe_m.encode(t)],
        val_by_lang,
    )
    summaries.append(s_bpe)

    return summaries


def print_comparison_tables(summaries: List[EngineBenchmarkSummary]) -> None:
    """Prints comprehensive overall and per-language comparison tables."""
    print("=" * 110)
    print("APPLES-TO-APPLES MULTILINGUAL TOKENIZER BENCHMARK (EQUAL COMPUTE & VOCAB)")
    print("=" * 110)

    hdr = f"{'Tokenizer Engine':<26} | {'Vocab':<6} | {'Tokens':<7} | {'Bytes/Tok':<10} | {'Fertility':<10} | {'Bits/Byte (BPB)':<16} | {'Tok/Sec':<10}"
    print(hdr)
    print("-" * len(hdr))

    for s in summaries:
        print(
            f"{s.engine_name:<26} | {s.actual_vocab:<6} | {s.total_tokens:<7} | "
            f"{s.overall_bytes_per_token:<10} | {s.overall_fertility:<10} | "
            f"{s.overall_bpb:<16} | {s.overall_tokens_sec:<10}"
        )
    print("=" * 110)

    # Per-Language Breakdown
    languages = list(summaries[0].by_language.keys())
    print("\n" + "=" * 110)
    print("LANGUAGE-BY-LANGUAGE COMPRESSION & FERTILITY BREAKDOWN")
    print("=" * 110)

    print(f"{'Language':<12} | " + " | ".join(f"{s.engine_name[:18]:<18} (B/T, Fert, BPB)" for s in summaries))
    print("-" * 110)

    for lang in languages:
        cols = []
        for s in summaries:
            res = s.by_language.get(lang)
            if res:
                cols.append(f"{res.bytes_per_token:>4.2f} / {res.tokens_per_word:>4.2f} / {res.bits_per_byte:>5.2f}")
            else:
                cols.append("N/A")
        print(f"{lang:<12} | " + " | ".join(f"{col:<25}" for col in cols))
    print("=" * 110 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apples-to-Apples Multilingual Tokenizer Benchmark")
    parser.add_argument("--vocab-size", type=int, default=600, help="Target vocabulary size for all engines")
    parser.add_argument("--superbpe-merges", type=int, default=40, help="SuperBPE cross-word merges")
    args = parser.parse_args()

    summaries = run_multilingual_benchmark(
        vocab_size=args.vocab_size,
        superbpe_merges=args.superbpe_merges,
    )
    print_comparison_tables(summaries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
