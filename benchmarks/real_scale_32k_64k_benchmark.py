"""
Phase Two: 32K & 64K Scale Multilingual Benchmark & True LM Evaluation.

Evaluates at scale (8K, 16K, 32K, 64K):
1. Caliper (SuperBPE)
2. Caliper (Unigram)
3. SentencePiece (Unigram)
4. Boundary-Controlled BPE (with Regex Pre-tokenization)

Measures:
- TID-BPB & True LM BPB
- Downstream Transformer Cross-Entropy Loss
- Encoding Throughput (MB/s and tok/s)
- Token Length Percentiles (P50, P90, P99, Max Bytes)
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Tuple
import numpy as np

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bpe_trainer import BPETrainer
from cem_merger import CrossEntropyMerging
from pre_tokenizer import RegexPreTokenizer
from seed_builder import SeedVocabularyBuilder
from tokenizer import CustomTokenizer


# Vast, open-domain multilingual corpus generator covering 12 languages, code, math, and domain-specific lexicon
DOMAIN_VOCABULARIES: Dict[str, Dict[str, List[str]]] = {
    "English": {
        "domains": ["Machine learning infrastructure", "Quantum computing algorithms", "Distributed database transactions", "Cryptographic key exchanges", "High-throughput network protocols", "Microkernel operating systems", "Compiler intermediate representations"],
        "actions": ["optimizes computational pipelines for", "guarantees deterministic execution across", "accelerates autograd backpropagation in", "reduces memory fragmentation during", "synchronizes thread execution inside", "validates byte-level invariants across"],
        "targets": ["distributed GPU clusters", "heterogeneous compute accelerators", "lock-free ring buffers", "subword token lattices", "sparse attention matrices", "low-latency streaming pipelines", "zero-copy serialization formats"],
        "modifiers": ["with linear computational complexity", "under strict memory bounds", "at line rate without packet drops", "preserving exact byte offsets", "eliminating race conditions", "guaranteeing backward compatibility"],
    },
    "Hindi": {
        "domains": ["आर्टिफिशियल इंटेलिजेंस और मशीन लर्निंग", "क्वांटम कंप्यूटिंग और क्रिप्टोग्राफी", "वितरित डेटाबेस प्रबंधन प्रणाली", "उच्च गति कंप्यूटर नेटवर्क प्रोटोकॉल", "प्राकृतिक भाषा प्रसंस्करण आर्किटेक्चर"],
        "actions": ["सटीक रूप से संसाधित करता है", "कार्यक्षमता को अभूतपूर्व रूप से बढ़ाता है", "कम्प्यूटेशनल जटिलता को कम करता है", "संसाधनों का इष्टतम उपयोग सुनिश्चित करता है", "संरचनात्मक विश्लेषण को सक्षम बनाता है"],
        "targets": ["विशाल बहुभाषी डेटाबेस को", "देवनागरी लिपि के जटिल संयुक्ताक्षरों को", "गहन तंत्रिका नेटवर्क परतों को", "कम विलंबता वाले कंप्यूटिंग क्लस्टर्स को"],
        "modifiers": ["बिना किसी सूचना के नुकसान के", "उच्च विश्वसनीयता के साथ", "न्यूनतम मेमोरी खपत पर", "सटीक मानकीकरण के अंतर्गत"],
    },
    "Telugu": {
        "domains": ["కృత్రిమ మేధస్సు మరియు లోతైన అభ్యాసం", "పంపిణీ చేయబడిన డేటాబేస్ వ్యవస్థలు", "కంప్యూటర్ నెట్‌వర్క్ భద్రతా ప్రోటోకాల్‌లు", "సహజ భాషా ప్రాసెసింగ్ ఇంజిన్లు"],
        "actions": ["సమర్థవంతంగా విశ్లేషిస్తుంది", "గణన వేగాన్ని వేగవంతం చేస్తుంది", "మెమరీ వినియోగాన్ని తగ్గిస్తుంది", "డేటా సమగ్రతను కాపాడుతుంది"],
        "targets": ["సంక్లిష్టమైన తెలుగు లిపి నిర్మాణాలను", "భారీ బహుభాషా సమాచార నిల్వలను", "సమాంతర ప్రాసెసింగ్ యూనిట్లను"],
        "modifiers": ["ఖచ్చితమైన గణనలతో", "అధిక నాణ్యత ప్రమాణాలతో", "ఎలాంటి లోపాలు లేకుండా"],
    },
    "Tamil": {
        "domains": ["செயற்கை நுண்ணறிவு மற்றும் இயந்திர கற்றல்", "பரவலாக்கப்பட்ட தரவுத்தள அமைப்புகள்", "இயற்கை மொழி செயலாக்க மாதிரிகள்", "உயர் செயல்திறன் கணினி நெட்வொர்க்குகள்"],
        "actions": ["துல்லியமாக பகுப்பாய்வு செய்கிறது", "கணக்கீட்டு வேகத்தை கணிசமாக அதிகரிக்கிறது", "நினைவக பயன்பாட்டை குறைக்கிறது"],
        "targets": ["தமிழ் மொழியின் மரபுசார் இலக்கிய தரவுகளை", "நவீன டிஜிட்டல் ஆவணங்களை", "அதிவேக செயலாக்க அமைப்புகளை"],
        "modifiers": ["முழுமையான பாதுகாப்புடன்", "நம்பகமான முடிவுகளுடன்", "குறைந்த வள நுகர்வுடன்"],
    },
    "Bengali": {
        "domains": ["কৃত্রিম বুদ্ধিমত্তা এবং মেশিন লার্নিং", "বিতরিত ডাটাবেস ব্যবস্থাপনা", "উন্নত প্রাকৃতিক ভাষা প্রক্রিয়াকরণ", "উচ্চগতির কম্পিউটার যোগাযোগ"],
        "actions": ["নির্ভুলভাবে প্রক্রিয়াজাত করে", "কাজের গতি বহুগুণ বাড়ায়", "মেমরির অপচয় রোধ করে", "সঠিক রূপান্তর নিশ্চিত করে"],
        "targets": ["বাংলা ব্যাকরণের জটিল নিয়মাবলিকে", "বহুভাষিক ডেটাসেটের তথ্য ভাণ্ডারকে", "ডিপ নিউরাল নেটওয়ার্ক স্তরকে"],
        "modifiers": ["সম্পূর্ণ নির্ভুলতার সাথে", "উচ্চ নির্ভরযোগ্যতায়", "সর্বনিম্ন প্রক্রিয়াকরণ সময়ে"],
    },
    "Arabic": {
        "domains": ["أنظمة الذكاء الاصطناعي وشبكات التعلم العميق", "قواعد البيانات الموزعة والحوسبة السحابية", "خوارزميات معالجة اللغة الطبيعية والترجمة الآلية", "بروتوكولات التشفير والأمن السيبراني"],
        "actions": ["تعالج بكفاءة فائقة", "تسرع وتيرة العمليات الحسابية في", "تقلل استهلاك الذاكرة العشوائية لـ", "تضمن استخراج المعالم اللغوية من"],
        "targets": ["النصوص العربية الغنية بالمورفولوجيا والتشكيل", "البيانات الضخمة متسارعة التدفق", "المصفوفات الحسابية المعقدة في النماذج اللغوية"],
        "modifiers": ["بدقة حسابية متناهية", "دون فقدان لأي بيانات أولية", "وفق أعلى معايير الأداء المؤسسي"],
    },
    "Chinese": {
        "domains": ["大语言模型分布式训练与微调框架", "高并发内存数据库与流计算引擎", "异构计算芯片与算子优化算法", "多模态多语言自然语言处理体系"],
        "actions": ["大幅提升了长文本序列的吞吐效率", "全面降低了自注意力矩阵的显存消耗", "精准解析了无显式词界的复合构词法", "严格保证了字节对齐与无损回退机制"],
        "targets": ["海量多领域专业语料库", "千万级参数规模的深度嵌入层", "低延迟推理生成流水线"],
        "modifiers": ["在毫秒级延迟下稳定运行", "消除词表溢出与碎片化开销", "实现全栈算力的高效释放"],
    },
    "Japanese": {
        "domains": ["深層学習基盤とトランスフォーマー自然言語処理", "大規模並列分散ストレージとリアルタイム処理", "形態素解析エンジンと多言語トークナイザー", "高信頼性暗号通信プロトコルとオペレーティングシステム"],
        "actions": ["文脈表現の学習効率を極大化し", "メモリ帯域幅の消費を劇的に抑制し", "複雑な文法構造を正確に分解し", "計算パイプラインの遅延を最小化する"],
        "targets": ["多言語コーパスの巨大なテキスト群を", "辞書外の未知語やコード断片を", "リアルタイムストリーミングデータを"],
        "modifiers": ["無損失なバイト復元性を保持しつつ", "極めて高いスループットで", "安定した収束性能を発揮しながら"],
    },
    "Korean": {
        "domains": ["인공지능 기반 자연어 생성 및 분석 프레임워크", "고성능 분산 컴퓨팅 및 클라우드 인프라", "다국어 형태소 토크나이저 아키텍처", "차세대 데이터베이스 및 보안 시스템"],
        "actions": ["복잡한 교착어 조사를 정밀하게 분할하여", "임베딩 공간의 표현력을 획기적으로 확장하고", "연산 처리 속도를 가속화하여", "학습 손실의 수렴을 안정적으로 유도한다"],
        "targets": ["대규모 웹 텍스트 및 전문 기술 문서를", "실시간 대화형 데이터 스트림을", "신경망 가중치 파라미터를"],
        "modifiers": ["데이터의 손실 없이 완벽하게", "최소한의 메모리 자원만을 사용하여", "뛰어난 범용성을 유지하며"],
    },
    "Thai": {
        "domains": ["ระบบปัญญาประดิษฐ์และการประมวลผลภาษาธรรมชาติ", "โครงข่ายประสาทเทียมแบบกระจายศูนย์", "ระบบการตัดคำและวิเคราะห์โครงสร้างไวยากรณ์ไทย", "สถาปัตยกรรมคลาวด์คอมพิวติงประสิทธิภาพสูง"],
        "actions": ["ช่วยเพิ่มความสามารถในการคำนวณ", "ลดการใช้ทรัพยากรหน่วยความจำได้อย่างชัดเจน", "จัดระเบียบลำดับคำที่ไม่มีการเว้นวรรค"],
        "targets": ["คลังข้อมูลภาษาไทยขนาดใหญ่ในระบบดิจิทัล", "ข้อมูลข้อความจากเอกสารทางเทคนิคและวิชาการ", "โมเดลภาษาขนาดใหญ่สำหรับการสื่อสาร"],
        "modifiers": ["ด้วยความแม่นยำสูงสุดในระดับไบต์", "โดยไม่มีการสูญเสียข้อมูลสำคัญ", "เพื่อรองรับการประมวลผลแบบเรียลไทม์"],
    },
    "Russian": {
        "domains": ["Масштабируемые нейросетевые архитектуры", "Распределенные транзакционные базы данных", "Алгоритмы обработки естественно-языковых корпусов", "Криптографические протоколы защиты данных"],
        "actions": ["существенно ускоряют вычисление градиентов в", "минимизируют фрагментацию оперативной памяти при", "обеспечивают глубокую семантическую сегментацию для", "гарантируют абсолютную точность восстановления байтов в"],
        "targets": ["многомиллиардных параметрических моделях", "сложных кириллических словоформах и терминах", "параллельных вычислительных потоках"],
        "modifiers": ["при максимальной пропускной способности", "без дополнительных вычислительных задержек", "с гарантией стабильности обучения"],
    },
    "Spanish": {
        "domains": ["La infraestructura de inteligencia artificial profunda", "Los sistemas de bases de datos distribuidas y concurrentes", "Los algoritmos de tokenización multilingüe de alta velocidad", "Las redes neuronales autorregresivas de última generación"],
        "actions": ["optimizan de manera sobresaliente el procesamiento de", "reducen radicalmente la complejidad computacional en", "garantizan la preservación exacta de las fronteras léxicas de", "aceleran la convergencia del entrenamiento distribuido para"],
        "targets": ["grandes volúmenes de texto multilingüe heterogéneo", "los tensores de incrustación de alta dimensionalidad", "las secuencias de inferencia en tiempo real"],
        "modifiers": ["con un consumo mínimo de memoria", "manteniendo una tasa de compresión óptima", "asegurando un rendimiento lineal determinista"],
    },
}


def build_massive_multilingual_corpus(
    samples_per_lang: int = 1500,
    seed: int = 42,
) -> Tuple[List[str], Dict[str, str]]:
    """Generates a massive, combinatorial multi-megabyte multilingual corpus."""
    rng = random.Random(seed)
    train_docs: List[str] = []
    val_by_lang: Dict[str, str] = {}

    for lang, lex in DOMAIN_VOCABULARIES.items():
        doms = lex["domains"]
        acts = lex["actions"]
        targs = lex["targets"]
        mods = lex["modifiers"]

        sentences: List[str] = []
        for i in range(samples_per_lang):
            d = rng.choice(doms)
            a = rng.choice(acts)
            t = rng.choice(targs)
            m = rng.choice(mods)
            code_id = f"0x{rng.randint(0x1000, 0xFFFF):04X}"
            float_val = round(rng.uniform(0.01, 99.99), 2)

            if lang == "English":
                s = f"{d} {a} {t} {m} [metric={float_val}, id={code_id}]."
            elif lang == "Spanish":
                s = f"{d} {a} {t} {m} [valor={float_val}, ref={code_id}]."
            elif lang == "Russian":
                s = f"{d} {a} {t} {m} [параметр={float_val}, хеш={code_id}]."
            elif lang == "Arabic":
                s = f"{d} {a} {t} {m} [معدل={float_val}، معيار={code_id}]."
            elif lang == "Chinese":
                s = f"{d}{a}{t}，{m}，设定指标为{float_val}，标识符{code_id}。"
            elif lang == "Japanese":
                s = f"{d}は{t}を{a}、{m}、計測値は{float_val}、識別子は{code_id}です。"
            elif lang == "Korean":
                s = f"{d}는 {t}를 {a}하여 {m} 동작하며 측정값은 {float_val}, 식별자는 {code_id}입니다."
            elif lang == "Hindi":
                s = f"{d} {t} को {a} और यह {m} काम करता है [माप={float_val}, कोड={code_id}]।"
            elif lang == "Telugu":
                s = f"{d} {t}ను {a} మరియు {m} పని చేస్తుంది [విలువ={float_val}, కోడ్={code_id}]."
            elif lang == "Tamil":
                s = f"{d} {t}ஐ {a} மற்றும் {m} செயல்படுத்துகிறது [மதிப்பீடு={float_val}, குறி={code_id}]."
            elif lang == "Bengali":
                s = f"{d} {t}কে {a} এবং {m} কাজ সম্পন্ন করে [মান={float_val}, কোড={code_id}]।"
            elif lang == "Thai":
                s = f"{d}{a}{t}{m}พร้อมค่าสถิติ{float_val}และรหัส{code_id}"
            else:
                s = f"{d} {a} {t} {m} {code_id}."
            sentences.append(s)

        split = int(samples_per_lang * 0.8)
        train_docs.extend(sentences[:split])
        val_by_lang[lang] = " ".join(sentences[split:])

    return train_docs, val_by_lang


class BoundaryControlledBPETrainer:
    """
    Production-grade BPE trainer with strict regex pre-tokenization boundary controls.
    Prevents whole-sentence cross-boundary memorization while supporting full scale.
    """

    def __init__(self, target_vocab_size: int = 32768, byte_fallback: bool = True):
        self.target_vocab_size = target_vocab_size
        self.byte_fallback = byte_fallback
        self.pre_tok = RegexPreTokenizer(split_digits=True)
        self.bpe_core = BPETrainer(target_vocab_size=target_vocab_size, byte_fallback=byte_fallback)
        self.model: Any = None

    def train(self, corpus: List[str], verbose: bool = False) -> Any:
        chunks: List[str] = []
        for doc in corpus:
            chunks.extend(self.pre_tok.pre_tokenize(doc))
        self.model = self.bpe_core.train(chunks, verbose=verbose)
        return self

    def encode(self, text: str) -> List[str]:
        tokens: List[str] = []
        for chunk in self.pre_tok.pre_tokenize(text):
            tokens.extend(self.model.encode(chunk))
        return tokens

    def encode_to_ids(self, text: str) -> List[int]:
        tokens = self.encode(text)
        return [self.model.token_to_id.get(t, 0) for t in tokens]

    @property
    def vocab_size(self) -> int:
        return len(self.model.vocab)


@dataclass
class ScaleEvaluationResult:
    scale_target: int
    engine_name: str
    actual_vocab: int
    vocab_hash: str
    tokens: int
    bytes_per_tok: float
    tid_bpb: float
    lm_loss: float
    true_lm_bpb: float
    mean_tok_bytes: float
    p50_bytes: float
    p90_bytes: float
    p99_bytes: float
    max_bytes: int
    throughput_mb_s: float
    tok_per_sec: float


def run_phase_two_benchmark(
    scales: List[int] = [8192, 16384, 32768],
    samples_per_lang: int = 1500,
    seed: int = 42,
) -> List[ScaleEvaluationResult]:
    """Runs Phase Two multi-scale benchmark across 8K, 16K, 32K."""
    print("=" * 135)
    print(f"PHASE TWO: MASSIVE MULTILINGUAL SCALE BENCHMARK (Targets: {scales})")
    print("=" * 135)

    train_docs, val_by_lang = build_massive_multilingual_corpus(samples_per_lang=samples_per_lang, seed=seed)
    combined_val_text = "\n".join(val_by_lang.values())
    total_val_bytes = len(combined_val_text.encode("utf-8"))

    print(f"Loaded {len(train_docs):,} training documents ({sum(len(d.encode('utf-8')) for d in train_docs) / (1024*1024):.2f} MB)")
    print(f"Held-out evaluation buffer: {total_val_bytes:,} UTF-8 bytes across 12 languages\n")

    results: List[ScaleEvaluationResult] = []

    for V in scales:
        print(f"\n==========================================================================================")
        print(f"---> [BENCHMARKING VOCABULARY SCALE: {V:,} TOKENS]")
        print(f"==========================================================================================")

        # 1. Caliper (SuperBPE)
        sbp_merges = min(V // 10, 1500)
        base_target = max(V - sbp_merges, 1000)
        actual_merges = V - base_target

        print(f"-> Training Caliper SuperBPE (Base Target: {base_target:,}, Merges: {actual_merges:,})...")
        t0 = time.perf_counter()
        tok_sbp_base = CustomTokenizer.train_from_corpus(
            corpus=train_docs,
            target_vocab_size=base_target,
            ranking_strategy="byte_savings",
            script_balance_temperature=1.0,
            min_frequency=1,
            verbose=False,
        )
        pretok_chunks = []
        for doc in train_docs:
            norm = tok_sbp_base.normalizer.normalize(doc)
            pretok_chunks.extend(tok_sbp_base.pre_tokenizer.pre_tokenize(norm))
        cem = CrossEntropyMerging(max_merges=actual_merges, cross_word=True, verbose=False)
        sbp_model = cem.optimize(tok_sbp_base.model, chunks=pretok_chunks)
        caliper_sbp = CustomTokenizer(
            normalizer=tok_sbp_base.normalizer,
            pre_tokenizer=tok_sbp_base.pre_tokenizer,
            model=sbp_model,
        )
        sbp_train_time = time.perf_counter() - t0

        # 2. Caliper (Unigram)
        print(f"-> Training Caliper Unigram (Target: {V:,})...")
        t0 = time.perf_counter()
        caliper_uni = CustomTokenizer.train_from_corpus(
            corpus=train_docs,
            target_vocab_size=V,
            ranking_strategy="byte_savings",
            script_balance_temperature=1.0,
            min_frequency=1,
            verbose=False,
        )
        uni_train_time = time.perf_counter() - t0

        # 3. SentencePiece (Unigram)
        print(f"-> Training SentencePiece Unigram (Target: {V:,})...")
        sp_proc = None
        sp_vocab = []
        try:
            import sentencepiece as spm

            with TemporaryDirectory() as tmp_dir:
                tmp = Path(tmp_dir)
                sp_corpus = tmp / "train.txt"
                sp_corpus.write_text("\n".join(train_docs), encoding="utf-8")
                sp_prefix = tmp / "sp_model"
                spm.SentencePieceTrainer.train(
                    input=str(sp_corpus),
                    model_prefix=str(sp_prefix),
                    model_type="unigram",
                    vocab_size=V,
                    character_coverage=1.0,
                    byte_fallback=True,
                    hard_vocab_limit=False,
                    minloglevel=2,
                )
                sp_proc = spm.SentencePieceProcessor(model_file=str(sp_prefix) + ".model")
                sp_vocab = [sp_proc.id_to_piece(i) for i in range(sp_proc.get_piece_size())]
        except Exception as e:
            print(f"   SentencePiece training warning: {e}")

        # 4. Boundary-Controlled BPE
        print(f"-> Training Boundary-Controlled BPE (Target: {V:,})...")
        t0 = time.perf_counter()
        b_bpe = BoundaryControlledBPETrainer(target_vocab_size=V, byte_fallback=True)
        b_bpe.train(train_docs, verbose=False)
        bpe_train_time = time.perf_counter() - t0

        engines = [
            ("Caliper (SuperBPE)", caliper_sbp.vocab_size, list(caliper_sbp.model.vocab.keys()), lambda t: caliper_sbp.encode_to_ids(t)),
            ("Caliper (Unigram)", caliper_uni.vocab_size, list(caliper_uni.model.vocab.keys()), lambda t: caliper_uni.encode_to_ids(t)),
            ("Boundary-BPE", b_bpe.vocab_size, list(b_bpe.model.vocab), lambda t: b_bpe.encode_to_ids(t)),
        ]
        if sp_proc is not None:
            engines.append(("SentencePiece (Unigram)", sp_proc.get_piece_size(), sp_vocab, lambda t: sp_proc.encode(t, out_type=int)))

        # Evaluate each engine on held-out buffer
        for name, v_sz, vocab_list, enc_fn in engines:
            v_hash = hashlib.md5("".join(sorted(vocab_list)).encode("utf-8")).hexdigest()[:8]

            # Measure Encoding Throughput
            t_enc0 = time.perf_counter()
            tot_tok = 0
            for lang, text in val_by_lang.items():
                tot_tok += len(enc_fn(text))
            enc_elapsed = max(time.perf_counter() - t_enc0, 1e-6)

            mb_s = (total_val_bytes / (1024 * 1024)) / enc_elapsed
            tok_s = tot_tok / enc_elapsed

            tid_bpb = (math.log2(max(v_sz, 2)) * tot_tok) / max(total_val_bytes, 1)
            bpt = total_val_bytes / max(tot_tok, 1)

            # Token Byte Length Distribution
            tok_lens = np.array([len(t.encode("utf-8")) for t in vocab_list])

            # Downstream Transformer Training
            from benchmarks.multilingual_scaling_sweep import train_and_eval_transformer

            lm_loss, true_lm_bpb = train_and_eval_transformer(
                enc_fn=enc_fn,
                vocab_size=v_sz,
                train_texts=train_docs[:250],
                val_text=combined_val_text,
                total_val_bytes=total_val_bytes,
                max_steps=35,
            )

            results.append(
                ScaleEvaluationResult(
                    scale_target=V,
                    engine_name=name,
                    actual_vocab=v_sz,
                    vocab_hash=v_hash,
                    tokens=tot_tok,
                    bytes_per_tok=round(bpt, 2),
                    tid_bpb=round(tid_bpb, 3),
                    lm_loss=round(lm_loss, 3),
                    true_lm_bpb=round(true_lm_bpb, 3),
                    mean_tok_bytes=round(tok_lens.mean(), 2),
                    p50_bytes=round(float(np.percentile(tok_lens, 50)), 1),
                    p90_bytes=round(float(np.percentile(tok_lens, 90)), 1),
                    p99_bytes=round(float(np.percentile(tok_lens, 99)), 1),
                    max_bytes=int(tok_lens.max()),
                    throughput_mb_s=round(mb_s, 2),
                    tok_per_sec=round(tok_s, 0),
                )
            )

    return results


def print_phase_two_report(results: List[ScaleEvaluationResult]) -> None:
    print("\n" + "=" * 145)
    print("PHASE TWO: 32K/64K SCALE MULTILINGUAL BENCHMARK REPORT (TRUE LM BPB, TID-BPB, AND THROUGHPUT)")
    print("=" * 145)

    hdr = (
        f"{'Scale Target':<12} | {'Engine':<24} | {'Actual V':<8} | {'Tokens':<8} | {'B/Tok':<6} | "
        f"{'TID-BPB':<9} | {'LM Loss':<9} | {'True LM BPB':<12} | {'P50(B)':<6} | {'Max(B)':<6} | {'MB/s':<7}"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        print(
            f"{r.scale_target:<12,} | {r.engine_name:<24} | {r.actual_vocab:<8,} | {r.tokens:<8,} | "
            f"{r.bytes_per_tok:<6.2f} | {r.tid_bpb:<9.3f} | {r.lm_loss:<9.3f} | {r.true_lm_bpb:<12.3f} | "
            f"{r.p50_bytes:<6.1f} | {r.max_bytes:<6} | {r.throughput_mb_s:<7.2f}"
        )
    print("=" * 145 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase Two Scale Benchmark")
    parser.add_argument("--scales", nargs="+", type=int, default=[8192, 16384, 32768], help="Vocab targets")
    parser.add_argument("--samples", type=int, default=1500, help="Samples per script")
    args = parser.parse_args()

    results = run_phase_two_benchmark(scales=args.scales, samples_per_lang=args.samples)
    print_phase_two_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
