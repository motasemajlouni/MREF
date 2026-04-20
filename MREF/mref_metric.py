"""
mref_metric_equations.py

"Equation version" of MREF that matches the paper section equations
for G_axis, M_axis, and S_axis.
"""

#from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Any
import math
import json
import re
from pathlib import Path

import spacy
from spacy.cli import download as spacy_download

from wordfreq import word_frequency, zipf_frequency

# ---------------------------------------------------------------------
# Optional WordNet (used to match the paper section for M_axis)
# ---------------------------------------------------------------------
try:
    from nltk.corpus import wordnet as wn  # type: ignore
    _HAVE_WORDNET = True
except Exception:
    wn = None
    _HAVE_WORDNET = False


PACKAGE_DIR = Path(__file__).resolve().parent


# ============================= Utilities =============================

def _safe_load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _sentences(doc) -> List[Any]:
    # spaCy sentence iterator
    return list(doc.sents) if doc.has_annotation("SENT_START") else [doc]

def _is_number_like(tok) -> bool:
    # strict numeric / date-ish pattern
    if tok.like_num:
        return True
    return bool(re.fullmatch(r"[\d\.,:/\-]+", tok.text))

def _norm_alnum_tokens(doc) -> List[str]:
    # matches "remove purely non-alphanumeric" intent reasonably well
    out = []
    for t in doc:
        s = t.text
        if any(ch.isalnum() for ch in s):
            out.append(s)
    return out

def _gibbs_softmin(values: List[float], beta: float) -> float:
    # softmin_beta(x) = sum x_i exp(-x_i/beta) / sum exp(-x_i/beta)
    if not values:
        return 0.0
    if beta <= 0:
        return min(values)
    w = [math.exp(-v / beta) for v in values]
    Z = sum(w)
    if Z <= 0:
        return min(values)
    return sum(v * wi for v, wi in zip(values, w)) / Z

def _softmax_fusion(a: float, b: float, beta: float) -> float:
    # paper’s softmax-style fusion used for G_axis local/global
    ea = math.exp(a / beta)
    eb = math.exp(b / beta)
    denom = ea + eb
    if denom == 0:
        return 0.0
    return (a * ea + b * eb) / denom


# ============================= Axis Weights =============================

@dataclass(frozen=True)
class AxisWeights:
    w_G: float = 1/3
    w_M: float = 1/3
    w_S: float = 1/3

    def normalized(self) -> "AxisWeights":
        s = self.w_G + self.w_M + self.w_S
        if s <= 0:
            return AxisWeights(1/3, 1/3, 1/3)
        return AxisWeights(self.w_G/s, self.w_M/s, self.w_S/s)


# ============================= G_axis (equations) =============================

class GrammarAxis_Equations:
    """
    Implements the *section equations* for G_axis:

    - POS tagging of comp and each output sentence
    - Local overlap: n in {2,3,4}
    - Global prior: POS 6-gram index (freq >= 2) OR short-sequence index for L<6
    - Fusion: softmax-style with beta=0.65
    - Sentence pooling: softmin with lambda_sent=0.35
    """

    LANG_TO_SPACY_MODEL: Dict[str, str] = {
        "en": "en_core_web_sm",
        "de": "de_core_news_sm",
        "fr": "fr_core_news_sm",
        "es": "es_core_news_sm",
        "pt": "pt_core_news_sm",
        "it": "it_core_news_sm",
        "nl": "nl_core_news_sm",
        "zh": "zh_core_web_sm",
    }
    def __init__(
        self,
        language: str = "en",
        spacy_model: Optional[str] = None,
        pos_tagset: str = "ptb",  # "ptb" or "upos"
        pos_index_path: Optional[str] = None,
        ngram_n: int = 6,
        count_threshold: int = 2,
        beta_fusion: float = 0.65,
        lambda_sent: float = 0.35,
        max_tatoeba_sentences: Optional[int] = None,
    ) -> None:
        self.language = language.lower()
        self.pos_tagset = pos_tagset.lower()
        self.ngram_n = ngram_n
        self.count_threshold = count_threshold
        self.beta_fusion = beta_fusion
        self.lambda_sent = lambda_sent
        self.max_tatoeba_sentences = max_tatoeba_sentences

        if spacy_model is None:
            if self.language not in self.LANG_TO_SPACY_MODEL:
                raise ValueError(f"No default spaCy model for '{self.language}'. Provide spacy_model.")
            spacy_model = self.LANG_TO_SPACY_MODEL[self.language]
        self.nlp = self._load_spacy_model(spacy_model)

        if pos_index_path is None:
            pos_index_path = str(PACKAGE_DIR / f"pos_prior_{self.language}.json")

        pos_index_path_obj = Path(pos_index_path)

        if not pos_index_path_obj.exists():
            print(
                f"[G_axis eq] POS prior file not found at {pos_index_path_obj}; "
                f"building it automatically from Tatoeba..."
            )
            try:
                self._build_pos_prior_from_tatoeba(out_path=pos_index_path_obj)
            except Exception as e:
                print(f"[G_axis eq] Failed to auto-build POS prior: {e}")

        # Expect:
        # { "idx_ngrams": {"TAG TAG ...": count, ...}, "short_seqs": {"TAG TAG": count, ...} }
        data = _safe_load_json(str(pos_index_path_obj))
        self.idx_ngrams: Dict[str, int] = data.get("idx_ngrams", {}) if isinstance(data, dict) else {}
        self.short_seqs: Dict[str, int] = data.get("short_seqs", {}) if isinstance(data, dict) else {}

    def _load_spacy_model(self, model_name: str):
        try:
            return spacy.load(model_name)
        except Exception:
            print(f"[G_axis eq] spaCy model '{model_name}' not found; downloading...")
            spacy_download(model_name)
            return spacy.load(model_name)

    def _tag_seq(self, sent_doc) -> List[str]:
        tags: List[str] = []
        for tok in sent_doc:
            if tok.is_space:
                continue
            if self.pos_tagset == "upos":
                tag = tok.pos_
                # Keep UPOS consistent: punctuation becomes "PUNCT"
                if tok.is_punct:
                    tag = "PUNCT"
            else:
                # PTB-like: use tok.tag_ (fine-grained)
                tag = tok.tag_
                if tok.is_punct and tag == "":
                    tag = "."
            tags.append(tag)

        # Paper draft forces a terminal punctuation token.
        if tags:
            if self.pos_tagset == "upos":
                if tags[-1] != "PUNCT":
                    tags = tags + ["PUNCT"]
            else:
                if tags[-1] not in {".", "?", "!"}:
                    tags = tags + ["."]
        return tags

    @staticmethod
    def _ngrams(seq: List[str], n: int) -> List[Tuple[str, ...]]:
        if n <= 0 or len(seq) < n:
            return []
        return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]


    def _tatoeba_lang_code(self, lang: str) -> str:
        mapping = {
            "en": "eng",
            "de": "deu",
            "fr": "fra",
            "es": "spa",
            "pt": "por",
            "it": "ita",
            "nl": "nld",
            "zh": "cmn",
        }
        return mapping.get(lang.lower(), "eng")


    def _build_pos_prior_from_tatoeba(self, out_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Build the final grammar resource file for the equation-based MREF version:

            pos_prior_<language>.json

        Output format:
            {
              "idx_ngrams": {...},   # punctuation-aware POS n-gram counts
              "short_seqs": {...}    # punctuation-aware full POS sequences for short sentences
            }
        """
        import urllib.request
        import shutil
        from collections import Counter

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        url = "https://downloads.tatoeba.org/exports/sentences.csv"
        tatoeba_lang = self._tatoeba_lang_code(self.language)
        n = self.ngram_n

        cache_dir = PACKAGE_DIR / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"tatoeba_sentences_{tatoeba_lang}.csv"

        idx_ngrams = Counter()
        short_seqs = Counter()
        max_len_short = 6
        use_upos = (self.pos_tagset == "upos")
        num_used = 0

        if out_path is None:
            out_path = PACKAGE_DIR / f"pos_prior_{self.language}.json"

        # Use cache if available, otherwise download
        if cache_path.exists():
            print(f"[*] MREF[{self.language}]: using cached Tatoeba file at {cache_path}")
            f = open(cache_path, "rb")
        else:
            print(f"[*] MREF[{self.language}]: downloading Tatoeba to {cache_path}")
            with urllib.request.urlopen(url) as resp, open(cache_path, "wb") as out:
                shutil.copyfileobj(resp, out)
            print(f"[*] MREF[{self.language}]: download complete")
            f = open(cache_path, "rb")

        with f:
            pbar = None
            if tqdm is not None:
                if self.max_tatoeba_sentences is not None:
                    pbar = tqdm(
                        total=self.max_tatoeba_sentences,
                        desc=f"MREF[{self.language}] POS prior",
                        unit="sent",
                    )
                else:
                    pbar = tqdm(
                        desc=f"MREF[{self.language}] POS prior",
                        unit="sent",
                    )

            for raw_line in f:
                line = raw_line.decode("utf-8", errors="ignore").rstrip("\n")
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 3:
                    continue

                _, lang_code, text = parts[0], parts[1], parts[2]
                if lang_code != tatoeba_lang:
                    continue
                if not text or not text.strip():
                    continue

                doc = self.nlp(text)

                for sent in doc.sents:
                    tokens = [t for t in sent if not t.is_space]
                    if not tokens:
                        continue

                    # punctuation-aware tag sequence
                    tags = [(t.pos_ if use_upos else t.tag_) for t in tokens]
                    tags = [t for t in tags if t]
                    if not tags:
                        continue

                    # long-sentence n-grams
                    if len(tags) >= n:
                        for i in range(len(tags) - n + 1):
                            key = " ".join(tags[i:i+n])
                            idx_ngrams[key] += 1

                    # short-sentence full sequence fallback
                    if 0 < len(tags) <= max_len_short:
                        key = " ".join(tags)
                        short_seqs[key] += 1

                num_used += 1
                if pbar is not None:
                    pbar.update(1)

                if (
                    self.max_tatoeba_sentences is not None
                    and num_used >= self.max_tatoeba_sentences
                ):
                    break

            if pbar is not None:
                pbar.close()

        data = {
            "idx_ngrams": dict(idx_ngrams),
            "short_seqs": dict(short_seqs),
        }

        with open(out_path, "w", encoding="utf-8") as f_out:
            json.dump(data, f_out, ensure_ascii=False, indent=2)

        print(f"[*] MREF[{self.language}]: saved final POS prior to {out_path}")
        return data

    def _local_score(self, out_tags: List[str], comp_tags: List[str]) -> float:
        # Eq: OL_n = (1/|G_n(out)|) sum_{u in G_n(out)} 1{u in G_n(comp)}
        # g_local = mean over n in {2,3,4} that have positive overlap (N_k^+)
        comp_sets = {
            2: set(self._ngrams(comp_tags, 2)),
            3: set(self._ngrams(comp_tags, 3)),
            4: set(self._ngrams(comp_tags, 4)),
        }
        ol_vals: List[float] = []
        for n in (2, 3, 4):
            outs = self._ngrams(out_tags, n)
            if not outs:
                continue
            hits = sum(1 for u in outs if u in comp_sets[n])
            ol = hits / len(outs)
            if ol > 0:
                ol_vals.append(ol)
        if not ol_vals:
            return 0.0
        return sum(ol_vals) / len(ol_vals)

    def _global_score(self, out_tags: List[str]) -> float:
        L = len(out_tags)
        thr = self.count_threshold
        if L >= self.ngram_n:
            grams = self._ngrams(out_tags, self.ngram_n)
            if not grams:
                return 0.0
            good = 0
            for g in grams:
                key = " ".join(g)
                if self.idx_ngrams.get(key, 0) >= thr:
                    good += 1
            return good / len(grams)
        # short sequence fallback
        key = " ".join(out_tags)
        return 1.0 if self.short_seqs.get(key, 0) >= thr else 0.0

    def score(self, comp: str, simp: str) -> float:
        comp_doc = self.nlp(comp)
        simp_doc = self.nlp(simp)

        comp_tags = self._tag_seq(comp_doc)
        simp_sents = _sentences(simp_doc)

        g_sent: List[float] = []
        for s in simp_sents:
            out_tags = self._tag_seq(s)
            g_local = self._local_score(out_tags, comp_tags)
            g_global = self._global_score(out_tags)
            gk = _softmax_fusion(g_local, g_global, beta=self.beta_fusion)
            g_sent.append(gk)

        if not g_sent:
            return 0.0

        # softmin pooling over sentences
        # alpha_k ∝ exp(-g_k / lambda_sent), SynG = sum alpha_k g_k
        lam = self.lambda_sent
        w = [math.exp(-g / lam) for g in g_sent] if lam > 0 else [0.0]*len(g_sent)
        Z = sum(w)
        if Z <= 0:
            return min(g_sent)
        alpha = [wi / Z for wi in w]
        return sum(a * g for a, g in zip(alpha, g_sent))


# ============================= M_axis (equations) =============================

class MeaningAxis_Equations:
    """
    Implements the *section equations* for M_axis:

    - Lemmatize + lowercase
    - Token regimes:
        stop lemmas: exact only
        block lemmas: exact only (numbers + named entities)
        content lemmas: WordNet expansion (synonyms + derivational forms)
    - Information weights via per-text normalized wordfreq frequency:
        q_C(t)=freq(t)/sum_{u in V_C}freq(u), w_C(t)=-log(q_C(t))
        q_S similarly
    - Soft overlap:
        OV_C(t) = min{ c_C(t), sum_{u in Syn(t) ∩ V_S} c_S(u) }
        OV_S(t) = min{ c_S(t), sum_{u in Syn(t) ∩ V_C} c_C(u) }
    - Final M_axis: harmonic mean of P and R
    """

    LANG_TO_SPACY_MODEL: Dict[str, str] = {
        "en": "en_core_web_sm",
        "de": "de_core_news_sm",
        "fr": "fr_core_news_sm",
        "es": "es_core_news_sm",
        "pt": "pt_core_news_sm",
        "it": "it_core_news_sm",
        "nl": "nl_core_news_sm",
        "zh": "zh_core_web_sm",
    }

    def __init__(
        self,
        language: str = "en",
        spacy_model: Optional[str] = None,
        eps: float = 1e-12,
        use_wordnet: bool = True,
        max_synonyms: int = 20,
    ) -> None:
        self.language = language.lower()
        self.eps = eps
        self.use_wordnet = bool(use_wordnet)
        self.max_synonyms = int(max_synonyms)

        if spacy_model is None:
            if self.language not in self.LANG_TO_SPACY_MODEL:
                raise ValueError(f"No default spaCy model for '{self.language}'. Provide spacy_model.")
            spacy_model = self.LANG_TO_SPACY_MODEL[self.language]
        self.nlp = self._load_spacy_model(spacy_model)

        # If WordNet requested but unavailable, fall back safely.
        if self.use_wordnet and not _HAVE_WORDNET:
            print("[M_axis eq] WordNet not available; falling back to identity-only synonym sets.")
            self.use_wordnet = False

    def _load_spacy_model(self, model_name: str):
        try:
            return spacy.load(model_name)
        except Exception:
            print(f"[M_axis eq] spaCy model '{model_name}' not found; downloading...")
            spacy_download(model_name)
            return spacy.load(model_name)

    def _lemma_stream(self, text: str) -> List[Tuple[str, str, bool, bool]]:
        """
        Returns (lemma, coarse_pos, is_stop, is_blocked) per token.
        """
        doc = self.nlp(text)
        out: List[Tuple[str, str, bool, bool]] = []
        for tok in doc:
            if tok.is_space or tok.is_punct:
                continue
            if not any(ch.isalnum() for ch in tok.text):
                continue

            lemma = tok.lemma_.lower().strip()
            if lemma == "":
                continue

            coarse_pos = tok.pos_  # UPOS-style (NOUN/VERB/PROPN/...)
            is_stop = bool(tok.is_stop)

            # "blocked" = named entity OR number-like OR PROPN
            is_ent = tok.ent_type_ != ""
            is_num = _is_number_like(tok)
            is_blocked = bool(is_ent or is_num)

            out.append((lemma, coarse_pos, is_stop, is_blocked))
        return out

    def _count_vocab(self, items: List[Tuple[str, str, bool, bool]]) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for lemma, _, _, _ in items:
            c[lemma] = c.get(lemma, 0) + 1
        return c

    def _freq_prior(self, vocab: Set[str]) -> Dict[str, float]:
        # q(t) = freq(t) / sum freq(u)
        # using wordfreq's word_frequency, stabilized
        freqs = {t: max(word_frequency(t, self.language), self.eps) for t in vocab}
        Z = sum(freqs.values())
        if Z <= 0:
            # uniform fallback
            u = 1.0 / max(1, len(vocab))
            return {t: u for t in vocab}
        return {t: freqs[t] / Z for t in vocab}

    def _info_weight(self, q: float) -> float:
        # w(t) = -log q(t) (base does not matter for ranking)
        return -math.log(max(q, self.eps))

    def _wn_pos(self, coarse_pos: str):
        if not _HAVE_WORDNET:
            return None
        # Map spaCy POS to WordNet POS
        if coarse_pos == "NOUN" or coarse_pos == "PROPN":
            return wn.NOUN
        if coarse_pos == "VERB":
            return wn.VERB
        if coarse_pos == "ADJ":
            return wn.ADJ
        if coarse_pos == "ADV":
            return wn.ADV
        return None

    def _synset(self, lemma: str, coarse_pos: str, is_stop: bool, is_blocked: bool) -> Set[str]:
        # Regimes from the paper section:
        # stop/block => identity-only
        if is_stop or is_blocked or not self.use_wordnet:
            return {lemma}

        pos = self._wn_pos(coarse_pos)
        if pos is None:
            return {lemma}

        syns: Set[str] = {lemma}

        # WordNet synonyms
        try:
            for syn in wn.synsets(lemma, pos=pos):
                for l in syn.lemmas():
                    syns.add(l.name().replace("_", " ").lower())
                    # derivationally related forms
                    for d in l.derivationally_related_forms():
                        syns.add(d.name().replace("_", " ").lower())
                if len(syns) >= self.max_synonyms:
                    break
        except Exception:
            return {lemma}

        # Keep it bounded
        #if len(syns) > self.max_synonyms:
        #    syns = set(list(syns)[: self.max_synonyms])

        if len(syns) > self.max_synonyms:
            syns = set(sorted(syns)[: self.max_synonyms])
        return syns

    def score(self, comp: str, simp: str) -> float:
        C = self._lemma_stream(comp)
        S = self._lemma_stream(simp)

        cC = self._count_vocab(C)
        cS = self._count_vocab(S)

        VC = set(cC.keys())
        VS = set(cS.keys())

        if not VC and not VS:
            return 0.0
        if not VC or not VS:
            return 0.0

        qC = self._freq_prior(VC)
        qS = self._freq_prior(VS)

        wC = {t: self._info_weight(qC[t]) for t in VC}
        wS = {t: self._info_weight(qS[t]) for t in VS}

        # Build lemma->flags maps (use first occurrence; good enough)
        flagsC: Dict[str, Tuple[str, bool, bool]] = {}
        for lemma, pos, is_stop, is_blocked in C:
            flagsC.setdefault(lemma, (pos, is_stop, is_blocked))

        flagsS: Dict[str, Tuple[str, bool, bool]] = {}
        for lemma, pos, is_stop, is_blocked in S:
            flagsS.setdefault(lemma, (pos, is_stop, is_blocked))

        # Syn(t) computed using source-side flags when measuring source-view,
        # and target-side flags when measuring target-view (matches the draft idea).
        synC: Dict[str, Set[str]] = {}
        for t in VC:
            pos, st, bl = flagsC.get(t, ("NOUN", False, False))
            synC[t] = self._synset(t, pos, st, bl)
         #   print(f"[M_axis eq] Syn({t}): {synC[t]}"    )

        synS: Dict[str, Set[str]] = {}
        for t in VS:
            pos, st, bl = flagsS.get(t, ("NOUN", False, False))
            synS[t] = self._synset(t, pos, st, bl)
         #   print(f"[M_axis eq] Syn({t}): {synS[t]}"    )

        # Soft overlaps:
        # OV_C(t) = min{ c_C(t), sum_{u in Syn(t) ∩ V_S} c_S(u) }
        numP = 0.0
        denP = 0.0
        for t in VC:
            denom = cC[t] * wC[t]
            denP += denom
            overlap_count = 0
            for u in synC[t]:
                if u in VS:
                    overlap_count += cS[u]
            ov = min(cC[t], overlap_count)
            numP += ov * wC[t]

        P = (numP / denP) if denP > 0 else 0.0

        # OV_S(t) = min{ c_S(t), sum_{u in Syn(t) ∩ V_C} c_C(u) }
        numR = 0.0
        denR = 0.0
        for t in VS:
            denom = cS[t] * wS[t]
            denR += denom
            overlap_count = 0
            for u in synS[t]:
                if u in VC:
                    overlap_count += cC[u]
            ov = min(cS[t], overlap_count)
            numR += ov * wS[t]

        R = (numR / denR) if denR > 0 else 0.0

        if P > 0 and R > 0:
            return 2 * P * R / (P + R)
        return 0.0


# ============================= S_axis (equations) =============================

class SimplicityAxis_Equations:
    """
    Implements the paper’s S_axis equations:

    TLS: decreasing logistic on token length
    TFS: mean Zipf(word)/(k*len(word)) over alphabetic tokens
    TSS: per-sentence TLS_s / (MDL * CL), then Gibbs soft-min over sentences
    S(T) = avg-weighted TLS/TFS/TSS
    S_axis = S(T_S) / (S(T_S) + S(T_C) + eps)
    """

    CLAUSE_WEIGHTS = {
        "ccomp": 0.2,
        "xcomp": 0.1,
        "tcomp": 0.1,
        "advcl": 0.3,
        "acl": 0.4,
        "relcl": 0.4,
        "csubj": 0.5,
        "csubjpass": 0.5,
    }

    def __init__(
        self,
        nlp,  # reuse spaCy pipeline (same as M_axis)
        tau: float = 8.0,
        omega: float = 30.0,
        k_len: float = 2.0,
        # sentence-level TLS_s parameters for TSS
        tau_s: float = 5.0,
        omega_s: float = 18.0,
        beta_softmin: float = 0.2,
        w_len: float = 1/3,
        w_lex: float = 1/3,
        w_str: float = 1/3,
        eps: float = 1e-12,
        language: str = "en",
    ) -> None:
        self.nlp = nlp
        self.tau = float(tau)
        self.omega = float(omega)
        self.k_len = float(k_len)
        self.tau_s = float(tau_s)
        self.omega_s = float(omega_s)
        self.beta_softmin = float(beta_softmin)
        self.w_len = float(w_len)
        self.w_lex = float(w_lex)
        self.w_str = float(w_str)
        self.eps = float(eps)
        self.language = language

        # normalize weights
        s = self.w_len + self.w_lex + self.w_str
        if s <= 0:
            self.w_len = self.w_lex = self.w_str = 1/3
        else:
            self.w_len /= s
            self.w_lex /= s
            self.w_str /= s

    def _TLS(self, doc, tau: float, omega: float) -> float:
        x = 0
        for t in doc:
            if t.is_space or t.is_punct:
                continue
            if any(ch.isalnum() for ch in t.text):
                x += 1
        return 1.0 / (1.0 + math.exp((x - omega) / max(tau, 1e-9)))

    def _TFS(self, doc) -> float:
        toks = [t for t in doc if t.is_alpha]
        if not toks:
            return 0.0
        s = 0.0
        for t in toks:
            w = t.lower_
            z = zipf_frequency(w, self.language)
            s += z / (self.k_len * max(1, len(w)))
        return s / len(toks)

    def _MDL(self, sent) -> float:
        # mean dependency length excluding punct
        arcs = []
        for tok in sent:
            if tok.is_punct or tok.is_space:
                continue
            if tok.head is None:
                continue
            if tok.head.i == tok.i:
                continue
            # ignore arcs that point outside the sentence span
            if tok.head.i < sent.start or tok.head.i >= sent.end:
                continue
            arcs.append(abs(tok.i - tok.head.i))
        return sum(arcs) / len(arcs) if arcs else 0.0

    def _CL(self, sent) -> float:
        b = 0.5
        s = b
        for tok in sent:
            dep = tok.dep_
            if dep in self.CLAUSE_WEIGHTS:
                s += self.CLAUSE_WEIGHTS[dep]
        return s

    def _TSS_sentence(self, sent) -> float:
        tls_s = self._TLS(sent, self.tau_s, self.omega_s)
        mdl = self._MDL(sent)
        cl = self._CL(sent)
        denom = mdl * cl
        if denom <= 0:
            return 1.0
        return tls_s / denom

    def _TSS(self, doc) -> float:
        sents = _sentences(doc)
        vals = [self._TSS_sentence(s) for s in sents] if sents else []
        return _gibbs_softmin(vals, beta=self.beta_softmin) if vals else 0.0

    def S_text(self, text: str) -> float:
        doc = self.nlp(text)
        tls = self._TLS(doc, self.tau, self.omega)
        tfs = self._TFS(doc)
        tss = self._TSS(doc)
        return self.w_len * tls + self.w_lex * tfs + self.w_str * tss

    def score(self, comp: str, simp: str) -> float:
        Sc = self.S_text(comp)
        Ss = self.S_text(simp)
        return Ss / (Ss + Sc + self.eps)


# ============================= MREF (equations) =============================

class MREF:
    """
    Full "equation version" MREF:
      MREF = sqrt(wG*G^2 + wM*M^2 + wS*S^2)
    """

    def __init__(
        self,
        language: str = "en",
        axis_weights: Optional[AxisWeights] = None,
        # G_axis
        g_spacy_model: Optional[str] = None,
        g_pos_index_path: Optional[str] = None,
        g_pos_tagset: str = "ptb",
        max_tatoeba_sentences: Optional[int] = None,
        # M_axis
        m_spacy_model: Optional[str] = None,
        use_wordnet: bool = True,
    ) -> None:
        self.language = language.lower()
        self.weights = (axis_weights or AxisWeights()).normalized()

        self.M_axis = MeaningAxis_Equations(
            language=self.language,
            spacy_model=m_spacy_model,
            use_wordnet=use_wordnet,
        )
        # reuse same spaCy pipeline for S_axis
        self.S_axis = SimplicityAxis_Equations(self.M_axis.nlp, language=self.language)

        # Grammar axis can reuse the same spaCy model name if none given
        self.G_axis = GrammarAxis_Equations(
            language=self.language,
            spacy_model=g_spacy_model or (m_spacy_model),
            pos_tagset=g_pos_tagset,
            pos_index_path=g_pos_index_path,
            max_tatoeba_sentences=max_tatoeba_sentences,
        )
        

   def score(
        self,
        comp: str,
        simp: str,
        axis_weights: Optional[AxisWeights] = None,
    ) -> float:
        G = self.G_axis.score(comp, simp)
        M = self.M_axis.score(comp, simp)
        S = self.S_axis.score(comp, simp)
    
        w = (axis_weights or self.weights).normalized()
    
        return math.sqrt(
            w.w_G * (G ** 2) +
            w.w_M * (M ** 2) +
            w.w_S * (S ** 2)
        )

    def score_axes(self, comp: str, simp: str) -> Dict[str, float]:
        G = self.G_axis.score(comp, simp)
        M = self.M_axis.score(comp, simp)
        S = self.S_axis.score(comp, simp)
        w = self.weights
        mref = math.sqrt(w.w_G * (G ** 2) + w.w_M * (M ** 2) + w.w_S * (S ** 2))
    
        return {
            "G_axis": G,
            "M_axis": M,
            "S_axis": S,
            "MREFscore": mref,
        }

