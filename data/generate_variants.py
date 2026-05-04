from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


LOW_PROMPT = """Edit the following text with only minor revisions:

"[TEXT]"

Requirements:
- Preserve the original meaning exactly
- Keep wording and sentence structure almost identical
- Only correct small grammatical issues or slightly awkward phrasing
- Do not rephrase sentences unnecessarily
- Do not change the length significantly
- Do not introduce new information
"""

MEDIUM_PROMPT = """Paraphrase the following text by rewording it:

"[TEXT]"

Requirements:
- Preserve the original meaning
- Reword sentences to improve clarity and flow
- Keep the overall structure similar, but allow moderate changes in wording
- Maintain a similar tone and approximate length
- Do not introduce new information
- Do not remove important details
"""

HIGH_PROMPT = """Rewrite the following text with noticeable structural changes:

"[TEXT]"

Requirements:
- Preserve the core meaning
- Change sentence structure and vocabulary more freely
- Improve clarity and coherence
- Maintain the same topic and approximate length
- Do not introduce new information
- Do not summarize or omit details
- Preserve all key information and examples
"""

SOURCE_METADATA_PROMPT = """You will be given an English text. Extract neutral generation metadata only.

The next model will use this metadata to write a new text. It must not receive the original text.

Return ONLY valid JSON (no markdown, no extra text) with exactly these keys:
- topic: a short neutral topic label (5-12 words)
- text_type: one of ["news_article", "report", "essay", "narrative", "dialogue", "mixed", "other"]
- target_length_words: integer word count target based on the source length
- structural_features: array of 2-6 structural observations, such as article/report/essay/narrative organization, sections, chronological order, dialogue presence, speaker turns, introduction/body/conclusion, or list-like structure
- content_requirements: array of 2-5 high-level content elements needed to stay on topic

Rules:
- Do not quote, paraphrase, or reuse distinctive phrases from the text.
- Do not include stylistic guidance such as tone, voice, vocabulary level, sentence length, formality, or author-like phrasing.
- Do not include named entities unless they are unavoidable for identifying the topic; prefer generic descriptions.
- Keep the metadata structural and topical, not stylistic.

TEXT:
"""

AI_GENERATION_PROMPT = """Write an original English text using only the metadata below.

Topic: {topic}
Text type: {text_type}
Approximate length: {target_length_words} words (±15%)

Structural requirements:
{structural_features}

Content requirements:
{content_requirements}

Constraints:
- Follow the requested text type and structure.
- Stay on the topic and cover the listed content requirements.
- Do not include citations or URLs.
- Avoid specific names, dates, statistics, or claims unless they are necessary for a generic version of the text type.
"""


def words(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))


def dict_reader_auto(f) -> csv.DictReader:
    sample = f.read(8192)
    f.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    return csv.DictReader(f, dialect=dialect)


def _json_object_from_model_output(raw: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object")
    return obj


def _as_string_list(value, fallback: List[str]) -> List[str]:
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if cleaned:
            return cleaned
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return fallback


def _bullet_lines(items: List[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def sample_humans(
    master_csv: Path,
    n: int,
    max_words: int,
    seed: int,
    require_source_corpus: str = "pan25",
) -> List[dict]:
    """
    Sample n human texts from the master CSV.
    - Restrict to source_corpus=pan25 by default (matches your plan: PAN fully-human).
    - Restrict to <= max_words.
    - Deterministic: stable shuffle using seed.
    """
    csv.field_size_limit(10_000_000)
    rows: List[dict] = []
    with master_csv.open("r", encoding="utf-8-sig", newline="") as f:
        r = dict_reader_auto(f)
        for row in r:
            if require_source_corpus and row.get("source_corpus") != require_source_corpus:
                continue
            if row.get("class") != "human" or row.get("rewrite_level") != "none":
                continue
            text = (row.get("text") or "").strip()
            if not text:
                continue
            if words(text) > max_words:
                continue
            rows.append(row)

    if len(rows) < n:
        raise SystemExit(f"Not enough rows after filtering: have {len(rows)}, need {n}.")

    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


@dataclass(frozen=True)
class Job:
    job_id: str
    source_id: str
    out_id: str
    out_class: str
    rewrite_level: str
    prompt: str
    meta: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {
            "job_id": self.job_id,
            "source_id": self.source_id,
            "out_id": self.out_id,
            "out_class": self.out_class,
            "rewrite_level": self.rewrite_level,
            "prompt": self.prompt,
        }
        if self.meta is not None:
            d["meta"] = self.meta
        return d


def make_jobs_for_sources(sources: Iterable[dict], *, include_hybrid: bool = True) -> List[Job]:
    jobs: List[Job] = []
    for row in sources:
        sid = row["id"]
        text = row["text"]

        if include_hybrid:
            variants = [
                ("hybrid", "low", LOW_PROMPT),
                ("hybrid", "medium", MEDIUM_PROMPT),
                ("hybrid", "high", HIGH_PROMPT),
            ]
            for out_class, level, tmpl in variants:
                out_id = f"{sid}_{level}"
                prompt = tmpl.replace("[TEXT]", text)
                job_id = f"job_{out_id}"
                jobs.append(
                    Job(
                        job_id=job_id,
                        source_id=sid,
                        out_id=out_id,
                        out_class=out_class,
                        rewrite_level=level,
                        prompt=prompt,
                    )
                )

    return jobs


def write_jobs_jsonl(jobs: List[Job], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j.to_dict(), ensure_ascii=False) + "\n")


def _http_json(url: str, payload: dict, headers: dict, timeout_s: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return json.loads(body)


def _chat_completions(
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout_s: int,
    messages: List[dict],
) -> str:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
    }
    resp = _http_json(endpoint, payload=payload, headers=headers, timeout_s=timeout_s)
    return resp["choices"][0]["message"]["content"]


def _extract_topic_meta(
    *,
    text: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout_s: int,
    max_retries: int,
    sleep_s: float,
) -> dict:
    """
    Model sees the source text once to extract neutral metadata only.
    The generation call receives only this metadata, never the source text.
    """
    text = (text or "").strip()

    def fallback(error: Optional[Exception] = None) -> dict:
        meta = {
            "topic": "General topic",
            "text_type": "other",
            "target_length_words": max(80, min(words(text), 1200)),
            "structural_features": ["Use a coherent beginning, middle, and ending."],
            "content_requirements": ["Stay on the broad topic without adding unsupported details."],
        }
        if error is not None:
            meta["extract_error"] = str(error)
        return meta

    if not text:
        return fallback()

    messages = [
        {"role": "system", "content": "You are a careful annotator. Output strictly valid JSON only."},
        {"role": "user", "content": SOURCE_METADATA_PROMPT + text},
    ]

    attempt = 0
    while True:
        attempt += 1
        try:
            raw = _chat_completions(
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=0.0,
                timeout_s=timeout_s,
                messages=messages,
            )
            obj = _json_object_from_model_output(raw)
            # Minimal normalization / safety.
            obj["topic"] = str(obj.get("topic") or "General topic").strip()
            obj["text_type"] = str(obj.get("text_type") or "other").strip()
            obj.setdefault("target_length_words", words(text))
            if not isinstance(obj.get("target_length_words"), int):
                obj["target_length_words"] = int(words(text))
            # Clamp target length to avoid extreme cases.
            obj["target_length_words"] = max(80, min(int(obj["target_length_words"]), 1200))
            obj["structural_features"] = _as_string_list(
                obj.get("structural_features"),
                ["Use a coherent beginning, middle, and ending."],
            )
            obj["content_requirements"] = _as_string_list(
                obj.get("content_requirements"),
                ["Stay on the broad topic without adding unsupported details."],
            )
            return obj
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError, ValueError) as e:
            if attempt > max_retries:
                return fallback(e)
            time.sleep(sleep_s * attempt)


def _ai_prompt_from_meta(meta: dict) -> str:
    return AI_GENERATION_PROMPT.format(
        topic=meta.get("topic", "the given topic"),
        text_type=meta.get("text_type", "other"),
        target_length_words=int(meta.get("target_length_words", 200)),
        structural_features=_bullet_lines(
            _as_string_list(meta.get("structural_features"), ["Use a coherent beginning, middle, and ending."])
        ),
        content_requirements=_bullet_lines(
            _as_string_list(meta.get("content_requirements"), ["Stay on the broad topic."])
        ),
    )


def run_ai_only_two_stage(
    *,
    sources_csv: Path,
    out_jsonl: Path,
    topic_model: str,
    gen_model: str,
    api_key: str,
    base_url: str,
    gen_temperature: float,
    timeout_s: int,
    max_retries: int,
    sleep_s: float,
) -> None:
    """
    Re-generates ONLY the `*_ai` texts using:
      1) topic extractor model that sees the human text -> topic metadata JSON
      2) generator model that sees ONLY the metadata -> AI-generated text

    Output schema matches the existing pipeline (job_results.jsonl-like),
    plus an extra `topic_meta` field for auditability.
    """
    csv.field_size_limit(10_000_000)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with sources_csv.open("r", encoding="utf-8-sig", newline="") as fsrc, out_jsonl.open("w", encoding="utf-8") as fout:
        r = dict_reader_auto(fsrc)
        for row in r:
            sid = row["id"]
            text = (row.get("text") or "").strip()

            topic_meta = _extract_topic_meta(
                text=text,
                base_url=base_url,
                api_key=api_key,
                model=topic_model,
                timeout_s=timeout_s,
                max_retries=max_retries,
                sleep_s=sleep_s,
            )

            prompt = _ai_prompt_from_meta(topic_meta)
            out_id = f"{sid}_ai"
            job_id = f"job_{out_id}"

            attempt = 0
            while True:
                attempt += 1
                try:
                    gen_text = _chat_completions(
                        base_url=base_url,
                        api_key=api_key,
                        model=gen_model,
                        temperature=gen_temperature,
                        timeout_s=timeout_s,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a careful writer. Follow constraints exactly. Do not add unsupported specifics.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                    )
                    record = {
                        "job_id": job_id,
                        "source_id": sid,
                        "out_id": out_id,
                        "out_class": "ai_generated",
                        "rewrite_level": "none",
                        "prompt": prompt,
                        "response_text": gen_text,
                        "model": gen_model,
                        "temperature": gen_temperature,
                        "base_url": base_url,
                        "topic_meta": topic_meta,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    break
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as e:
                    if attempt > max_retries:
                        record = {
                            "job_id": job_id,
                            "source_id": sid,
                            "out_id": out_id,
                            "out_class": "ai_generated",
                            "rewrite_level": "none",
                            "prompt": prompt,
                            "error": str(e),
                            "model": gen_model,
                            "temperature": gen_temperature,
                            "base_url": base_url,
                            "topic_meta": topic_meta,
                            "failed": True,
                        }
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        break
                    time.sleep(sleep_s * attempt)


def build_final_corpus_multi(
    *,
    sampled_sources_csv: Path,
    jobs_results_jsonls: List[Path],
    out_csv: Path,
) -> None:
    """
    Like `build_final_corpus`, but can merge multiple results files.
    If the same `out_id` appears multiple times, the **last file wins**
    (useful for regenerating AI-only while keeping existing hybrid outputs).
    """
    csv.field_size_limit(10_000_000)

    def _clean_generated_text(t: str) -> str:
        t = (t or "").strip()
        if len(t) >= 2 and t[0] == '"' and t[-1] == '"' and t.count('"') == 2:
            t = t[1:-1].strip()
        return t

    sources: Dict[str, dict] = {}
    with sampled_sources_csv.open("r", encoding="utf-8-sig", newline="") as f:
        r = dict_reader_auto(f)
        for row in r:
            sources[row["id"]] = row

    # Collect generated variants with de-dup by out_id (last wins)
    generated_by_out_id: Dict[str, dict] = {}
    for p in jobs_results_jsonls:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("failed"):
                    continue
                out_id = obj.get("out_id")
                if not out_id:
                    continue
                text = _clean_generated_text(obj.get("response_text") or "")
                if not text:
                    continue
                generated_by_out_id[out_id] = obj

    rows_out: List[dict] = []
    for _, row in sources.items():
        rows_out.append(
            {
                "id": row["id"],
                "source_id": row["id"],
                "class": "human",
                "rewrite_level": "none",
                "text": row["text"],
                "source_corpus": row.get("source_corpus", ""),
                "source_dataset": row.get("source_dataset", ""),
            }
        )

    for out_id, obj in generated_by_out_id.items():
        sid = obj["source_id"]
        meta = sources.get(sid, {})
        rows_out.append(
            {
                "id": out_id,
                "source_id": sid,
                "class": obj["out_class"],
                "rewrite_level": obj["rewrite_level"],
                "text": _clean_generated_text(obj.get("response_text") or ""),
                "source_corpus": meta.get("source_corpus", ""),
                "source_dataset": meta.get("source_dataset", ""),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "source_id", "class", "rewrite_level", "text", "source_corpus", "source_dataset"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        w.writeheader()
        w.writerows(rows_out)


def run_jobs_openai_compatible(
    jobs_path: Path,
    out_jsonl: Path,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    timeout_s: int,
    max_retries: int,
    sleep_s: float,
) -> None:
    """
    Runs jobs against an OpenAI-compatible Chat Completions endpoint.

    Env defaults:
    - OPENAI_API_KEY
    - OPENAI_BASE_URL (default: https://api.openai.com/v1)
    """
    endpoint = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jobs_path.open("r", encoding="utf-8") as fin, out_jsonl.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            job = json.loads(line)
            prompt = job["prompt"]
            payload = {
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": "You are a careful writer. Follow the user's constraints exactly."},
                    {"role": "user", "content": prompt},
                ],
            }

            attempt = 0
            while True:
                attempt += 1
                try:
                    resp = _http_json(endpoint, payload=payload, headers=headers, timeout_s=timeout_s)
                    text = resp["choices"][0]["message"]["content"]
                    record = {
                        **job,
                        "response_text": text,
                        "model": model,
                        "temperature": temperature,
                        "base_url": base_url,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    break
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as e:
                    if attempt > max_retries:
                        record = {
                            **job,
                            "error": str(e),
                            "model": model,
                            "temperature": temperature,
                            "base_url": base_url,
                            "failed": True,
                        }
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        break
                    time.sleep(sleep_s * attempt)


def build_final_corpus(
    sampled_sources_csv: Path,
    jobs_results_jsonl: Path,
    out_csv: Path,
) -> None:
    """
    Combines:
    - the sampled human sources CSV (id/source_id/class/rewrite_level/text + metadata)
    - the job results (hybrid low/med/high + ai_generated)

    into one dataset compatible with your existing pipeline.
    """
    csv.field_size_limit(10_000_000)

    def _clean_generated_text(t: str) -> str:
        """
        Make generated text more CSV-friendly without changing content.

        Some models wrap the entire response in a single leading/trailing quote
        (often when the prompt itself shows the source text inside quotes).
        This is harmless, but when combined with multiline text it can confuse
        downstream CSV readers. We only strip an *outer* wrapper quote when it
        looks like the model added it (exactly one leading and one trailing).
        """
        t = (t or "").strip()
        if len(t) >= 2 and t[0] == '"' and t[-1] == '"' and t.count('"') == 2:
            t = t[1:-1].strip()
        return t

    # Load sources
    sources: Dict[str, dict] = {}
    with sampled_sources_csv.open("r", encoding="utf-8-sig", newline="") as f:
        r = dict_reader_auto(f)
        for row in r:
            sources[row["id"]] = row

    rows_out: List[dict] = []
    # Add human sources first
    for sid, row in sources.items():
        rows_out.append(
            {
                "id": row["id"],
                "source_id": row["id"],
                "class": "human",
                "rewrite_level": "none",
                "text": row["text"],
                "source_corpus": row.get("source_corpus", ""),
                "source_dataset": row.get("source_dataset", ""),
            }
        )

    # Add generated variants
    with jobs_results_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("failed"):
                continue
            text = _clean_generated_text(obj.get("response_text") or "")
            if not text:
                continue
            sid = obj["source_id"]
            meta = sources.get(sid, {})
            rows_out.append(
                {
                    "id": obj["out_id"],
                    "source_id": sid,
                    "class": obj["out_class"],
                    "rewrite_level": obj["rewrite_level"],
                    "text": text,
                    "source_corpus": meta.get("source_corpus", ""),
                    "source_dataset": meta.get("source_dataset", ""),
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "source_id",
        "class",
        "rewrite_level",
        "text",
        "source_corpus",
        "source_dataset",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        # Be maximally compatible with strict / naive CSV readers:
        # - quote every field
        # - keep '\n' as the only line terminator
        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        w.writeheader()
        w.writerows(rows_out)


def write_sample_sources_csv(sources: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(sources[0].keys())
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=";",
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        w.writeheader()
        w.writerows(sources)


def write_texts_to_folder(
    *,
    human_sources: List[dict],
    job_results_jsonl: Optional[Path],
    texts_root: Path,
    ai_model_folder_name: str,
    write_human: bool = True,
    write_hybrid: bool = True,
    write_ai: bool = True,
) -> None:
    """
    Write texts into `data/texts/`-style folders:
      - human/{id}.txt
      - hybrid_low/{id}_low.txt, hybrid_medium/{id}_medium.txt, hybrid_high/{id}_high.txt
      - ai_generated/models/{MODEL}/{id}_ai.txt
    """
    texts_root.mkdir(parents=True, exist_ok=True)
    (texts_root / "human").mkdir(parents=True, exist_ok=True)
    (texts_root / "hybrid_low").mkdir(parents=True, exist_ok=True)
    (texts_root / "hybrid_medium").mkdir(parents=True, exist_ok=True)
    (texts_root / "hybrid_high").mkdir(parents=True, exist_ok=True)

    ai_root = texts_root / "ai_generated" / "models" / ai_model_folder_name
    ai_root.mkdir(parents=True, exist_ok=True)

    def _write(path: Path, text: str) -> None:
        path.write_text(text.strip() + "\n", encoding="utf-8")

    # Human
    if write_human:
        for row in human_sources:
            sid = row["id"]
            _write(texts_root / "human" / f"{sid}.txt", row["text"])

    if not job_results_jsonl or not job_results_jsonl.exists():
        return

    # Generated
    with job_results_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("failed"):
                continue
            out_id = obj["out_id"]
            level = obj["rewrite_level"]
            out_class = obj["out_class"]
            text = (obj.get("response_text") or "").strip()
            if not text:
                continue

            if out_class == "hybrid" and write_hybrid:
                if level == "low":
                    _write(texts_root / "hybrid_low" / f"{out_id}.txt", text)
                elif level == "medium":
                    _write(texts_root / "hybrid_medium" / f"{out_id}.txt", text)
                elif level == "high":
                    _write(texts_root / "hybrid_high" / f"{out_id}.txt", text)
            elif out_class == "ai_generated" and write_ai:
                _write(ai_root / f"{out_id}.txt", text)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample PAN25 human texts and generate hybrid/AI variants via API.")
    ap.add_argument(
        "--master",
        type=Path,
        default=Path("data/master_human_texts_pan25.csv"),
        help="Master CSV produced by build_master_dataset.py (PAN25 fully human)",
    )
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-words", type=int, default=500)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out-dir", type=Path, default=Path("data/generated"))
    ap.add_argument("--model", type=str, default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    ap.add_argument(
        "--topic-model",
        type=str,
        default=os.environ.get("OPENAI_TOPIC_MODEL", ""),
        help="Model used to extract topic metadata (defaults to --model if empty).",
    )
    ap.add_argument("--temperature", type=float, default=float(os.environ.get("OPENAI_TEMPERATURE", "0.4")))
    ap.add_argument("--timeout-s", type=int, default=90)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--sleep-s", type=float, default=2.0)
    ap.add_argument("--base-url", type=str, default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--no-run", action="store_true", help="Only write jobs; do not call the API.")
    ap.add_argument(
        "--ai-only",
        action="store_true",
        help="Regenerate only ai_generated outputs using a 2-stage (topic->generate) pipeline; keep existing hybrid outputs.",
    )
    ap.add_argument(
        "--hybrid-only",
        action="store_true",
        help="Generate only hybrid rewrite variants; skip metadata-only AI generation.",
    )
    ap.add_argument(
        "--texts-root",
        type=Path,
        default=Path("data/texts"),
        help="Root folder to save texts into (human/hybrid_*/ai_generated/...)",
    )
    args = ap.parse_args()

    if not args.master.exists():
        raise SystemExit(f"Master CSV not found: {args.master}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sampled_sources_csv = out_dir / "sampled_sources.csv"
    if args.ai_only and sampled_sources_csv.exists():
        # Reuse the exact same sampled set as the original run.
        sources = []
        with sampled_sources_csv.open("r", encoding="utf-8-sig", newline="") as f:
            r = dict_reader_auto(f)
            for row in r:
                sources.append(row)
    else:
        sources = sample_humans(args.master, n=args.n, max_words=args.max_words, seed=args.seed)
        write_sample_sources_csv(sources, sampled_sources_csv)

    jobs_path = out_dir / "jobs.jsonl"
    results_path = out_dir / "job_results.jsonl"
    ai_results_path = out_dir / "job_results_ai.jsonl"

    topic_model = (args.topic_model or "").strip() or args.model

    if not args.ai_only:
        jobs = make_jobs_for_sources(sources, include_hybrid=True)
        write_jobs_jsonl(jobs, jobs_path)

        if not args.no_run:
            if not args.api_key.strip():
                raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
            run_jobs_openai_compatible(
                jobs_path=jobs_path,
                out_jsonl=results_path,
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
                temperature=args.temperature,
                timeout_s=args.timeout_s,
                max_retries=args.max_retries,
                sleep_s=args.sleep_s,
            )
            if not args.hybrid_only:
                run_ai_only_two_stage(
                    sources_csv=sampled_sources_csv,
                    out_jsonl=ai_results_path,
                    topic_model=topic_model,
                    gen_model=args.model,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    gen_temperature=args.temperature,
                    timeout_s=args.timeout_s,
                    max_retries=args.max_retries,
                    sleep_s=args.sleep_s,
                )
    else:
        # AI-only regeneration path (2-stage).
        if args.hybrid_only:
            raise SystemExit("Use either --ai-only or --hybrid-only, not both.")
        if args.no_run:
            raise SystemExit("--ai-only requires running the API (remove --no-run).")
        if not args.api_key.strip():
            raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

        run_ai_only_two_stage(
            sources_csv=sampled_sources_csv,
            out_jsonl=ai_results_path,
            topic_model=topic_model,
            gen_model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            gen_temperature=args.temperature,
            timeout_s=args.timeout_s,
            max_retries=args.max_retries,
            sleep_s=args.sleep_s,
        )

    if args.no_run:
        print(f"Wrote sources to {sampled_sources_csv} and hybrid jobs to {jobs_path}.")
        return

    # Always save the sampled human texts to the shared folder structure.
    # Save generated variants only if results exist.
    ai_model_folder_name = args.model.replace("/", "_").replace("\\", "_").strip() or "model"
    if args.ai_only:
        write_texts_to_folder(
            human_sources=sources,
            job_results_jsonl=ai_results_path if ai_results_path.exists() else None,
            texts_root=args.texts_root,
            ai_model_folder_name=ai_model_folder_name,
            write_human=False,
            write_hybrid=False,
            write_ai=True,
        )
    else:
        write_texts_to_folder(
            human_sources=sources,
            job_results_jsonl=results_path if results_path.exists() else None,
            texts_root=args.texts_root,
            ai_model_folder_name=ai_model_folder_name,
            write_human=True,
            write_hybrid=True,
            write_ai=False,
        )
        if ai_results_path.exists():
            write_texts_to_folder(
                human_sources=sources,
                job_results_jsonl=ai_results_path,
                texts_root=args.texts_root,
                ai_model_folder_name=ai_model_folder_name,
                write_human=False,
                write_hybrid=False,
                write_ai=True,
            )

    final_csv = out_dir / "corpus_100_human_plus_variants.csv"
    if args.ai_only:
        # Merge existing hybrid results (if present) with newly generated AI results.
        inputs: List[Path] = []
        if results_path.exists():
            inputs.append(results_path)
        if ai_results_path.exists():
            inputs.append(ai_results_path)
        if inputs:
            build_final_corpus_multi(sampled_sources_csv=sampled_sources_csv, jobs_results_jsonls=inputs, out_csv=final_csv)
            print(f"Wrote final corpus to {final_csv}")
        else:
            print(f"Wrote AI-only results to {ai_results_path}")
    else:
        inputs = []
        if results_path.exists():
            inputs.append(results_path)
        if ai_results_path.exists():
            inputs.append(ai_results_path)
        if inputs:
            build_final_corpus_multi(sampled_sources_csv=sampled_sources_csv, jobs_results_jsonls=inputs, out_csv=final_csv)
            print(f"Wrote final corpus to {final_csv}")
        else:
            print(f"Wrote sources to {sampled_sources_csv} and jobs to {jobs_path}.")


if __name__ == "__main__":
    main()

