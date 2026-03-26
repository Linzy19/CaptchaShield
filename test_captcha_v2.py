#!/usr/bin/env python3
"""
CaptchaShield VLM Evaluation v2 — High-Performance Batch Evaluation

Key performance improvements over v1:
  1. Per-image API calls are fired in parallel (model × question concurrently)
  2. Source image results are computed ONCE and reused across all attack methods
  3. ID and SDXL are evaluated independently with separate metrics + combined report
  4. JSON cache: skip already-evaluated images on re-run
  5. Progress bar with ETA

Architecture:
  For each gen_model (ID / SDXL):
    Phase 0: Evaluate source images (ONCE, shared baseline)
    Phase 1: Evaluate adv images for each attack method (in parallel across images)
    Phase 2: Compute per-method metrics + per-gen_model summary
  Final: Combined cross-model comparison report

Usage:
  python test_captcha_v2.py --mini_test                    # 3 samples, all VLMs
  python test_captcha_v2.py --num_images 50                # 50 samples
  python test_captcha_v2.py --mini_test --skip_gpt         # Save GPT cost
  python test_captcha_v2.py                                # Full run (1000 samples)
"""

import os
import re
import json
import base64
import glob
import time
import argparse
import threading
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import AzureOpenAI, OpenAI
from google import genai
from google.genai import types as genai_types

# API timeout in seconds (prevents hanging on slow/unresponsive endpoints)
# Can be overridden via --timeout command line argument
API_TIMEOUT = 10


# ==================== Constants ====================

GEMINI_CANDIDATE_MODELS = [
    "gemini-2.5-flash", "gemini-2.5-pro",
    "gemini-3-flash-preview", "gemini-3-pro-preview",
]
GLM_CANDIDATE_MODELS = ["glm-4v-flash", "GLM-4V-Plus", "glm-4v"]
GPT_CANDIDATE_MODELS = ["gpt-5.2"]

# Pricing table (USD per 1M tokens)
PRICING = {
    "gpt-5.2": {"input": 2.50, "output": 10.00, "currency": "USD"},
    "gemini-3-flash-preview": {"input": 0.10, "output": 0.40, "currency": "USD"},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60, "currency": "USD"},
    "glm-4v-flash": {"input": 0.00, "output": 0.00, "currency": "RMB"},
    "GLM-4V-Plus": {"input": 8.00, "output": 8.00, "currency": "RMB"},
}
USD_TO_RMB = 7.25

print_lock = threading.Lock()
def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

# ==================== Prompts ====================

Q1_PROMPT_EN = "Please identify the Chinese character shown in this image. Respond with only the single Chinese character, nothing else."
Q1_PROMPT_CN = "请识别图片中显示的中文字符。仅回答一个汉字，不要回答其他内容。"
Q2_PROMPT_EN = "Is there a clearly readable Chinese character in this image? Answer only 'yes' or 'no'."
Q2_PROMPT_CN = "图片中是否有一个可以清楚辨认的中文字符？仅回答yes或no。"
Q3_PROMPT_EN = "If this image contains a Chinese character, what character is it most likely? Respond with only one Chinese character, or 'none' if you cannot identify any character."
Q3_PROMPT_CN = "如果图片中包含一个汉字，你认为最可能是哪个字？仅回答一个汉字；如果完全无法识别，回答'无'。"


# ==================== Cost Tracker ====================

class CostTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._stats: Dict[str, Dict] = {}

    def record(self, model: str, prompt_tokens: int, completion_tokens: int,
               total_tokens: int, latency_sec: float, success: bool):
        with self._lock:
            if model not in self._stats:
                self._stats[model] = {"calls": 0, "success": 0, "fail": 0,
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                    "total_latency": 0.0}
            s = self._stats[model]
            s["calls"] += 1
            s["success" if success else "fail"] += 1
            s["prompt_tokens"] += prompt_tokens
            s["completion_tokens"] += completion_tokens
            s["total_tokens"] += total_tokens
            s["total_latency"] += latency_sec

    def get_cost_summary(self) -> Dict:
        with self._lock:
            summary = {"models": {}, "grand_total_usd": 0.0, "grand_total_rmb": 0.0}
            for model, s in self._stats.items():
                pricing = PRICING.get(model, {"input": 0, "output": 0, "currency": "USD"})
                input_cost = (s["prompt_tokens"] / 1_000_000) * pricing["input"]
                output_cost = (s["completion_tokens"] / 1_000_000) * pricing["output"]
                total_cost = input_cost + output_cost
                if pricing["currency"] == "RMB":
                    cost_rmb = total_cost; cost_usd = total_cost / USD_TO_RMB
                else:
                    cost_usd = total_cost; cost_rmb = total_cost * USD_TO_RMB
                ms = {"calls": s["calls"], "success": s["success"], "fail": s["fail"],
                    "prompt_tokens": s["prompt_tokens"], "completion_tokens": s["completion_tokens"],
                    "total_tokens": s["total_tokens"],
                    "avg_latency_sec": round(s["total_latency"] / max(s["calls"], 1), 3),
                    "cost_usd": round(cost_usd, 6), "cost_rmb": round(cost_rmb, 4)}
                summary["models"][model] = ms
                summary["grand_total_usd"] += cost_usd
                summary["grand_total_rmb"] += cost_rmb
            summary["grand_total_usd"] = round(summary["grand_total_usd"], 6)
            summary["grand_total_rmb"] = round(summary["grand_total_rmb"], 4)
            return summary

cost_tracker = CostTracker()


# ==================== Utility Functions ====================

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

def get_image_media_type(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".gif": "image/gif", ".webp": "image/webp"}.get(ext, "image/jpeg")

def get_char_label(image_path: str) -> str:
    try:
        base, _ = os.path.splitext(image_path)
        json_path = base + ".json"
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            anns = data.get("annotations", [])
            if anns and "label" in anns[0]:
                return anns[0]["label"]
        if "_adv" in base:
            json_path2 = base.replace("_adv", "") + ".json"
            if os.path.exists(json_path2):
                with open(json_path2, "r", encoding="utf-8") as f:
                    data = json.load(f)
                anns = data.get("annotations", [])
                if anns and "label" in anns[0]:
                    return anns[0]["label"]
        return "?"
    except Exception:
        return "?"

def is_single_cjk_char(text: str) -> bool:
    text = text.strip()
    if len(text) != 1:
        return False
    try:
        return "CJK" in unicodedata.name(text, "")
    except ValueError:
        cp = ord(text)
        return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
                0x20000 <= cp <= 0x2A6DF or 0xF900 <= cp <= 0xFAFF)

def parse_yes_no(response: str) -> str:
    if not response: return "unclear"
    resp = response.lower().strip().rstrip('.!').strip()
    if resp in ("yes", "是", "是的"): return "yes"
    if resp in ("no", "否", "不是", "不能", "没有", "无"): return "no"
    if "yes" in resp and "no" not in resp: return "yes"
    if "no" in resp and "yes" not in resp: return "no"
    return "unclear"

def parse_q1_response(response: str, ground_truth: str) -> Dict:
    if not response:
        return {"answer": "", "correct": False, "is_cjk": False, "is_refusal": True}
    answer = response.strip().strip('"\'`「」。.!').strip()
    is_cjk = is_single_cjk_char(answer)
    return {"answer": answer, "correct": (answer == ground_truth) if is_cjk else False,
            "is_cjk": is_cjk, "is_refusal": not is_cjk}

def parse_q3_response(response: str, ground_truth: str) -> Dict:
    if not response:
        return {"answer": "", "correct": False, "is_cjk": False, "is_none": True}
    answer = response.strip().strip('"\'`「」。.!').strip()
    if answer.lower() in ("none", "无", "n/a", "null", "没有", "不确定", "无法识别"):
        return {"answer": answer, "correct": False, "is_cjk": False, "is_none": True}
    is_cjk = is_single_cjk_char(answer)
    return {"answer": answer, "correct": (answer == ground_truth) if is_cjk else False,
            "is_cjk": is_cjk, "is_none": False}

def get_image_files(directory: str, limit: int = 0) -> List[str]:
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    paths = sorted(paths)
    return paths[:limit] if limit > 0 else paths

def match_source_for_adv(adv_path: str, source_dir: str) -> Optional[str]:
    adv_name = Path(adv_path).stem
    source_name = adv_name.replace("_adv", "")
    for ext in [".png", ".jpg", ".jpeg"]:
        source_path = os.path.join(source_dir, source_name + ext)
        if os.path.exists(source_path):
            return source_path
    return None

def discover_attack_configs(adv_base_dir: str) -> List[Dict]:
    configs = []
    images_dir = os.path.join(adv_base_dir, "images")
    if not os.path.isdir(images_dir):
        return configs
    for entry in sorted(os.listdir(images_dir)):
        method_dir = os.path.join(images_dir, entry)
        if os.path.isdir(method_dir):
            adv_images = get_image_files(method_dir)
            if adv_images:
                configs.append({"method_name": entry, "method_dir": method_dir, "num_images": len(adv_images)})
    return configs


# ==================== API Testers ====================

class GPT52Tester:
    def __init__(self, api_key, endpoint, deployment, api_version="2024-12-01-preview"):
        self.client = AzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=api_key)
        self.deployment = deployment
        self.model_name = "gpt-5.2"

    def analyze_image(self, image_path, prompt):
        t0 = time.time()
        try:
            b64 = encode_image_to_base64(image_path)
            mt = get_image_media_type(image_path)
            resp = self.client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}}
                ]}], max_completion_tokens=64, timeout=API_TIMEOUT)
            lat = time.time() - t0
            txt = resp.choices[0].message.content
            u = resp.usage
            pt, ct, tt = (u.prompt_tokens or 0, u.completion_tokens or 0, u.total_tokens or 0) if u else (0, 0, 0)
            cost_tracker.record(self.model_name, pt, ct, tt, lat, True)
            return {"success": True, "response": txt, "model": self.model_name,
                    "tokens": {"prompt": pt, "completion": ct, "total": tt}}
        except Exception as e:
            cost_tracker.record(self.model_name, 0, 0, 0, time.time() - t0, False)
            return {"success": False, "error": str(e), "model": self.model_name}

class GeminiTester:
    def __init__(self, api_key, model_name="gemini-3-flash-preview"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def analyze_image(self, image_path, prompt):
        t0 = time.time()
        try:
            with open(image_path, "rb") as f:
                img_data = f.read()
            mt = get_image_media_type(image_path)
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=[{"parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mt,
                                     "data": base64.standard_b64encode(img_data).decode("utf-8")}}
                ]}],
                config=genai_types.GenerateContentConfig(
                    http_options=genai_types.HttpOptions(timeout=API_TIMEOUT * 1000)  # milliseconds
                ))
            lat = time.time() - t0
            txt = resp.text
            pt, ct, tt = 0, 0, 0
            u = getattr(resp, 'usage_metadata', None)
            if u:
                pt = getattr(u, 'prompt_token_count', 0) or 0
                ct = getattr(u, 'candidates_token_count', 0) or 0
                tt = getattr(u, 'total_token_count', 0) or 0
            cost_tracker.record(self.model_name, pt, ct, tt, lat, True)
            return {"success": True, "response": txt, "model": self.model_name,
                    "tokens": {"prompt": pt, "completion": ct, "total": tt}}
        except Exception as e:
            cost_tracker.record(self.model_name, 0, 0, 0, time.time() - t0, False)
            return {"success": False, "error": str(e), "model": self.model_name}

class GLM4Tester:
    def __init__(self, api_key, model_name="glm-4v-flash"):
        self.client = OpenAI(api_key=api_key, base_url="https://open.bigmodel.cn/api/paas/v4/")
        self.model_name = model_name

    def analyze_image(self, image_path, prompt):
        t0 = time.time()
        try:
            b64 = encode_image_to_base64(image_path)
            mt = get_image_media_type(image_path)
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}}
                ]}], max_tokens=64, timeout=API_TIMEOUT)
            lat = time.time() - t0
            txt = resp.choices[0].message.content
            u = resp.usage
            pt, ct, tt = (u.prompt_tokens or 0, u.completion_tokens or 0, u.total_tokens or 0) if u else (0, 0, 0)
            cost_tracker.record(self.model_name, pt, ct, tt, lat, True)
            return {"success": True, "response": txt, "model": self.model_name,
                    "tokens": {"prompt": pt, "completion": ct, "total": tt}}
        except Exception as e:
            cost_tracker.record(self.model_name, 0, 0, 0, time.time() - t0, False)
            return {"success": False, "error": str(e), "model": self.model_name}


# ==================== Parallel API Caller ====================

def call_single_api(tester, image_path: str, prompt: str, max_retries: int = 4) -> dict:
    """Call a single VLM API with retry."""
    for attempt in range(max_retries + 1):
        result = tester.analyze_image(image_path, prompt)
        if result.get("success"):
            return result
        if attempt < max_retries:
            time.sleep(1.0 * (attempt + 1))
    return result

def evaluate_image_parallel(
    image_path: str, ground_truth: str,
    testers: Dict[str, Tuple], questions: List[str],
    max_retries: int = 4,
) -> Dict:
    """Evaluate one image: fire ALL (model × question) API calls in parallel.
    
    Uses its own internal ThreadPoolExecutor to avoid deadlock when called
    from within another executor's thread.
    """
    result = {"image_path": image_path, "ground_truth": ground_truth}
    tasks = []  # (q_id, model_key, tester, prompt)
    for q_id in questions:
        result[q_id] = {}
        for model_key, (tester, lang) in testers.items():
            if tester is None: continue
            if q_id == "q1": prompt = Q1_PROMPT_CN if lang == "cn" else Q1_PROMPT_EN
            elif q_id == "q2": prompt = Q2_PROMPT_CN if lang == "cn" else Q2_PROMPT_EN
            elif q_id == "q3": prompt = Q3_PROMPT_CN if lang == "cn" else Q3_PROMPT_EN
            else: continue
            tasks.append((q_id, model_key, tester, prompt))

    # Use a SEPARATE internal executor to avoid nested deadlock
    with ThreadPoolExecutor(max_workers=len(tasks)) as inner_executor:
        futures = {}
        for q_id, model_key, tester, prompt in tasks:
            fut = inner_executor.submit(call_single_api, tester, image_path, prompt, max_retries)
            futures[fut] = (q_id, model_key)

        for fut in as_completed(futures):
            q_id, model_key = futures[fut]
            try:
                api_result = fut.result()
            except Exception as e:
                api_result = {"success": False, "error": str(e)}
            response_text = api_result.get("response", "") if api_result.get("success") else ""
            if q_id == "q1": parsed = parse_q1_response(response_text, ground_truth)
            elif q_id == "q2": parsed = {"answer": parse_yes_no(response_text)}
            elif q_id == "q3": parsed = parse_q3_response(response_text, ground_truth)
            else: parsed = {}
            result[q_id][model_key] = {
                "success": api_result.get("success", False), "raw_response": response_text,
                "parsed": parsed, "tokens": api_result.get("tokens", {}),
                "error": api_result.get("error", ""),
            }
    return result


# ==================== Metrics ====================

def compute_metrics(results: List[Dict], questions: List[str], model_keys: List[str]) -> Dict:
    metrics = {}
    for q_id in questions:
        metrics[q_id] = {}
        for mk in model_keys:
            if q_id == "q1":
                total, correct, refusal, cjk_count = 0, 0, 0, 0
                for r in results:
                    data = r.get(q_id, {}).get(mk, {})
                    if not data or not data.get("success"): continue
                    p = data.get("parsed", {}); total += 1
                    if p.get("correct"): correct += 1
                    if p.get("is_refusal"): refusal += 1
                    if p.get("is_cjk"): cjk_count += 1
                ca = correct / total if total > 0 else 0
                metrics[q_id][mk] = {"total": total, "correct": correct,
                    "cjk_responses": cjk_count, "refusals": refusal,
                    "CA": round(ca * 100, 2), "ASR": round((1 - ca) * 100, 2),
                    "TFR": round(refusal / total * 100, 2) if total > 0 else 0}
            elif q_id == "q2":
                total, yes_count, no_count, unclear = 0, 0, 0, 0
                for r in results:
                    data = r.get(q_id, {}).get(mk, {})
                    if not data or not data.get("success"): continue
                    total += 1; ans = data.get("parsed", {}).get("answer", "unclear")
                    if ans == "yes": yes_count += 1
                    elif ans == "no": no_count += 1
                    else: unclear += 1
                metrics[q_id][mk] = {"total": total, "yes": yes_count, "no": no_count,
                    "unclear": unclear, "TVR": round(yes_count / total * 100, 2) if total > 0 else 0}
            elif q_id == "q3":
                total, correct, none_count, cjk_count = 0, 0, 0, 0
                for r in results:
                    data = r.get(q_id, {}).get(mk, {})
                    if not data or not data.get("success"): continue
                    p = data.get("parsed", {}); total += 1
                    if p.get("is_none"): none_count += 1
                    elif p.get("is_cjk"):
                        cjk_count += 1
                        if p.get("correct"): correct += 1
                ca = correct / total if total > 0 else 0
                metrics[q_id][mk] = {"total": total, "correct": correct,
                    "cjk_responses": cjk_count, "none_responses": none_count,
                    "CA": round(ca * 100, 2), "ASR": round((1 - ca) * 100, 2),
                    "none_rate": round(none_count / total * 100, 2) if total > 0 else 0}
    return metrics


# ==================== JSON Cache ====================

class ResultCache:
    """Thread-safe JSON cache to avoid re-calling APIs on re-run.
    
    Cache key: (image_path, question_id, model_key) -> result
    Stores results incrementally to disk.
    """

    def __init__(self, cache_dir: str):
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._cache: Dict[str, dict] = {}
        self._lock = __import__('threading').Lock()
        self._load_existing()

    def _cache_path(self) -> str:
        return os.path.join(self._cache_dir, "api_cache.json")

    def _load_existing(self):
        cp = self._cache_path()
        if os.path.exists(cp):
            try:
                with open(cp, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                safe_print(f"  📦 从缓存加载了 {len(self._cache)} 条结果: {cp}")
            except Exception:
                self._cache = {}

    def _make_key(self, image_path: str, q_id: str, model_key: str) -> str:
        # Use parent_dir/filename as key to avoid collision across attack methods
        # e.g. "aspl_eps0.05_steps200/mosaic_0006_image_0012_adv.png|q1|gpt52"
        p = Path(image_path)
        return f"{p.parent.name}/{p.name}|{q_id}|{model_key}"

    def get(self, image_path: str, q_id: str, model_key: str):
        key = self._make_key(image_path, q_id, model_key)
        with self._lock:
            return self._cache.get(key)

    def put(self, image_path: str, q_id: str, model_key: str, result: dict):
        key = self._make_key(image_path, q_id, model_key)
        with self._lock:
            self._cache[key] = result

    def save(self):
        """Persist cache to disk."""
        with self._lock:
            with open(self._cache_path(), "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False)

    def has_complete(self, image_path: str, questions: List[str], model_keys: List[str]) -> bool:
        """Check if all (q, model) combinations are cached for this image."""
        for q in questions:
            for mk in model_keys:
                if self.get(image_path, q, mk) is None:
                    return False
        return True

    def get_full_result(self, image_path: str, ground_truth: str,
                        questions: List[str], model_keys: List[str]) -> dict:
        """Reconstruct a full result dict from cache."""
        result = {"image_path": image_path, "ground_truth": ground_truth}
        for q in questions:
            result[q] = {}
            for mk in model_keys:
                cached = self.get(image_path, q, mk)
                if cached:
                    result[q][mk] = cached
        return result


# ==================== Batch Evaluator ====================

def evaluate_batch(
    image_paths: List[str],
    ground_truths: Dict[str, str],  # {image_path: char_label}
    testers: Dict[str, Tuple],
    questions: List[str],
    model_keys: List[str],
    cache: ResultCache,
    max_workers: int = 20,
    label: str = "eval",
    max_retries: int = 4,
) -> List[Dict]:
    """Evaluate a batch of images with full parallelism.
    
    Uses a shared ThreadPoolExecutor so that API calls across all images
    are multiplexed concurrently (not image-by-image serial).
    """
    results = []
    to_eval = []
    cached_count = 0

    # Separate cached vs. uncached
    for img_path in image_paths:
        gt = ground_truths.get(img_path, "?")
        if cache.has_complete(img_path, questions, model_keys):
            results.append(cache.get_full_result(img_path, gt, questions, model_keys))
            cached_count += 1
        else:
            to_eval.append(img_path)

    if cached_count > 0:
        safe_print(f"    📦 {cached_count} 张来自缓存, {len(to_eval)} 张待评测")

    if not to_eval:
        safe_print(f"    ✅ 全部 {cached_count} 张图片已有缓存!")
        return results

    total = len(to_eval)
    completed = 0
    start_time = time.time()

    # Use a ThreadPoolExecutor to process multiple images in parallel.
    # Each image's evaluate_image_parallel() creates its own INTERNAL executor
    # for API calls, avoiding nested-executor deadlock.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all images for evaluation
        futures = {}
        for img_path in to_eval:
            gt = ground_truths.get(img_path, "?")
            fut = executor.submit(
                evaluate_image_parallel,
                img_path, gt, testers, questions,
                max_retries
            )
            futures[fut] = img_path

        for fut in as_completed(futures):
            img_path = futures[fut]
            gt = ground_truths.get(img_path, "?")
            completed += 1

            try:
                result = fut.result()
                results.append(result)

                # Cache individual results
                for q_id in questions:
                    for mk in model_keys:
                        if mk in result.get(q_id, {}):
                            cache.put(img_path, q_id, mk, result[q_id][mk])

                # Progress with ETA
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (total - completed) if completed > 0 else 0
                q1_info = ""
                if "q1" in result:
                    for mk in model_keys:
                        if mk in result["q1"]:
                            p = result["q1"][mk].get("parsed", {})
                            q1_info += f" {mk}='{p.get('answer', '?')}'"
                safe_print(f"    [{label}] [{completed}/{total}] "
                           f"GT='{gt}'{q1_info}  ETA:{eta:.0f}s")

            except Exception as e:
                safe_print(f"    [{label}] [{completed}/{total}] ERROR: {e}")

    # Save cache incrementally
    cache.save()

    return results


# ==================== Report Generation ====================

def generate_per_method_report(
    gen_model: str, method_name: str,
    source_metrics: Dict, adv_metrics: Dict,
    source_count: int, adv_count: int,
    model_keys: List[str], questions: List[str],
) -> str:
    """Generate a text report for one gen_model + attack_method combination."""
    lines = []
    lines.append(f"  ┌─ {gen_model} / {method_name} ─")
    lines.append(f"  │  原始图片: {source_count} 张, 对抗图片: {adv_count} 张")

    if "q1" in questions:
        lines.append(f"  │  Q1 (汉字识别):")
        lines.append(f"  │  {'模型':>10} {'正确率CA↓':>12} {'攻击成功率ASR↑':>16} {'拒答率TFR':>12}")
        for label_prefix, metrics in [("原始", source_metrics), ("对抗", adv_metrics)]:
            for mk in model_keys:
                ms = metrics.get("q1", {}).get(mk, {})
                if ms:
                    lines.append(f"  │  {label_prefix}_{mk:>6}: "
                                 f"{ms.get('CA', 0):>7.2f}% {ms.get('ASR', 0):>7.2f}% {ms.get('TFR', 0):>7.2f}%")

    if "q2" in questions:
        lines.append(f"  │  Q2 (文字可读性):")
        lines.append(f"  │  {'模型':>10} {'可见率TVR':>12}")
        for label_prefix, metrics in [("原始", source_metrics), ("对抗", adv_metrics)]:
            for mk in model_keys:
                ms = metrics.get("q2", {}).get(mk, {})
                if ms:
                    lines.append(f"  │  {label_prefix}_{mk:>6}: {ms.get('TVR', 0):>7.2f}%")

    if "q3" in questions:
        lines.append(f"  │  Q3 (宽松识别):")
        lines.append(f"  │  {'模型':>10} {'正确率CA↓':>12} {'攻击成功率ASR↑':>16} {'无答率':>10}")
        for label_prefix, metrics in [("原始", source_metrics), ("对抗", adv_metrics)]:
            for mk in model_keys:
                ms = metrics.get("q3", {}).get(mk, {})
                if ms:
                    lines.append(f"  │  {label_prefix}_{mk:>6}: "
                                 f"{ms.get('CA', 0):>7.2f}% {ms.get('ASR', 0):>7.2f}% {ms.get('none_rate', 0):>7.2f}%")

    lines.append(f"  └{'─'*50}")
    return "\n".join(lines)


def generate_combined_report(
    all_results: Dict,  # {gen_model: {method_name: {source_m, adv_m, ...}}}
    model_keys: List[str],
    questions: List[str],
    cost_summary: Dict,
    timestamp: str,
    output_dir: str,
    total_elapsed: float,
) -> str:
    """Generate the final combined report with ID/SDXL separate + merged view."""
    lines = []
    lines.append(f"{'═'*80}")
    lines.append(f"  CaptchaShield VLM 评测报告 v2 — 综合汇总")
    lines.append(f"{'═'*80}")
    lines.append(f"  时间       : {timestamp}")
    lines.append(f"  耗时       : {total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分钟)")
    lines.append(f"  VLM模型    : {', '.join(model_keys)}")
    lines.append(f"  评测问题   : {', '.join(questions)}")
    lines.append(f"{'═'*80}")
    lines.append("")

    # === Per gen_model tables ===
    for gen_model in sorted(all_results.keys()):
        methods = all_results[gen_model]
        lines.append(f"{'─'*80}")
        lines.append(f"  📊 {gen_model} — 共 {len(methods)} 种攻击方法")
        lines.append(f"{'─'*80}")

        # Q1 table: ASR comparison across methods
        if "q1" in questions:
            lines.append("")
            lines.append(f"  Q1: 攻击成功率 (ASR↑) — 越高表示保护效果越好")
            header = f"  {'攻击方法':<35}"
            for mk in model_keys:
                header += f" {mk:>10}"
            lines.append(header)
            lines.append(f"  {'─'*35}" + f" {'─'*10}" * len(model_keys))

            # Source baseline (same for all methods in this gen_model)
            first_method = list(methods.values())[0]
            src_line = f"  {'[原始基线]':<35}"
            for mk in model_keys:
                ca = first_method["source_metrics"].get("q1", {}).get(mk, {}).get("CA", 0)
                src_line += f" {ca:>9.2f}%"
            lines.append(src_line)
            lines.append(f"  {'─'*35}" + f" {'─'*10}" * len(model_keys))

            for method_name in sorted(methods.keys()):
                m = methods[method_name]
                row = f"  {method_name:<35}"
                for mk in model_keys:
                    asr = m["adv_metrics"].get("q1", {}).get(mk, {}).get("ASR", 0)
                    row += f" {asr:>9.2f}%"
                lines.append(row)

        # Q2 table: TVR comparison
        if "q2" in questions:
            lines.append("")
            lines.append(f"  Q2: 文字可见率 (TVR↓) — 越低表示隐藏效果越好")
            header = f"  {'攻击方法':<35}"
            for mk in model_keys:
                header += f" {mk:>10}"
            lines.append(header)
            lines.append(f"  {'─'*35}" + f" {'─'*10}" * len(model_keys))

            src_line = f"  {'[原始基线]':<35}"
            for mk in model_keys:
                tvr = first_method["source_metrics"].get("q2", {}).get(mk, {}).get("TVR", 0)
                src_line += f" {tvr:>9.2f}%"
            lines.append(src_line)
            lines.append(f"  {'─'*35}" + f" {'─'*10}" * len(model_keys))

            for method_name in sorted(methods.keys()):
                m = methods[method_name]
                row = f"  {method_name:<35}"
                for mk in model_keys:
                    tvr = m["adv_metrics"].get("q2", {}).get(mk, {}).get("TVR", 0)
                    row += f" {tvr:>9.2f}%"
                lines.append(row)

        # Q3 table: ASR comparison (lenient recognition)
        if "q3" in questions:
            lines.append("")
            lines.append(f"  Q3: 宽松识别攻击成功率 (ASR↑) — 越高表示保护效果越好")
            header = f"  {'攻击方法':<35}"
            for mk in model_keys:
                header += f" {mk:>10}"
            lines.append(header)
            lines.append(f"  {'─'*35}" + f" {'─'*10}" * len(model_keys))

            src_line = f"  {'[原始基线]':<35}"
            for mk in model_keys:
                ca = first_method["source_metrics"].get("q3", {}).get(mk, {}).get("CA", 0)
                src_line += f" {ca:>9.2f}%"
            lines.append(src_line)
            lines.append(f"  {'─'*35}" + f" {'─'*10}" * len(model_keys))

            for method_name in sorted(methods.keys()):
                m = methods[method_name]
                row = f"  {method_name:<35}"
                for mk in model_keys:
                    asr = m["adv_metrics"].get("q3", {}).get(mk, {}).get("ASR", 0)
                    row += f" {asr:>9.2f}%"
                lines.append(row)

        lines.append("")

    # === Cross gen_model comparison (merged) ===
    if len(all_results) > 1 and "q1" in questions:
        lines.append(f"{'═'*80}")
        lines.append(f"  🔀 跨模型对比 (ID vs SDXL)")
        lines.append(f"{'═'*80}")
        # Collect all method names
        all_method_names = set()
        for gm in all_results.values():
            all_method_names.update(gm.keys())

        for method_name in sorted(all_method_names):
            lines.append(f"\n  攻击方法: {method_name}")
            header = f"  {'生成模型':<12}"
            for mk in model_keys:
                header += f" {mk+'_ASR':>12} {mk+'_TVR':>12}"
            lines.append(header)

            for gen_model in sorted(all_results.keys()):
                if method_name not in all_results[gen_model]:
                    continue
                m = all_results[gen_model][method_name]
                row = f"  {gen_model:<12}"
                for mk in model_keys:
                    asr = m["adv_metrics"].get("q1", {}).get(mk, {}).get("ASR", 0)
                    tvr_val = m["adv_metrics"].get("q2", {}).get(mk, {}).get("TVR", 0) if "q2" in questions else 0
                    row += f" {asr:>11.2f}% {tvr_val:>11.2f}%"
                lines.append(row)

    # === Cost summary ===
    lines.append(f"\n{'═'*80}")
    lines.append(f"  💰 费用统计")
    lines.append(f"{'═'*80}")
    lines.append(f"  {'模型':<25} {'调用次数':>8} {'成功':>5} {'失败':>5} "
                 f"{'输入Token':>10} {'输出Token':>10} {'美元':>10} {'人民币':>10}")
    for model, ms in cost_summary.get("models", {}).items():
        lines.append(f"  {model:<25} {ms['calls']:>6d} {ms['success']:>4d} {ms['fail']:>4d} "
                     f"{ms['prompt_tokens']:>9d} {ms['completion_tokens']:>9d} "
                     f"${ms['cost_usd']:>8.4f} ¥{ms['cost_rmb']:>8.4f}")
    lines.append(f"  {'合计':<25} {'':>6} {'':>4} {'':>4} {'':>9} {'':>9} "
                 f"${cost_summary['grand_total_usd']:>8.4f} ¥{cost_summary['grand_total_rmb']:>8.4f}")
    lines.append(f"{'═'*80}")

    report_text = "\n".join(lines)

    # Save
    txt_path = os.path.join(output_dir, f"combined_report_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    safe_print(f"\n📄 综合报告: {txt_path}")

    return report_text


# ==================== Main Entry ====================

def main():
    parser = argparse.ArgumentParser(
        description="CaptchaShield VLM Evaluation v2 — High-Performance Batch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_captcha_v2.py --mini_test                   # 3 samples, quick validation
  python test_captcha_v2.py --num_images 50               # 50 samples per method
  python test_captcha_v2.py --mini_test --skip_gpt        # Save GPT cost
  python test_captcha_v2.py                               # Full run (all samples)
  python test_captcha_v2.py --gen_models ID               # Only SD-ID
        """
    )

    # Data paths
    parser.add_argument("--id_source_dir", type=str,
                        default="/Users/lzy/Code_lzy/posion_attack/outputs/ID_2k/batch_test_20260306_142512",
                        help="SD-ID source图片目录")
    parser.add_argument("--id_adv_base", type=str,
                        default="/Users/lzy/Code_lzy/posion_attack/outputs/run_ID_full_20260310_223019",
                        help="SD-ID adv输出的base目录")
    parser.add_argument("--sdxl_source_dir", type=str,
                        default="/Users/lzy/Code_lzy/posion_attack/outputs/sdxl_2k/batch_test_20260306_162734",
                        help="SDXL source图片目录")
    parser.add_argument("--sdxl_adv_base", type=str,
                        default="/Users/lzy/Code_lzy/posion_attack/outputs/run_SDXL_full_20260311_101341",
                        help="SDXL adv输出的base目录")

    # Test mode
    parser.add_argument("--mini_test", action="store_true",
                        help="Mini-test模式: 每个配置仅测试3个样本")
    parser.add_argument("--num_images", type=int, default=0,
                        help="每个配置测试的图片数 (0=全部, --mini_test覆盖此值)")
    parser.add_argument("--questions", nargs="+", default=["q1", "q2", "q3"],
                        choices=["q1", "q2", "q3"],
                        help="要评测的问题 (默认: q1 q2 q3)")
    parser.add_argument("--gen_models", nargs="+", default=["ID", "SDXL"],
                        choices=["ID", "SDXL"],
                        help="要评测的生成模型 (默认: 两个都跑)")

    # VLM API keys
    parser.add_argument("--gpt_api_key", type=str,
                        default="XXXXX")
    parser.add_argument("--gpt_endpoint", type=str,
                        default="XXXXX")
    parser.add_argument("--gpt_deployment", type=str, default="XXXXX")
    parser.add_argument("--gemini_api_key", type=str,
                        default="XXXXX")
    parser.add_argument("--gemini_model", type=str, default="gemini-3-flash-preview",
                        help=f"Gemini模型, 候选: {GEMINI_CANDIDATE_MODELS}")
    parser.add_argument("--glm4_api_key", type=str,
                        default="XXXXX")
    parser.add_argument("--glm4_model", type=str, default="glm-4v-flash",
                        help=f"GLM-4V模型, 候选: {GLM_CANDIDATE_MODELS}")

    # Skip VLMs
    parser.add_argument("--skip_gpt", action="store_true", help="跳过GPT-5.2")
    parser.add_argument("--skip_gemini", action="store_true", help="跳过Gemini")
    parser.add_argument("--skip_glm4", action="store_true", help="跳过GLM-4V")

    # Performance
    parser.add_argument("--max_workers", type=int, default=20,
                        help="最大并行线程数 (默认: 20, 同时发出多个API请求)")
    parser.add_argument("--timeout", type=int, default=10,
                        help="API请求超时时间/秒 (默认: 10)")
    parser.add_argument("--max_retries", type=int, default=4,
                        help="API请求最大重试次数 (默认: 4)")
    parser.add_argument("--output_dir", type=str, default="",
                        help="输出目录 (默认: 自动)")

    args = parser.parse_args()

    # Update global API_TIMEOUT from command line argument
    global API_TIMEOUT
    API_TIMEOUT = args.timeout

    if args.mini_test:
        args.num_images = 3
    safe_print("🧪 Mini-test 模式: 每配置仅3个样本")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not args.output_dir:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "eval_results_v2", f"run_{timestamp}"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    safe_print(f"📁 输出目录: {args.output_dir}")

    # Initialize VLM testers
    testers: Dict[str, Tuple] = {}
    if not args.skip_gpt and args.gpt_api_key:
        try:
            testers["gpt52"] = (GPT52Tester(args.gpt_api_key, args.gpt_endpoint, args.gpt_deployment), "en")
            safe_print(f"✅ GPT-5.2 初始化成功")
        except Exception as e:
            safe_print(f"❌ GPT-5.2 初始化失败: {e}")

    if not args.skip_gemini and args.gemini_api_key:
        try:
            testers["gemini"] = (GeminiTester(args.gemini_api_key, args.gemini_model), "en")
            safe_print(f"✅ Gemini ({args.gemini_model}) 初始化成功")
        except Exception as e:
            safe_print(f"❌ Gemini 初始化失败: {e}")

    if not args.skip_glm4 and args.glm4_api_key:
        try:
            testers["glm4"] = (GLM4Tester(args.glm4_api_key, args.glm4_model), "cn")
            safe_print(f"✅ GLM-4V ({args.glm4_model}) 初始化成功")
        except Exception as e:
            safe_print(f"❌ GLM-4V 初始化失败: {e}")

    if not testers:
        safe_print("❌ 没有可用的VLM模型，退出。")
        return

    active_model_keys = list(testers.keys())
    safe_print(f"🤖 VLM模型: {active_model_keys}")
    safe_print(f"❓ 评测问题: {args.questions}\n")

    # Initialize cache
    cache = ResultCache(os.path.join(args.output_dir, "cache"))

    # Configure gen_model paths
    gen_configs = []
    if "ID" in args.gen_models:
        gen_configs.append({"name": "SD-ID", "source_dir": args.id_source_dir, "adv_base": args.id_adv_base})
    if "SDXL" in args.gen_models:
        gen_configs.append({"name": "SDXL", "source_dir": args.sdxl_source_dir, "adv_base": args.sdxl_adv_base})

    # ==================== Main Evaluation ====================
    all_results: Dict[str, Dict] = {}  # {gen_model: {method: {...}}}
    total_start = time.time()

    for gen_cfg in gen_configs:
        gen_name = gen_cfg["name"]
        source_dir = gen_cfg["source_dir"]
        adv_base = gen_cfg["adv_base"]

        safe_print(f"\n{'='*70}")
        safe_print(f"  🔬 {gen_name}")
        safe_print(f"{'='*70}")

        if not os.path.isdir(source_dir):
            safe_print(f"  ⚠️  原始目录不存在: {source_dir}")
            continue

        attack_configs = discover_attack_configs(adv_base)
        if not attack_configs:
            safe_print(f"  ⚠️  未发现攻击配置: {adv_base}")
            continue

        safe_print(f"  发现 {len(attack_configs)} 个攻击方法:")
        for ac in attack_configs:
            safe_print(f"    📂 {ac['method_name']} ({ac['num_images']} 张图片)")

        # ══════════════════════════════════════════════
        #  Phase 0: Evaluate SOURCE images (ONCE, shared)
        # ══════════════════════════════════════════════
        # Collect all unique source paths needed across all attack methods
        safe_print(f"\n  Phase 0: 收集 {gen_name} 的原始图片...")
        all_source_paths = set()
        method_pairs: Dict[str, list] = {}  # method_name -> list of (src, adv, gt)

        for ac in attack_configs:
            adv_images = get_image_files(ac["method_dir"], limit=args.num_images)
            pairs = []
            for adv_path in adv_images:
                src_path = match_source_for_adv(adv_path, source_dir)
                if src_path is None:
                    continue
                gt = get_char_label(src_path)
                if gt == "?":
                    gt = get_char_label(adv_path)
                pairs.append({"src": src_path, "adv": adv_path, "gt": gt})
                all_source_paths.add(src_path)
            method_pairs[ac["method_name"]] = pairs

        all_source_paths = sorted(all_source_paths)
        safe_print(f"  不重复原始图片: {len(all_source_paths)} 张")

        # Build ground truth map for source
        source_gt_map = {}
        for pairs_list in method_pairs.values():
            for p in pairs_list:
                source_gt_map[p["src"]] = p["gt"]

        # Evaluate source images ONCE
        source_questions = [q for q in args.questions if q in ("q1", "q2", "q3")]
        safe_print(f"\n  Phase 0: 评测 {len(all_source_paths)} 张原始图片 (共享基线)...")
        source_results = evaluate_batch(
            all_source_paths, source_gt_map, testers, source_questions,
            active_model_keys, cache,
            max_workers=args.max_workers, label=f"{gen_name}/src",
            max_retries=args.max_retries
        )
        source_metrics = compute_metrics(source_results, source_questions, active_model_keys)
        safe_print(f"  ✅ 原始基线评测完成: {len(source_results)} 张图片")

        # ══════════════════════════════════════════════
        #  Phase 1: Evaluate ADV images (per method)
        # ══════════════════════════════════════════════
        all_results[gen_name] = {}

        for ac in attack_configs:
            method_name = ac["method_name"]
            pairs = method_pairs.get(method_name, [])
            if not pairs:
                safe_print(f"\n  ⚠️ {method_name}: 没有匹配的原始-对抗图片对")
                continue

            safe_print(f"\n  Phase 1: {gen_name} / {method_name} ({len(pairs)} 张对抗图片)...")

            # Build adv image list and gt map
            adv_paths = [p["adv"] for p in pairs]
            adv_gt_map = {p["adv"]: p["gt"] for p in pairs}

            adv_results = evaluate_batch(
                adv_paths, adv_gt_map, testers, args.questions,
                active_model_keys, cache,
                max_workers=args.max_workers, label=f"{gen_name}/{method_name}",
                max_retries=args.max_retries
            )
            adv_metrics = compute_metrics(adv_results, args.questions, active_model_keys)

            # Build source metrics only for this method's source images
            method_src_paths = [p["src"] for p in pairs]
            method_src_results = [r for r in source_results if r["image_path"] in set(method_src_paths)]
            method_src_metrics = compute_metrics(method_src_results, source_questions, active_model_keys)

            # Store results
            all_results[gen_name][method_name] = {
                "source_metrics": method_src_metrics,
                "adv_metrics": adv_metrics,
                "source_count": len(method_src_results),
                "adv_count": len(adv_results),
            }

            # Save per-method JSON
            result_data = {
                "gen_model": gen_name, "attack_method": method_name,
                "timestamp": timestamp, "questions": args.questions,
                "source_count": len(method_src_results), "adv_count": len(adv_results),
                "source_metrics": method_src_metrics, "adv_metrics": adv_metrics,
                "source_results": method_src_results, "adv_results": adv_results,
            }
            json_path = os.path.join(args.output_dir, f"results_{gen_name}_{method_name}_{timestamp}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            safe_print(f"  💾 {json_path}")

            # Print per-method summary inline
            report = generate_per_method_report(
                gen_name, method_name, method_src_metrics, adv_metrics,
                len(method_src_results), len(adv_results), active_model_keys, args.questions
            )
            safe_print(report)

    # ==================== Final Report ====================
    total_elapsed = time.time() - total_start
    final_cost = cost_tracker.get_cost_summary()

    # Combined report
    report_text = generate_combined_report(
        all_results, active_model_keys, args.questions,
        final_cost, timestamp, args.output_dir, total_elapsed
    )
    safe_print(report_text)

    # Save final JSON
    final_json = {
        "timestamp": timestamp, "total_elapsed_sec": round(total_elapsed, 2),
        "questions": args.questions, "vlm_models": active_model_keys,
        "cost": final_cost,
        "results": {
            gm: {
                mn: {
                    "source_metrics": mv["source_metrics"],
                    "adv_metrics": mv["adv_metrics"],
                    "source_n": mv["source_count"],
                    "adv_n": mv["adv_count"],
                }
                for mn, mv in methods.items()
            }
            for gm, methods in all_results.items()
        },
    }
    final_json_path = os.path.join(args.output_dir, f"final_summary_{timestamp}.json")
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)

    safe_print(f"\n💰 总费用: ${final_cost['grand_total_usd']:.4f} / ¥{final_cost['grand_total_rmb']:.4f}")
    safe_print(f"⏱  总耗时: {total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分钟)")
    safe_print(f"📁 所有输出: {args.output_dir}")


if __name__ == "__main__":
    main()
