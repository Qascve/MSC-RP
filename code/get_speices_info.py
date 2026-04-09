# -*- coding: utf-8 -*-
"""
Batch classify diet type for taxa using Wikipedia API only.
Rules:
    0 = carnivore
    1 = herbivore
    2 = omnivore
    NA = unconfirmed / insufficient Wikipedia evidence

Features:
- Reads taxa from unique_taxon_names.txt
- Uses Wikipedia search + summary + wikitext + categories
- Extracts evidence text
- Writes CSV incrementally
- Supports resume if CSV already exists
- Never fabricates: if unclear, outputs NA

Author: ChatGPT
Date: 2026-04-09
"""

import csv
import os
import re
import time
import html
import json
import requests
from typing import List, Tuple, Dict, Optional

INPUT_FILE = "unique_taxon_names.txt"
OUTPUT_FILE = "taxon_diet_wikipedia.csv"
LOG_FILE = "taxon_diet_wikipedia.log"

REQUEST_SLEEP = 0.4
TIMEOUT = 25

HEADERS = {
    "User-Agent": "TaxonDietClassifier/1.0 (Wikipedia API research script; contact: local-use)"
}

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

CSV_FIELDS = [
    "taxon_name",
    "resolved_title",
    "page_url",
    "diet_type",
    "evidence_text",
    "evidence_source",
    "confidence",
    "status"
]

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ---------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------

def load_taxa(input_file: str) -> List[str]:
    taxa = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                taxa.append(name)
    return taxa

def load_processed(output_file: str) -> set:
    processed = set()
    if not os.path.exists(output_file):
        return processed
    with open(output_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(row["taxon_name"])
    return processed

def ensure_output_header(output_file: str) -> None:
    if not os.path.exists(output_file):
        with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()

def append_row(output_file: str, row: Dict[str, str]) -> None:
    with open(output_file, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)

# ---------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------

def safe_get(url: str, params: Optional[dict] = None) -> Optional[requests.Response]:
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        return r
    except Exception as e:
        log(f"HTTP error for {url} params={params}: {e}")
        return None

def request_json(url: str, params: Optional[dict] = None) -> Optional[dict]:
    r = safe_get(url, params=params)
    if r is None:
        return None
    if r.status_code != 200:
        log(f"Non-200 response {r.status_code} for {url} params={params}")
        return None
    try:
        return r.json()
    except Exception as e:
        log(f"JSON parse error for {url} params={params}: {e}")
        return None

# ---------------------------------------------------------------------
# Wikipedia API wrappers
# ---------------------------------------------------------------------

def search_wikipedia_title(query: str) -> Optional[str]:
    """
    Search a likely Wikipedia page title for the taxon.
    Prefer exact-looking results if available.
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 5,
        "format": "json"
    }
    data = request_json(WIKI_API, params=params)
    time.sleep(REQUEST_SLEEP)
    if not data:
        return None

    results = data.get("query", {}).get("search", [])
    if not results:
        return None

    qnorm = query.strip().lower()

    # Prefer exact title match
    for item in results:
        title = item.get("title", "")
        if title.lower() == qnorm:
            return title

    # Otherwise use top result
    return results[0].get("title")

def get_page_summary(title: str) -> Tuple[str, str]:
    """
    Returns (summary_text, status)
    """
    url = WIKI_SUMMARY_API.format(title.replace(" ", "_"))
    data = request_json(url)
    time.sleep(REQUEST_SLEEP)
    if not data:
        return "", "summary_unavailable"

    # REST summary may include "type": "disambiguation"
    if data.get("type") == "disambiguation":
        return "", "disambiguation"

    extract = data.get("extract", "") or ""
    return extract, "ok"

def get_page_wikitext(title: str) -> Tuple[str, str]:
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "formatversion": "2",
        "format": "json",
        "titles": title,
        "redirects": 1
    }
    data = request_json(WIKI_API, params=params)
    time.sleep(REQUEST_SLEEP)
    if not data:
        return "", "wikitext_unavailable"

    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return "", "no_page"

    page = pages[0]
    if page.get("missing"):
        return "", "missing"

    revisions = page.get("revisions", [])
    if not revisions:
        return "", "no_revisions"

    slots = revisions[0].get("slots", {})
    main = slots.get("main", {})
    content = main.get("content", "") or ""
    return content, "ok"

def get_page_categories(title: str) -> Tuple[List[str], str]:
    categories = []
    clcontinue = None

    while True:
        params = {
            "action": "query",
            "prop": "categories",
            "cllimit": "max",
            "format": "json",
            "titles": title,
            "redirects": 1
        }
        if clcontinue:
            params["clcontinue"] = clcontinue

        data = request_json(WIKI_API, params=params)
        time.sleep(REQUEST_SLEEP)

        if not data:
            return categories, "categories_unavailable"

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return categories, "no_page"

        page = next(iter(pages.values()))
        cats = page.get("categories", [])
        categories.extend([c.get("title", "") for c in cats if c.get("title")])

        if "continue" in data and "clcontinue" in data["continue"]:
            clcontinue = data["continue"]["clcontinue"]
        else:
            break

    return categories, "ok"

def get_resolved_title(title: str) -> str:
    """
    Resolve redirects / canonical page title.
    """
    params = {
        "action": "query",
        "titles": title,
        "redirects": 1,
        "format": "json"
    }
    data = request_json(WIKI_API, params=params)
    time.sleep(REQUEST_SLEEP)
    if not data:
        return title

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return title

    page = next(iter(pages.values()))
    return page.get("title", title)

# ---------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------

def strip_wikitext(text: str) -> str:
    """
    Convert raw wikitext to rough plain text for keyword scanning.
    This is intentionally simple and conservative.
    """
    if not text:
        return ""

    t = text

    # Remove comments
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.DOTALL)

    # Remove refs
    t = re.sub(r"<ref[^>]*>.*?</ref>", " ", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"<ref[^/]*/\s*>", " ", t, flags=re.IGNORECASE)

    # Remove templates {{...}} repeatedly
    for _ in range(8):
        new_t = re.sub(r"\{\{[^{}]*\}\}", " ", t)
        if new_t == t:
            break
        t = new_t

    # Convert wikilinks [[A|B]] -> B, [[A]] -> A
    t = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", t)
    t = re.sub(r"\[\[([^\]]+)\]\]", r"\1", t)

    # Remove external links [http... text] -> text
    t = re.sub(r"\[https?://[^\s\]]+\s([^\]]+)\]", r"\1", t)
    t = re.sub(r"\[https?://[^\]]+\]", " ", t)

    # Remove HTML tags
    t = re.sub(r"<[^>]+>", " ", t)

    # Remove headings markup
    t = re.sub(r"={2,}\s*(.*?)\s*={2,}", r"\n\1\n", t)

    # Remove table markup lines and pipes roughly
    t = re.sub(r"^\{\|.*?$", " ", t, flags=re.MULTILINE)
    t = re.sub(r"^\|[-+}].*$", " ", t, flags=re.MULTILINE)
    t = re.sub(r"^\!.*$", " ", t, flags=re.MULTILINE)
    t = re.sub(r"^\|.*$", " ", t, flags=re.MULTILINE)

    # Decode HTML entities
    t = html.unescape(t)

    # Normalize whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    t = t.strip()

    return t

def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    text = text.replace("\n", " ")
    # Simple sentence split
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

# ---------------------------------------------------------------------
# Classification rules
# ---------------------------------------------------------------------

# NOTE:
# We classify ONLY when there is explicit evidence from Wikipedia text/categories.
# Otherwise return NA.

OMNIVORE_PATTERNS = [
    r"\bomnivor(?:e|ous)\b",
    r"\bfeeds on both plant and animal\b",
    r"\bfeed on both plant and animal\b",
    r"\beats both plant and animal\b",
    r"\bdiet includes.*?\b(algae|plants?|seeds?|fruit|fruits|nectar|phytoplankton)\b.*?\b(insects?|fish|worms?|crustaceans?|zooplankton|invertebrates?|animals?)\b",
    r"\bdiet includes.*?\b(insects?|fish|worms?|crustaceans?|zooplankton|invertebrates?|animals?)\b.*?\b(algae|plants?|seeds?|fruit|fruits|nectar|phytoplankton)\b",
]

HERBIVORE_PATTERNS = [
    r"\bherbivor(?:e|ous)\b",
    r"\bgranivor(?:e|ous)\b",
    r"\bfrugivor(?:e|ous)\b",
    r"\bnectarivor(?:e|ous)\b",
    r"\bgraze(?:s|d)? on algae\b",
    r"\bfeed(?:s)? on algae\b",
    r"\bfeed(?:s)? on phytoplankton\b",
    r"\bfeed(?:s)? on plants\b",
    r"\bfeed(?:s)? primarily on plants\b",
    r"\bfeed(?:s)? mainly on plants\b",
    r"\bfeed(?:s)? on seagrass\b",
    r"\bfeed(?:s)? on leaves\b",
    r"\bfeed(?:s)? on seeds\b",
    r"\bfeed(?:s)? on fruit\b",
    r"\bfeed(?:s)? on fruits\b",
    r"\bfeed(?:s)? on nectar\b",
    r"\bphytoplankton\b"
]

CARNIVORE_PATTERNS = [
    r"\bcarnivor(?:e|ous)\b",
    r"\binsectivor(?:e|ous)\b",
    r"\bpiscivor(?:e|ous)\b",
    r"\bpredator\b",
    r"\bpredatory\b",
    r"\bprey(?:s|ing)? on\b",
    r"\bfeed(?:s)? on insects\b",
    r"\bfeed(?:s)? on insect larvae\b",
    r"\bfeed(?:s)? on crustaceans\b",
    r"\bfeed(?:s)? on worms\b",
    r"\bfeed(?:s)? on molluscs\b",
    r"\bfeed(?:s)? on mollusks\b",
    r"\bfeed(?:s)? on fish\b",
    r"\bfeed(?:s)? on small fish\b",
    r"\bfeed(?:s)? on invertebrates\b",
    r"\bfeed(?:s)? mainly on invertebrates\b",
    r"\bfeed(?:s)? primarily on invertebrates\b",
    r"\beats insects\b",
    r"\beats small fish\b",
    r"\beats crustaceans\b",
    r"\beats worms\b",
    r"\beats invertebrates\b",
    r"\bzooplankton\b",  # cautious: only used if no plant evidence around
]

CATEGORY_HERBIVORE = [
    "Herbivorous animals",
    "Herbivorous fish",
    "Herbivorous mammals",
    "Herbivorous reptiles",
    "Nectarivores",
    "Frugivores",
    "Granivores",
]

CATEGORY_CARNIVORE = [
    "Carnivorous animals",
    "Carnivorous fish",
    "Carnivorous mammals",
    "Carnivorous reptiles",
    "Insectivores",
    "Piscivorous animals",
]

CATEGORY_OMNIVORE = [
    "Omnivorous animals",
    "Omnivorous fish",
    "Omnivorous mammals",
    "Omnivorous reptiles",
    "Omnivorous birds",
]

def find_sentence_matches(sentences: List[str], patterns: List[str]) -> List[str]:
    hits = []
    for s in sentences:
        sl = s.lower()
        for pat in patterns:
            if re.search(pat, sl, flags=re.IGNORECASE):
                hits.append(s.strip())
                break
    return hits

def classify_from_categories(categories: List[str]) -> Tuple[Optional[str], str, str]:
    """
    Return (diet_type, evidence_text, confidence) or (None, "", "")
    """
    cat_text = " | ".join(categories)

    for key in CATEGORY_OMNIVORE:
        if key.lower() in cat_text.lower():
            return "2", f"Wikipedia category match: {key}", "high"

    for key in CATEGORY_HERBIVORE:
        if key.lower() in cat_text.lower():
            return "1", f"Wikipedia category match: {key}", "high"

    for key in CATEGORY_CARNIVORE:
        if key.lower() in cat_text.lower():
            return "0", f"Wikipedia category match: {key}", "high"

    return None, "", ""

def classify_from_text(summary_text: str, plain_text: str) -> Tuple[str, str, str, str]:
    """
    Return (diet_type, evidence_text, evidence_source, confidence)
    """
    combined = []
    if summary_text:
        combined.append(("summary", summary_text))
    if plain_text:
        combined.append(("wikitext", plain_text))

    # Collect sentences
    summary_sentences = split_sentences(summary_text)
    body_sentences = split_sentences(plain_text)

    # 1) Omnivore first
    om_sum = find_sentence_matches(summary_sentences, OMNIVORE_PATTERNS)
    om_body = find_sentence_matches(body_sentences, OMNIVORE_PATTERNS)
    if om_sum:
        return "2", om_sum[0], "summary", "high"
    if om_body:
        return "2", om_body[0], "wikitext", "high"

    # 2) Herbivore evidence
    hb_sum = find_sentence_matches(summary_sentences, HERBIVORE_PATTERNS)
    hb_body = find_sentence_matches(body_sentences, HERBIVORE_PATTERNS)

    # 3) Carnivore evidence
    ca_sum = find_sentence_matches(summary_sentences, CARNIVORE_PATTERNS)
    ca_body = find_sentence_matches(body_sentences, CARNIVORE_PATTERNS)

    # Resolve conflicts conservatively
    has_h = bool(hb_sum or hb_body)
    has_c = bool(ca_sum or ca_body)

    if has_h and has_c:
        # If both plant and animal evidence occur, mark omnivore
        ev = (hb_sum + hb_body + ca_sum + ca_body)[0]
        return "2", ev, "mixed_text_evidence", "medium"

    if hb_sum:
        return "1", hb_sum[0], "summary", "medium"
    if hb_body:
        return "1", hb_body[0], "wikitext", "medium"

    if ca_sum:
        return "0", ca_sum[0], "summary", "medium"
    if ca_body:
        return "0", ca_body[0], "wikitext", "medium"

    return "NA", "", "", "low"

# ---------------------------------------------------------------------
# Main taxon processing
# ---------------------------------------------------------------------

def build_page_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")

def process_taxon(taxon_name: str) -> Dict[str, str]:
    """
    Full workflow for one taxon.
    """
    result = {
        "taxon_name": taxon_name,
        "resolved_title": "",
        "page_url": "",
        "diet_type": "NA",
        "evidence_text": "",
        "evidence_source": "",
        "confidence": "low",
        "status": ""
    }

    # Step 1: search title
    searched_title = search_wikipedia_title(taxon_name)
    if not searched_title:
        result["status"] = "search_not_found"
        return result

    # Step 2: resolve title
    resolved_title = get_resolved_title(searched_title)
    result["resolved_title"] = resolved_title
    result["page_url"] = build_page_url(resolved_title)

    # Step 3: get summary
    summary_text, summary_status = get_page_summary(resolved_title)

    # Step 4: get wikitext
    raw_wikitext, wiki_status = get_page_wikitext(resolved_title)
    plain_text = strip_wikitext(raw_wikitext)

    # Step 5: get categories
    categories, cat_status = get_page_categories(resolved_title)

    # Step 6: classify from categories first if explicit
    diet_cat, ev_cat, conf_cat = classify_from_categories(categories)
    if diet_cat is not None:
        result["diet_type"] = diet_cat
        result["evidence_text"] = ev_cat
        result["evidence_source"] = "categories"
        result["confidence"] = conf_cat
        result["status"] = "ok"
        return result

    # Step 7: classify from text
    diet_text, evidence_text, evidence_source, conf_text = classify_from_text(summary_text, plain_text)
    result["diet_type"] = diet_text
    result["evidence_text"] = evidence_text
    result["evidence_source"] = evidence_source
    result["confidence"] = conf_text

    # Step 8: status
    status_bits = []
    if summary_status != "ok":
        status_bits.append(summary_status)
    if wiki_status != "ok":
        status_bits.append(wiki_status)
    if cat_status != "ok":
        status_bits.append(cat_status)

    if not status_bits:
        result["status"] = "ok" if diet_text != "NA" else "unconfirmed"
    else:
        result["status"] = ";".join(status_bits) if diet_text == "NA" else "ok_with_partial_fetch"

    return result

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    taxa = load_taxa(INPUT_FILE)
    processed = load_processed(OUTPUT_FILE)
    ensure_output_header(OUTPUT_FILE)

    total = len(taxa)
    remaining = [t for t in taxa if t not in processed]

    log(f"Loaded {total} taxa from {INPUT_FILE}")
    log(f"Already processed: {len(processed)}")
    log(f"Remaining: {len(remaining)}")

    for idx, taxon in enumerate(remaining, 1):
        log(f"Processing {idx}/{len(remaining)}: {taxon}")
        try:
            row = process_taxon(taxon)
            append_row(OUTPUT_FILE, row)
            log(
                f"Done: {taxon} -> diet_type={row['diet_type']}, "
                f"title={row['resolved_title']}, source={row['evidence_source']}, status={row['status']}"
            )
        except KeyboardInterrupt:
            log("Interrupted by user.")
            raise
        except Exception as e:
            log(f"Unhandled error for {taxon}: {e}")
            fallback = {
                "taxon_name": taxon,
                "resolved_title": "",
                "page_url": "",
                "diet_type": "NA",
                "evidence_text": "",
                "evidence_source": "",
                "confidence": "low",
                "status": f"error:{type(e).__name__}"
            }
            append_row(OUTPUT_FILE, fallback)

    log("All done.")

if __name__ == "__main__":
    main()