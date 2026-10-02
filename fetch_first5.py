"""
fetch_first5.py
---------------
Fetches the first 5 entries of the grey literature list (hard-coded below) and writes:

    data/documents.csv   one row per document (fetch status, word count, date found on the page)
    data/sentences.csv   one row per sentence, with section heading and bullet lead-in

Raw pages are saved in raw/ so you can open them and check what was captured.
If a site blocks the script, save the page from your browser as raw/<doc_id>.html
and run again with --cached: it will use your saved file instead of downloading.

Run:  python fetch_first5.py            (download)
      python fetch_first5.py --cached   (use files already in raw/)
"""

import argparse
import csv
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

DOCS = [
    dict(doc_id="NCSC_001", issuer="NCSC", year="2025", doc_type="blog post",
         title="New ETSI standard protects AI systems from evolving cyber threats",
         url="https://www.ncsc.gov.uk/blog-post/new-etsi-standard-protects-ai-systems-from-evolving-cyber-threats"),
    dict(doc_id="NCSC_002", issuer="NCSC", year="2025", doc_type="blog post",
         title="From bugs to bypasses: adapting vulnerability disclosure for AI safeguards",
         url="https://www.ncsc.gov.uk/blog-post/from-bugs-to-bypasses-adapting-vulnerability-disclosure-for-ai-safeguards"),
    dict(doc_id="NCSC_003", issuer="NCSC", year="2024", doc_type="blog post",
         title="Raising the cyber resilience of software 'at scale'",
         url="https://www.ncsc.gov.uk/blog-post/raising-cyber-resilience-software-at-scale"),
    dict(doc_id="NCSC_004", issuer="NCSC", year="2025", doc_type="report",
         title="Impact of AI on cyber threat from now to 2027",
         url="https://www.ncsc.gov.uk/report/impact-ai-cyber-threat-now-2027"),
    dict(doc_id="NCSC_005", issuer="NCSC", year="2024", doc_type="guidance (multi-page collection)",
         title="Machine learning principles",
         url="https://www.ncsc.gov.uk/collection/machine-learning-principles"),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
           "Accept-Language": "en-GB,en;q=0.9"}
DELAY = 2            # seconds between requests
MAX_SUBPAGES = 40    # safety cap for multi-page collections

RAW = Path("raw"); DATA = Path("data")

# Text that belongs to the website, not the document. Extraction stops at the first STOP heading.
STOP_AT = re.compile(r"^(was this article helpful|also see|share and print|topics|written by|published|"
                     r"related content|further reading|pages)\b", re.I)
SKIP = re.compile(r"download & print|please enable javascript|skip to main|copy link|share on|"
                  r"back to top|cookies?\b|privacy notice", re.I)


def get(url):
    r = requests.get(url, headers=HEADERS, timeout=40, allow_redirects=True)
    return r.status_code, r.text, r.url


def clean(t):
    return re.sub(r"\s+", " ", t.replace("\xa0", " ")).strip()


def split_sentences(text):
    # simple splitter: full stop / ? / ! followed by a capital letter or a quote
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'‘“(])", text)
    return [p.strip() for p in parts if p.strip()]


def extract_blocks(html):
    """Ordered (kind, text, section, lead_in) blocks from the page's main content."""
    soup = BeautifulSoup(html, "lxml")
    found = re.search(r"Published\s*(\d{1,2} \w+ \d{4})", soup.get_text(" "))
    date = found.group(1) if found else ""
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg"]):
        tag.decompose()
    root = soup.select_one("main article") or soup.select_one("main") or soup.body
    blocks, heads, last_para = [], {}, ""
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"]):
        if el.name == "li" and el.find("p"):
            continue
        text = clean(el.get_text(" "))
        if not text or SKIP.search(text) and len(text) < 150:
            continue
        if el.name.startswith("h"):
            if STOP_AT.match(text) and blocks:
                break
            lvl = int(el.name[1])
            heads = {k: v for k, v in heads.items() if k < lvl}
            heads[lvl] = text
            continue
        section = " > ".join(heads[k] for k in sorted(heads))
        if el.name == "li" or el.find_parent("li"):
            blocks.append(("bullet", text, section, last_para))
        else:
            blocks.append(("paragraph", text, section, ""))
            last_para = text
    return blocks, date


def subpages(html, base_url):
    """Links that sit under the collection's own path (e.g. /collection/machine-learning-principles/...)."""
    soup = BeautifulSoup(html, "lxml")
    path = urlparse(base_url).path.rstrip("/")
    seen, out = set(), []
    for a in soup.find_all("a", href=True):
        u = urljoin(base_url, a["href"]).split("#")[0].split("?")[0].rstrip("/")
        if urlparse(u).path.startswith(path + "/") and u not in seen:
            seen.add(u); out.append(u)
    return out[:MAX_SUBPAGES]


def main(cached=False):
    RAW.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
    doc_rows, sent_rows = [], []
    for d in DOCS:
        pages = [(d["url"], d["doc_id"])]
        status, note, date, words = "ok", "", "", 0
        page_texts = []
        i = 0
        while i < len(pages):
            url, fname = pages[i]
            fpath = RAW / f"{fname}.html"
            try:
                if cached and fpath.exists():
                    html, final, code = fpath.read_text(encoding="utf-8", errors="ignore"), url, 200
                else:
                    code, html, final = get(url)
                    fpath.write_text(html, encoding="utf-8")
                    time.sleep(DELAY)
                if code != 200:
                    note += f"{fname}: HTTP {code}; "
                    if i == 0:
                        status = "FAILED"
                    i += 1
                    continue
            except Exception as e:
                note += f"{fname}: {type(e).__name__}; "
                if i == 0:
                    status = "FAILED"
                i += 1
                continue
            if i == 0 and "/collection/" in url:          # multi-page guidance: follow its sub-pages
                for j, su in enumerate(subpages(html, final), start=1):
                    pages.append((su, f"{d['doc_id']}_p{j:02d}"))
            blocks, pdate = extract_blocks(html)
            date = date or pdate
            page_texts.append((url, blocks))
            i += 1

        n = 0
        for url, blocks in page_texts:
            for kind, text, section, lead_in in blocks:
                for s in split_sentences(text):
                    n += 1
                    words += len(s.split())
                    sent_rows.append(dict(doc_id=d["doc_id"], issuer=d["issuer"], title=d["title"],
                                          page_url=url, section=section, block_type=kind,
                                          lead_in=lead_in, sentence_no=n, sentence=s))
        if status == "ok" and n == 0:
            status, note = "EMPTY", note + "page fetched but no text extracted (check raw/ file); "
        doc_rows.append(dict(doc_id=d["doc_id"], issuer=d["issuer"], year=d["year"], doc_type=d["doc_type"],
                             title=d["title"], url=d["url"], status=status, pages_fetched=len(page_texts),
                             date_on_page=date, sentences=n, words=words, note=note))
        print(f"{d['doc_id']}  {status:7s} pages={len(page_texts):2d} sentences={n:4d} words={words:6d}  {note}")

    for name, rows in (("documents.csv", doc_rows), ("sentences.csv", sent_rows)):
        if rows:
            with open(DATA / name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
    print(f"\nWrote {DATA/'documents.csv'} and {DATA/'sentences.csv'} ({len(sent_rows)} sentences)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true", help="use pages already saved in raw/")
    main(ap.parse_args().cached)
