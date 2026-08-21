#!/usr/bin/env python3
"""Stage 2.5 deterministic checks for The_Agentic_Archive_v17_English.md
1) Ghost-citation check (Phase A3): in-text cites vs reference list
2) Duplicated reference entries
3) Duplicated sentences in body text
4) In-text year vs reference-list year mismatches
"""
import os, re, sys, json
from collections import Counter

PAPER = os.path.expanduser("~/Documents/My Projects/Things to Do/Articles/CNIR/FINAL-PAPER-CNIR/01_paper/The_Agentic_Archive_v17_English.md")
text = open(PAPER, encoding="utf-8").read()

# split body / references
idx = text.index("## References")
body, refs = text[:idx], text[idx:]

# ---- parse reference entries: blocks separated by blank lines ----
entries = []
for block in re.split(r"\n\s*\n", refs.split("## References", 1)[1].strip()):
    block = block.strip()
    if not block:
        continue
    first = block
    m = re.match(r"([A-ZÀ-ÿȘșȚț][A-Za-zÀ-ÿșȚț\.\- ]+?),\s", first)
    surname = m.group(1) if m else first[:30]
    ym = re.search(r"\b(19|20)\d{2}\b", first)
    year = ym.group(0) if ym else None
    entries.append({"surname": surname.strip(), "year": year, "text": first[:160]})

print(f"== REFERENCE ENTRIES: {len(entries)}")

# duplicates in reference list
seen = Counter(e["surname"] + " / " + str(e["year"]) for e in entries)
print("\n-- DUPLICATE (surname,year) pairs in reference list:")
for k, v in seen.items():
    if v > 1:
        print(f"  x{v}  {k}")

# exact duplicate entries
exact = Counter(e["text"] for e in entries)
print("-- EXACT duplicate first-lines:")
for k, v in exact.items():
    if v > 1:
        print(f"  x{v}  {k}")

# ---- in-text citations ----
cites = []
for m in re.finditer(r"\(([^()]{2,80}?)\)", body):
    s = m.group(1)
    if re.match(r"^[A-Z][A-Za-zà-ÿȘș\.\,\s\-]+,\s*(19|20)\d{2}[a-z]?$", s.strip()):
        cites.append(("paren", s.strip()))
# narrative: Name (Year)  /  Name's (Year)
for m in re.finditer(r"\b([A-Z][a-zà-ÿșȘ]{2,12})[^.()]{0,40}?[\.\s]?\((19|20)\d{2}([a-z])?\)", body):
    yr = m.group(0)[m.group(0).index("(")+1:m.group(0).index(")")]
    cites.append(("narr", f"{m.group(1)}, {yr}"))

print(f"\n== IN-TEXT CITATION TOKENS: {len(cites)}")

# normalize surnames
def surnames_of(c):
    return re.findall(r"[A-Z][a-zà-ÿȘș\-]+", c.split(",")[0])

ref_surnames = {}
for e in entries:
    ref_surnames[e["surname"]] = e["year"]

print("\n-- REFERENCE LIST ENTRIES (surname / year):")
for e in entries:
    print(f"  {e['surname']:40s} {e['year']}")

# ---- in-text year vs reference year ----
print("\n-- YEAR MISMATCHES (in-text year differs from reference-list year):")
for kind, c in set(cites):
    yr = re.findall(r"(19|20)\d{2}[a-z]?", c)
    yr = yr[0] if yr else None
    names = re.findall(r"[A-Z][a-zà-ÿȘș\-]+", c.split(",")[0])
    for n in names:
        hits = [e for e in entries if e["surname"].lower().startswith(n.lower()) or n.lower() in e["surname"].lower()]
        for h in hits:
            if yr and h["year"] and h["year"] not in yr:
                print(f"  IN-TEXT: {c}   vs REF: {h['surname']} {h['year']}")

# ---- duplicated sentences (exact, >= 12 words) ----
sents = re.split(r"(?<=[.!?])\s+", body)
cnt = Counter(s.strip() for s in sents if len(s.split()) >= 12)
print("\n-- DUPLICATED SENTENCES (>=12 words, exact):")
for s, v in cnt.items():
    if v > 1:
        print(f"  x{v}  {s[:150]}{'...' if len(s)>150 else ''}")

# duplicated long phrases (80+ char substrings via shingling)
import difflib
shingles = set()
dup_phrases = []
words = re.findall(r"\S+", body)
W = 14
for i in range(len(words) - W):
    sh = " ".join(words[i:i+W])
    if sh in shingles:
        continue
    shingles.add(sh)
    # find all occurrences
    occ = [j for j in range(len(words) - W) if " ".join(words[j:j+W]) == sh]
    if len(occ) > 1:
        dup_phrases.append((len(occ), sh[:160]))
print("\n-- DUPLICATED 14-WORD SHINGLES:")
for v, sh in dup_phrases:
    print(f"  x{v}  {sh}")

# ---- orphan / dangling ----
print("\n-- DANGLING CITATIONS (in text, not in reference list):")
listed = {e["surname"].lower() for e in entries}
for kind, c in sorted(set(cites)):
    names = re.findall(r"[A-Z][a-zà-ÿȘș\-]+", c.split(",")[0])
    missing = [n for n in names if not any(n.lower() in l or l.startswith(n.lower()) for l in listed)]
    if missing:
        print(f"  {kind}: {c}   MISSING: {missing}")

print("\n-- ORPHAN REFERENCES (in list, never cited in text):")
for e in entries:
    first_word = e["surname"].split()[0].split(",")[0]
    # search body for surname (first surname token)
    pat = re.escape(first_word)
    if not re.search(pat, body, re.IGNORECASE):
        print(f"  {e['surname']} ({e['year']})  ->  {e['text'][:90]}")
