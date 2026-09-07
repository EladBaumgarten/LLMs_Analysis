"""Corpus builder - Alterman (Ben-Yehuda authority 533), via the authenticated API.

He is `by_permission`, so unlike Ahad Ha'am his texts exist only behind the API:
    /authorities/533?author_detail=texts  -> author-role text ids
    POST /texts/batch                     -> metadata + a .txt download_url
    GET  <download_url>                   -> UTF-8 plaintext with nikud

Saves nikud/ and stripped/ variants and an index with the same schema as Ahad Ha'am, so every
downstream stage stays author-agnostic. Idempotent: existing texts and index rows survive a
re-run, which is what keeps the hand-verified dating layer intact.
"""
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT))
from authors_paths import for_author                   # noqa: E402

P = for_author("alterman", create=True)
BASE = "https://benyehuda.org/api/v1"
AUTHORITY_ID = 533
from text_clean import strip_nikud                    # preserves maqaf
YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")

load_dotenv(HERE / "benYehudaAPI.env")
KEY = (os.getenv("BENYEHUDA_API_KEY") or "").strip().strip('"').strip("'")
assert len(KEY) == 64, f"API key looks wrong (len {len(KEY)})"


def author_role_ids() -> list[int]:
    r = requests.get(f"{BASE}/authorities/{AUTHORITY_ID}",
                     params={"key": KEY, "author_detail": "texts"})
    r.raise_for_status()
    return r.json()["texts"]["author"]


def _scrub(msg: str) -> str:
    return msg.replace(KEY, "***KEY***") if KEY else msg


def _req(method: str, url: str, *, tries: int = 6, **kw):
    """One request with exponential backoff on 429/5xx; key scrubbed from any error."""
    for i in range(tries):
        r = requests.request(method, url, **kw)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** i)
            continue
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            raise requests.HTTPError(_scrub(str(e))) from None
        return r
    raise requests.HTTPError(_scrub(f"gave up after {tries} tries: {method} {url}"))


def batch_metadata(ids: list[int]) -> list[dict]:
    """Metadata + download_url for up to 25 ids."""
    r = _req("POST", f"{BASE}/texts/batch", params={"key": KEY},
             json={"ids": ids, "view": "metadata", "file_format": "txt"})
    data = r.json()
    return data if isinstance(data, list) else data.get("data", [])


def download_text(url: str) -> str:
    r = _req("GET", url)
    r.encoding = "utf-8"  # the text/plain response carries no charset
    return r.text


def year_of(meta: dict):
    """Original composition or first-publication year. Never the site's own pby_ date."""
    for field in ("orig_publication_date", "raw_publication_date", "creation_date", "raw_creation_date"):
        v = meta.get(field)
        if v:
            m = YEAR.search(str(v))
            if m:
                return int(m.group(1))
    return None


def merge_index(api_idx: pd.DataFrame) -> pd.DataFrame:
    """Fold fresh API metadata into the existing index without destroying curation.

    The index carries a hand-built dating layer no script can regenerate - verified COMPOSITION
    years, not the API's collected-edition years - plus provenance columns the API has no notion of.
    So existing rows are kept VERBATIM and only new ids are appended; differences are reported,
    never applied. --refresh-api-cols accepts API values for non-curated columns only.
    """
    path = P["metadata"] / "index.csv"
    if not path.exists():                                   # first build, nothing to protect
        return api_idx.sort_values("id").reset_index(drop=True)

    cur = pd.read_csv(path, encoding="utf-8-sig")
    backup = P["metadata"] / f"index.backup-{time.strftime('%Y%m%d-%H%M%S')}.csv"
    cur.to_csv(backup, index=False, encoding="utf-8-sig")
    print(f"\n[merge_index] existing index: {len(cur)} rows x {len(cur.columns)} cols "
          f"-> backed up to {backup.name}")

    CURATED = {"original_date", "year", "date_basis", "edition_date", "exact_date",
               "date_verdict", "date_confidence", "date_source", "user_year",
               "nonby_date", "nonby_source", "include_in_stylometry"}
    cur_ids, api_ids = set(cur["id"]), set(api_idx["id"])

    new = api_idx[api_idx["id"].isin(api_ids - cur_ids)]
    if len(new):
        print(f"[merge_index] appending {len(new)} NEW ids: {sorted(new['id'])}")
        cur = pd.concat([cur, new], ignore_index=True)
    if api_ids - cur_ids == set() and not len(new):
        print("[merge_index] no new ids")
    if cur_ids - api_ids:
        print(f"[merge_index] WARNING {len(cur_ids - api_ids)} indexed ids absent from the API "
              f"(kept): {sorted(cur_ids - api_ids)}")

    # Report disagreements, so a real upstream change is still visible.
    shared = api_idx[api_idx["id"].isin(cur_ids & api_ids)].set_index("id")
    base = cur.set_index("id")
    for col in shared.columns:
        if col not in base.columns:
            print(f"[merge_index] NOTE api column not in index, ignored: {col}")
            continue
        diff = shared.index[shared[col].astype(str).values != base.loc[shared.index, col].astype(str).values]
        if len(diff):
            tag = "CURATED, kept" if col in CURATED else "kept (use --refresh-api-cols to accept)"
            print(f"[merge_index] {col}: {len(diff)} rows differ from API -> {tag}")
            if col in CURATED:
                ex = diff[0]
                print(f"               e.g. id {ex}: index={base.loc[ex, col]!r} api={shared.loc[ex, col]!r}")
            elif "--refresh-api-cols" in sys.argv:
                cur.loc[cur["id"].isin(diff), col] = shared.loc[diff, col].values
                print(f"               -> refreshed {len(diff)} rows")

    out = cur.sort_values("id").reset_index(drop=True)
    assert set(base.columns) <= set(out.columns), "merge_index dropped a column"
    print(f"[merge_index] result: {len(out)} rows x {len(out.columns)} cols\n")
    return out


def main():
    ids = author_role_ids()
    print(f"author-role ids: {len(ids)}")

    rows = []
    n_new = 0
    for start in range(0, len(ids), 25):
        chunk = ids[start:start + 25]
        for item in batch_metadata(chunk):
            tid = item["id"]
            meta = item["metadata"]
            nikud_path = P["corpus_nikud"] / f"{tid}.txt"
            strip_path = P["corpus_stripped"] / f"{tid}.txt"

            if not (nikud_path.exists() and strip_path.exists()):
                body = download_text(item["download_url"])
                nikud_path.write_text(body, encoding="utf-8")
                strip_path.write_text(strip_nikud(body), encoding="utf-8")
                n_new += 1
                time.sleep(0.5)

            yr = year_of(meta)
            genre = meta.get("genre")
            rows.append({
                "id": tid,
                "title": meta.get("title"),
                "authors": meta.get("author_string"),
                "translators": "",
                "role": "author",
                "include_in_stylometry": genre != "reference",
                "genre": genre,
                "genre_clean": genre,
                "original_language": meta.get("orig_lang"),
                "source_edition": meta.get("publisher"),
                "original_date": yr if yr else "",
                "publication_date": meta.get("pby_publication_date"),
                "date_basis": "original" if yr else "undated",
                "ip_status": meta.get("intellectual_property"),
                "text_stripped_path": f"authors/alterman/corpus/stripped/{tid}.txt",
                "text_nikud_path": f"authors/alterman/corpus/nikud/{tid}.txt",
                "path": item.get("url"),
                "year": float(yr) if yr else "",
            })

    idx = merge_index(pd.DataFrame(rows))
    idx.to_csv(P["metadata"] / "index.csv", index=False, encoding="utf-8-sig")
    idx.to_json(P["metadata"] / "index.json", orient="records", force_ascii=False, indent=2)

    print(f"downloaded new: {n_new} | index rows: {len(idx)}")
    print(f"stripped files: {len(list(P['corpus_stripped'].glob('*.txt')))} | "
          f"nikud files: {len(list(P['corpus_nikud'].glob('*.txt')))}")
    empty = [p.name for p in P["corpus_stripped"].glob("*.txt") if p.stat().st_size == 0]
    print(f"empty stripped files: {len(empty)} {empty if empty else ''}")
    print("\ngenre_clean:")
    print(idx["genre_clean"].value_counts().to_string())
    print("\ndate_basis:")
    print(idx["date_basis"].value_counts().to_string())
    dated = idx[idx["original_date"] != ""]
    if len(dated):
        print(f"\nyear range: {int(dated['original_date'].min())}-{int(dated['original_date'].max())}")
    print("\nip_status:")
    print(idx["ip_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
