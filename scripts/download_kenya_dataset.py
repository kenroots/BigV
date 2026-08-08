#!/usr/bin/env python3
"""
download_kenya_dataset.py
─────────────────────────
Downloads iNaturalist "Research Grade" photos for every Kenya wildlife species
and organises them into a Roboflow-ready folder structure:

  dataset/
    images/
      lion/          ← one sub-folder per label
      leopard/
      ...
    data.yaml        ← YOLOv8 class list (copy to Roboflow on upload)
    manifest.csv     ← photo_id, url, label, license, attribution

Usage:
  python3 download_kenya_dataset.py              # 80 images per species
  python3 download_kenya_dataset.py --per 150    # 150 images per species
  python3 download_kenya_dataset.py --species lion,leopard  # subset only
  python3 download_kenya_dataset.py --out ~/my_dataset      # custom output dir

Requirements: Python 3.9+ stdlib only (no pip installs needed).
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# ─── Kenya species: (display_label, scientific_name, iNaturalist_taxon_id) ───
# Taxon IDs verified against api.inaturalist.org/v1/taxa
KENYA_SPECIES: list[tuple[str, str, int]] = [
    # Big Five
    ("lion",              "Panthera leo",               41964),
    ("leopard",           "Panthera pardus",             41968),
    ("elephant",          "Loxodonta africana",          46359),
    ("buffalo",           "Syncerus caffer",             42158),
    ("rhinoceros",        "Diceros bicornis",            42222),
    # Big cats
    ("cheetah",           "Acinonyx jubatus",            41969),
    # Plains grazers
    ("zebra",             "Equus quagga",                42396),
    ("giraffe",           "Giraffa camelopardalis",      42066),
    ("wildebeest",        "Connochaetes taurinus",       42159),
    ("topi",              "Damaliscus lunatus",          42160),
    ("thomson_gazelle",   "Eudorcas thomsonii",          42205),
    ("grant_gazelle",     "Nanger granti",               42207),
    ("impala",            "Aepyceros melampus",          42162),
    ("hartebeest",        "Alcelaphus buselaphus",       42161),
    ("eland",             "Tragelaphus oryx",            42167),
    ("oryx",              "Oryx beisa",                  42210),
    ("warthog",           "Phacochoerus africanus",      42251),
    # Water species
    ("hippopotamus",      "Hippopotamus amphibius",      42254),
    ("crocodile",         "Crocodylus niloticus",        46169),
    # Predators
    ("hyena",             "Crocuta crocuta",             41977),
    ("jackal",            "Canis mesomelas",             42037),
    ("african_wild_dog",  "Lycaon pictus",               42041),
    # Primates
    ("baboon",            "Papio anubis",                43545),
    ("colobus_monkey",    "Colobus guereza",             43565),
    # Birds
    ("ostrich",           "Struthio camelus",            4849),
    ("vulture",           "Gyps africanus",              4857),
    ("flamingo",          "Phoeniconaias minor",         65463),
    ("secretary_bird",    "Sagittarius serpentarius",    4862),
]

INAT_API       = "https://api.inaturalist.org/v1"
INAT_PHOTOS    = "https://inaturalist-open-data.s3.amazonaws.com/photos"
REQUEST_DELAY  = 0.4    # seconds between API calls — respect rate limit
KENYA_PLACE_ID = 7042   # iNaturalist place_id for Kenya


def _get(url: str, params: dict | None = None, retries: int = 3) -> dict:
    """GET a JSON endpoint with simple retry logic."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "BigV-dataset-builder/1.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 10 * (attempt + 1)
                print(f"  Rate limited — waiting {wait}s …")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2)
    return {}


def resolve_taxon_id(label: str, scientific: str, hint_id: int) -> int:
    """Verify the hint taxon ID or resolve it via the API."""
    try:
        d = _get(f"{INAT_API}/taxa/{hint_id}")
        results = d.get("results", [])
        if results and results[0].get("name", "").lower() == scientific.lower():
            return hint_id
    except Exception:
        pass
    # Fall back to name search
    d = _get(f"{INAT_API}/taxa", {"q": scientific, "rank": "species", "per_page": 1})
    results = d.get("results", [])
    if results:
        return results[0]["id"]
    raise RuntimeError(f"Cannot resolve taxon for {label} / {scientific}")


def fetch_photo_urls(taxon_id: int, label: str, count: int, place_id: int = KENYA_PLACE_ID) -> list[dict]:
    """
    Fetch up to `count` Research Grade observation photos for the taxon.
    place_id=7042 = Kenya on iNaturalist.  Pass place_id=None for global.
    Returns list of {photo_id, url, license, attribution}.
    """
    photos = []
    page   = 1
    per_page = min(count, 50)

    while len(photos) < count:
        params: dict = {
            "taxon_id":      taxon_id,
            "quality_grade": "research",
            "photo_licensed": "true",   # only photos with open licenses
            "per_page":      per_page,
            "page":          page,
            "order_by":      "votes",   # highest quality first
        }
        if place_id:
            params["place_id"] = place_id
        try:
            d = _get(f"{INAT_API}/observations", params)
        except Exception as e:
            print(f"  API error page {page}: {e}")
            break

        obs_list = d.get("results", [])
        if not obs_list:
            break

        for obs in obs_list:
            for photo in obs.get("photos", []):
                if len(photos) >= count:
                    break
                pid   = photo.get("id")
                lic   = photo.get("license_code", "cc-by-nc")
                attr  = photo.get("attribution", "")
                # Build medium-size URL
                url   = f"{INAT_PHOTOS}/{pid}/medium.jpg"
                photos.append({
                    "photo_id":    pid,
                    "url":         url,
                    "label":       label,
                    "license":     lic,
                    "attribution": attr,
                })

        total = d.get("total_results", 0)
        if len(obs_list) < per_page or len(photos) >= count:
            break
        page += 1
        time.sleep(REQUEST_DELAY)

    return photos[:count]


def download_photo(url: str, dest: Path, retries: int = 3) -> bool:
    """Download a single photo to dest. Returns True on success."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "BigV-dataset-builder/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
            if len(data) < 5_000:
                return False   # placeholder / error image
            dest.write_bytes(data)
            return True
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return False


def write_data_yaml(out_dir: Path, labels: list[str]) -> None:
    """Write a YOLOv8-compatible data.yaml."""
    yaml_lines = [
        f"# Kenya Wildlife Dataset — generated by download_kenya_dataset.py",
        f"# {len(labels)} classes",
        f"path: {out_dir.resolve()}",
        f"train: images",
        f"val:   images",   # user should split manually or via Roboflow
        f"",
        f"nc: {len(labels)}",
        f"names:",
    ]
    for lbl in labels:
        yaml_lines.append(f"  - {lbl}")
    (out_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n")


def write_manifest(out_dir: Path, all_photos: list[dict]) -> None:
    """Write manifest.csv with full attribution metadata."""
    path = out_dir / "manifest.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["photo_id", "url", "label", "license", "attribution"])
        writer.writeheader()
        writer.writerows(all_photos)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Kenya wildlife dataset from iNaturalist")
    parser.add_argument("--per",     type=int,   default=80,
                        help="Images per species (default: 80)")
    parser.add_argument("--out",     type=str,   default="dataset",
                        help="Output directory (default: ./dataset)")
    parser.add_argument("--species", type=str,   default="",
                        help="Comma-separated labels to download (default: all)")
    parser.add_argument("--place",   type=int,   default=KENYA_PLACE_ID,
                        help=f"iNaturalist place_id (default: {KENYA_PLACE_ID} = Kenya)")
    parser.add_argument("--no-download", action="store_true",
                        help="Dry run — fetch URLs only, skip image download")
    args = parser.parse_args()

    out_dir    = Path(args.out)
    images_dir = out_dir / "images"

    # Filter species if requested
    species_filter = {s.strip().lower() for s in args.species.split(",") if s.strip()}
    species = [
        (lbl, sci, tid) for lbl, sci, tid in KENYA_SPECIES
        if not species_filter or lbl.lower() in species_filter
    ]

    if not species:
        print("No matching species found. Check --species spelling.")
        sys.exit(1)

    print(f"\n{'─'*60}")
    print(f"  BigV — Kenya Wildlife Dataset Builder")
    print(f"  {len(species)} species · {args.per} images each")
    print(f"  Output: {out_dir.resolve()}")
    print(f"  Place:  iNaturalist place_id={args.place}")
    print(f"{'─'*60}\n")

    all_photos: list[dict] = []
    labels: list[str] = []

    for i, (label, scientific, hint_id) in enumerate(species, 1):
        print(f"[{i:2}/{len(species)}] {label:22} ({scientific})")

        # Resolve taxon
        try:
            taxon_id = resolve_taxon_id(label, scientific, hint_id)
        except Exception as e:
            print(f"  ⚠ Skipping — taxon lookup failed: {e}")
            continue

        # Fetch photo metadata
        photos = fetch_photo_urls(taxon_id, label, args.per, args.place)
        if not photos:
            print(f"  ⚠ No Research Grade photos found in Kenya — trying globally")
            photos = fetch_photo_urls(taxon_id, label, args.per, place_id=None)

        if not photos:
            print(f"  ⚠ No photos found at all — skipping")
            continue

        print(f"  Found {len(photos)} photos")
        all_photos.extend(photos)
        labels.append(label)

        if args.no_download:
            time.sleep(REQUEST_DELAY)
            continue

        # Download images
        label_dir = images_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        ok = 0
        for j, photo in enumerate(photos):
            dest = label_dir / f"{photo['photo_id']}.jpg"
            if dest.exists():
                ok += 1
                continue
            if download_photo(photo["url"], dest):
                ok += 1
            else:
                print(f"  ⚠ Failed: {photo['url']}")
            if j % 10 == 9:
                print(f"  … {ok}/{j+1} downloaded")
            time.sleep(0.05)   # small delay between image downloads

        print(f"  ✓ {ok}/{len(photos)} images saved → {label_dir}")
        time.sleep(REQUEST_DELAY)

    # Write metadata files
    out_dir.mkdir(parents=True, exist_ok=True)
    write_data_yaml(out_dir, labels)
    write_manifest(out_dir, all_photos)

    total = sum(1 for p in (images_dir).rglob("*.jpg")) if not args.no_download else len(all_photos)
    print(f"\n{'─'*60}")
    print(f"  ✅ Done — {total} images across {len(labels)} species")
    print(f"  data.yaml  → {out_dir/'data.yaml'}")
    print(f"  manifest   → {out_dir/'manifest.csv'}")
    print(f"\n  Next steps:")
    print(f"  1. Upload {images_dir}/ to Roboflow (drag-drop the folder)")
    print(f"  2. Auto-annotate with Roboflow's Label Assist (CLIP-based)")
    print(f"  3. Review & correct labels, then train YOLOv8s")
    print(f"  4. Set ROBOFLOW_PROJECT=kenya-wildlife in Railway env vars")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()

# Made with Bob
