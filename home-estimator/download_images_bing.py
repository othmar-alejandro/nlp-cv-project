"""Download training images using Bing image crawler (more reliable than Google)."""
import os
from icrawler.builtin import BingImageCrawler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")

SEARCH_QUERIES = {
    "plumbing": [
        "plumbing pipe leak repair",
        "clogged drain sink plumbing",
        "water heater residential",
        "leaking faucet dripping",
        "toilet repair bathroom plumbing",
    ],
    "painting": [
        "interior house wall painting",
        "peeling paint wall damage",
        "exterior house painting job",
        "ceiling paint stain repair",
        "room painting renovation color",
    ],
    "roofing": [
        "damaged roof shingles missing",
        "roof leak water damage attic",
        "residential roof replacement",
        "gutter repair cleaning roof",
        "storm damage roof shingles",
    ],
    "electrical": [
        "electrical outlet wall repair",
        "circuit breaker panel home",
        "electrical wiring home repair",
        "ceiling light fixture install",
        "home electrical work outlet",
    ],
    "hvac": [
        "hvac air conditioner outdoor unit",
        "furnace heating system residential",
        "air duct vent home",
        "thermostat hvac wall",
        "hvac maintenance technician",
    ],
    "general_repair": [
        "drywall repair home patch",
        "fence repair wood residential",
        "deck repair wood damage",
        "window repair home broken",
        "door repair home maintenance",
    ],
}

IMAGES_PER_QUERY = 15

for category, queries in SEARCH_QUERIES.items():
    cat_dir = os.path.join(IMAGE_DIR, category)
    os.makedirs(cat_dir, exist_ok=True)

    existing = len([f for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if existing >= 40:
        print(f"[SKIP] {category}: already has {existing} images")
        continue

    print(f"\n[DOWNLOADING] {category}...")
    for i, query in enumerate(queries):
        try:
            crawler = BingImageCrawler(
                storage={"root_dir": cat_dir},
                log_level="WARNING",
            )
            crawler.crawl(
                keyword=query,
                max_num=IMAGES_PER_QUERY,
                min_size=(100, 100),
            )
        except Exception as e:
            print(f"  Warning: {query} - {e}")

    final_count = len([f for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"[DONE] {category}: {final_count} images")

print("\n=== Image download complete ===")
for cat in SEARCH_QUERIES:
    cat_dir = os.path.join(IMAGE_DIR, cat)
    count = len([f for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  {cat}: {count} images")
