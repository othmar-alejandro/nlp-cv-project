"""Download training images for each home service category using icrawler."""
import os
from icrawler.builtin import GoogleImageCrawler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")

SEARCH_QUERIES = {
    "plumbing": [
        "broken pipe leak plumbing repair",
        "clogged drain plumbing",
        "water heater repair residential",
        "leaking faucet kitchen bathroom",
        "toilet repair plumbing service",
    ],
    "painting": [
        "house wall painting interior",
        "peeling paint wall damage",
        "exterior house painting",
        "ceiling paint water stain",
        "room painting renovation",
    ],
    "roofing": [
        "damaged roof shingles repair",
        "roof leak damage residential",
        "missing roof shingles storm damage",
        "roof replacement residential home",
        "gutter repair roof maintenance",
    ],
    "electrical": [
        "electrical outlet repair residential",
        "electrical panel circuit breaker",
        "home electrical wiring repair",
        "ceiling light fixture installation",
        "electrical switch outlet wall",
    ],
    "hvac": [
        "hvac air conditioner unit residential",
        "furnace repair residential",
        "air duct ventilation system",
        "thermostat hvac system home",
        "hvac outdoor unit maintenance",
    ],
    "general_repair": [
        "home repair handyman drywall",
        "fence repair residential",
        "deck repair wood damage",
        "broken window repair home",
        "door repair residential maintenance",
    ],
}

IMAGES_PER_QUERY = 15  # 5 queries × 15 = ~75 per category

for category, queries in SEARCH_QUERIES.items():
    cat_dir = os.path.join(IMAGE_DIR, category)
    os.makedirs(cat_dir, exist_ok=True)

    existing = len([f for f in os.listdir(cat_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    if existing >= 50:
        print(f"[SKIP] {category}: already has {existing} images")
        continue

    print(f"\n[DOWNLOADING] {category}...")
    for i, query in enumerate(queries):
        crawler = GoogleImageCrawler(
            storage={"root_dir": cat_dir},
            log_level="WARNING",
        )
        crawler.crawl(
            keyword=query,
            max_num=IMAGES_PER_QUERY,
            min_size=(100, 100),
        )

    final_count = len([f for f in os.listdir(cat_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"[DONE] {category}: {final_count} images")

print("\n=== Image download complete ===")
for cat in SEARCH_QUERIES:
    cat_dir = os.path.join(IMAGE_DIR, cat)
    count = len([f for f in os.listdir(cat_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  {cat}: {count} images")
