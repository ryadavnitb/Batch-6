import aiohttp
import asyncio
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")

BASE_URL = "https://api.openaq.org/v3/locations"
HEADERS = {"x-api-key": API_KEY}

LIMIT = 1000          # max allowed
BATCH_SIZE = 20       # fetch 20 pages in parallel


async def fetch_page(session, page):
    params = {"limit": LIMIT, "page": page}
    try:
        async with session.get(BASE_URL, headers=HEADERS, params=params) as resp:
            if resp.status == 429:
                await asyncio.sleep(2)
                return await fetch_page(session, page)
            data = await resp.json()
            return data.get("results", [])
    except:
        return []


async def main():
    all_results = []
    page = 1

    async with aiohttp.ClientSession() as session:
        while True:
            # Create batch of 20 parallel requests
            tasks = [fetch_page(session, p) for p in range(page, page + BATCH_SIZE)]
            results = await asyncio.gather(*tasks)

            # Add fetched data
            empty_count = 0
            for r in results:
                if r:
                    all_results.extend(r)
                else:
                    empty_count += 1

            print(f"Fetched pages {page} → {page + BATCH_SIZE - 1} | Total: {len(all_results)}")

            # Stop if all 20 pages returned empty
            if empty_count == BATCH_SIZE:
                break

            page += BATCH_SIZE

    df = pd.DataFrame(all_results)
    df.to_csv("openaq_locations_ultrafast.csv", index=False)
    print(f"\n🔥 DONE! Saved {len(df)} locations to openaq_locations_ultrafast.csv")


# Run it
asyncio.run(main())
