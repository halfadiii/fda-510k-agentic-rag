import requests

for term in ["wound dressing", "surgical drape", "computer assisted detection"]:
    r = requests.get(
        "https://api.fda.gov/device/510k.json",
        params={"search": f'device_name:"{term}"', "limit": 1},
        timeout=30,
    )
    total = r.json().get("meta", {}).get("results", {}).get("total", 0)
    print(f"{term:35} {total:>6} clearances")