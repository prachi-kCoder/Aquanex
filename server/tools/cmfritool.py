import os
import requests
from bs4 import BeautifulSoup
from tools import parsetool

EPRINTS_BASE = "https://eprints.cmfri.org.in/"

# def scrape_technical_reports(year=None, limit=5):
#     index_url = f"{EPRINTS_BASE}cgi/search/simple?screen=Public%3A%3AEPrintSearch&dataset=archive&_action_search=Search&_order=bytitle&exp=year%3A{year}" if year else f"{EPRINTS_BASE}view/year/"
#     r = requests.get(index_url)
#     r.raise_for_status()
#     soup = BeautifulSoup(r.text, "html.parser")

#     reports = []
#     links = soup.select("a[href$='.pdf']")
#     for a in links:
#         href = a["href"]
#         title = a.get_text(strip=True)

#         # Filter by year if provided
#         if year and str(year) not in title and str(year) not in href:
#             continue

#         reports.append({
#             "title": title,
#             "url": EPRINTS_BASE + href.lstrip("/"),
#             "relative_path": href
#         })

#         if len(reports) >= limit:
#             break

#     return reports
# def scrape_technical_reports(year=None, limit=5):
#     index_url = f"{EPRINTS_BASE}cgi/search/simple?screen=Public::EPrintSearch&dataset=archive&_action_search=Search&_order=bytitle&exp=year%3A{year}"
#     r = requests.get(index_url)
#     r.raise_for_status()
#     soup = BeautifulSoup(r.text, "html.parser")
#     reports = []

#     # First, get item pages
#     for item in soup.select("a[href*='id/eprint']"):
#         item_url = EPRINTS_BASE + item['href'].lstrip('/')
#         item_resp = requests.get(item_url)
#         item_soup = BeautifulSoup(item_resp.text, "html.parser")
#         pdf_link = item_soup.select_one("a[href$='.pdf']")
#         if not pdf_link:
#             continue
#         href = pdf_link['href']
#         title = pdf_link.get_text(strip=True)
#         # If filtering by year, check item metadata or avoid aggressive exclude
#         reports.append({"title": title, "url": EPRINTS_BASE + href.lstrip('/'), "relative_path": href})
#         if len(reports) >= limit:
#             break
#     return reports
EPRINTS_BASE = "https://eprints.cmfri.org.in/"
    # # Use search to filter by year
    # index_url = f"{EPRINTS_BASE}cgi/search/simple?screen=Public::EPrintSearch&dataset=archive&_action_search=Search&_order=bytitle&exp=year%3A{year}" if year else f"{EPRINTS_BASE}view/year/"
def scrape_technical_reports(year=None, limit=5):
    index_url = f"{EPRINTS_BASE}view/Document_Type/Technical_report.html"
    r = requests.get(index_url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    reports = []
    item_links = soup.select("a[href*='id/eprint']")
    for item in item_links:
        item_url = EPRINTS_BASE + item['href'].lstrip('/')
        item_resp = requests.get(item_url)
        item_soup = BeautifulSoup(item_resp.text, "html.parser")
        # Select all PDF links in the page
        for pdf_link in item_soup.select("a[href$='.pdf']"):
            href = pdf_link['href']
            title = pdf_link.get_text(strip=True) or item.get_text(strip=True)
            # If year is specified, check in item page title or href or metadata for more robust filter
            if year and str(year) not in title and str(year) not in href:
                continue
            reports.append({
                "title": title,
                "url": EPRINTS_BASE + href.lstrip('/'),
                "relative_path": href
            })
            if len(reports) >= limit:
                return reports
    return reports



def download_report(relative_path: str, out_dir="downloads") -> str:
    """Download CMFRI report PDF."""
    os.makedirs(out_dir, exist_ok=True)
    url = EPRINTS_BASE + relative_path.lstrip("/")
    fname = os.path.join(out_dir, os.path.basename(relative_path))
    if not os.path.exists(fname):
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(fname, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
    return fname

def parse_report(pdf_path: str) -> dict:
    text_data = parsetool.extract_text_ocr(pdf_path)
    sections = parsetool.split_sections_from_text(text_data.get("full_text", ""))
    tables = parsetool.extract_tables_ocr(pdf_path)

    metadata = parsetool.extract_metadata(text_data.get("pages", [])[0]) if text_data.get("pages") else {}

    return {
        "file": pdf_path,
        "pages": text_data.get("page_count", 0),
        "sections": sections,
        "tables": tables,
        "metadata": metadata
    }

# def parse_report(pdf_path: str) -> dict:
#     """OCR parse report and extract sections + tables."""
#     text_data = parsetool.extract_text_ocr(pdf_path)
#     sections = parsetool.split_sections_from_text(text_data.get("full_text", ""))
#     tables = parsetool.extract_tables_ocr(pdf_path)

#     return {
#         "file": pdf_path,
#         "pages": text_data.get("page_count", 0),
#         "sections": sections,
#         "tables": tables
#     }

# def scrape_technical_reports(year=None, limit=5):
#     index_url = f"{EPRINTS_BASE}cgi/search/simple?screen=Public%3A%3AEPrintSearch&dataset=archive&_action_search=Search&_order=bytitle&exp=year%3A{year}" if year else f"{EPRINTS_BASE}view/year/"
#     r = requests.get(index_url)
#     r.raise_for_status()
#     soup = BeautifulSoup(r.text, "html.parser")

#     reports = []
#     links = soup.select("a[href$='.pdf']")
#     for a in links:
#         href = a["href"]
#         title = a.get_text(strip=True)
#         reports.append({
#             "title": title,
#             "url": EPRINTS_BASE + href.lstrip("/"),
#             "relative_path": href
#         })
#         if len(reports) >= limit:
#             break
#     return reports

# from tools.nlptool import tag_entities

# def enrich_sections(sections: dict) -> dict:
#     enriched = {}
#     for k, v in sections.items():
#         enriched[k] = {
#             "text": v,
#             "tags": tag_entities(v)  # returns {species: [...], locations: [...], metrics: [...]}
#         }
#     return enriched