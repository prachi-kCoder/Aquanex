# from tools import cmfritool

# def display_report(result: dict):
#     pass
    # print("\n=== Report Summary ===")
    # print("File:", result["file"])
    # print("Pages:", result["pages"])
    # print("Sections found:", list(result["sections"].keys())[:10], "...")
    # print("Tables found:", len(result["tables"]))

    # print("\n--- Sample OCR Text (first 800 chars) ---")
    # if result["sections"]:
    #     first_section = list(result["sections"].values())[0]
    #     print(first_section[:800])
    # else:
    #     print("No text extracted.")

# if __name__ == "__main__":
#     print("Scraping CMFRI reports...")
#     reports = cmfritool.scrape_technical_reports(limit=2)
#     for r in reports:
#         print(f"- {r['title']} ({r['url']})")

#     if not reports:
#         print("No reports found.")
#     else:
#         print("\nDownloading first report...")
#         pdf_path = cmfritool.download_report(reports[0]["relative_path"])

#         print("Parsing OCR...")
#         result = cmfritool.parse_report(pdf_path)

#         display_report(result)



