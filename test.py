import xml.etree.ElementTree as ET

def extract_help_urls(file_path='sitemap.xml'):
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()

        root = ET.fromstring(content)
        ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

        all_urls = [url.find('ns:loc', ns).text for url in root.findall('ns:url', ns)]
        help_urls = [url for url in all_urls if '/help/' in url]

        print(f"Found {len(help_urls)} help URLs.")
        return help_urls

    except Exception as e:
        print(f"Error reading sitemap: {e}")
        return []
