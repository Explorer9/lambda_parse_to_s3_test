import requests
from bs4 import BeautifulSoup
import re
import boto3
import os

# S3 client and target bucket
s3 = boto3.client('s3')
BUCKET_NAME = os.environ.get("BUCKET_NAME", "your-s3-bucket-name")

def extract_text_with_links(element):
    result = ""
    for child in element.descendants:
        if child.name == 'a' and child.get('href'):
            link_text = child.get_text(strip=True)
            link_url = child.get('href')
            if link_url.startswith('/'):
                link_url = 'https://www.regions.com' + link_url
            elif link_url.startswith('#'):
                link_url = 'https://www.regions.com' + link_url
            if link_text:
                result += f"{link_text} ({link_url})"
        elif child.string and child.parent.name != 'a':
            result += child.string
    return re.sub(r'\s+', ' ', result).strip()

def extract_table_with_links(table_element):
    table_data = []
    for row in table_element.find_all('tr'):
        row_data = []
        for cell in row.find_all(['th', 'td']):
            links = cell.find_all('a')
            text = extract_text_with_links(cell) if links else cell.get_text(strip=True)
            row_data.append(text)
        if row_data:
            table_data.append(row_data)
    return table_data

def extract_list_with_links(list_element):
    items = []
    for li in list_element.find_all('li'):
        links = li.find_all('a')
        text = extract_text_with_links(li) if links else li.get_text(strip=True)
        if text:
            items.append(text)
    return items

def parse_regions_faq(url):
    try:
        resp = requests.get(url, headers={'User-Agent':'Mozilla/5.0'})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, 'html.parser')

        faq_data = []
        containers = soup.find_all('div', 
            class_=['regions-help-new-answer-content', 'regions-help-new-answer-container']
        )

        for c in containers:
            q = c.find(['h1','h2','h3'], class_='regions-help-new-answer-title') \
                or c.find(['h1','h2','h3'])
            a = c.find('div', class_='regions-help-new-answer')
            if not (q and a): 
                continue

            question = q.get_text(strip=True)
            answer = extract_text_with_links(a)

            tables = [td for t in (extract_table_with_links(t) for t in a.find_all('table')) if t]
            lists = [lst for lst in (extract_list_with_links(l) for l in a.find_all(['ul','ol'])) if lst]

            faq_data.append({
                'question': question,
                'answer': answer,
                'tables': tables,
                'lists': lists
            })
        return faq_data

    except Exception as e:
        print(f"Error while parsing '{url}':", e)
        return []

def faq_to_text(faq_data):
    lines = []
    for f in faq_data:
        lines.append(f"Q: {f['question']}\nA: {f['answer']}\n")
        if f['tables']:
            lines.append("Tables:")
            for table in f['tables']:
                for row in table:
                    lines.append("  " + " | ".join(row))
        if f['lists']:
            lines.append("Lists:")
            for lst in f['lists']:
                for item in lst:
                    lines.append("  - " + item)
        lines.append("\n" + "-"*60)
    return "\n".join(lines)

def lambda_handler(event=None, context=None):
    help_urls = [
        "https://www.regions.com/help/online-banking-help/manage-accounts/alerts/how-to-turn-on-alerts",
        "https://www.regions.com/help/online-banking-help/manage-accounts/available-balance/standard-overdraft-coverage",
        "https://www.regions.com/help/products-services/checking-accounts/deposits-and-wire-transfers/what-is-my-routing-transit-number",
        "https://www.regions.com/help/online-banking-help/transfer-money/online-and-mobile-banking-transfers/next-steps-after-enrolling-in-external-transfers",
        # Add more URLs here, each a full https:// link starting with /help/
    ]

    for url in help_urls:
        print("Processing:", url)
        faq = parse_regions_faq(url)
        if not faq:
            print(" → No FAQ found")
            continue

        text = faq_to_text(faq)
        filename = url.rstrip('/').split('/')[-1] or 'index'
        key = f"faqs/{filename}.txt"

        try:
            s3.put_object(
                Bucket=BUCKET_NAME,
                Key=key,
                Body=text.encode('utf-8')
            )
            print(" → Uploaded:", key)
        except Exception as e:
            print(" → Upload failed:", e)

if __name__ == "__main__":
    lambda_handler()
