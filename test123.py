import requests
from bs4 import BeautifulSoup
import json
import re

def extract_text_with_links(element):
    """
    Extract text from an element while preserving hyperlinks in brackets
    """
    result = ""
    
    for child in element.descendants:
        if child.name == 'a' and child.get('href'):
            # Get the link text and URL
            link_text = child.get_text(strip=True)
            link_url = child.get('href')
            
            # Handle relative URLs
            if link_url.startswith('/'):
                link_url = 'https://www.regions.com' + link_url
            elif link_url.startswith('#'):
                link_url = 'https://www.regions.com/help/products-services/checking-accounts/deposits-and-wire-transfers/what-is-my-routing-transit-number' + link_url
            
            # Add link text with URL in brackets
            if link_text:
                result += f"{link_text} ({link_url})"
        elif child.string and child.parent.name != 'a':
            # Add regular text (but skip text that's already inside <a> tags)
            result += child.string
    
    # Clean up extra whitespace
    result = re.sub(r'\s+', ' ', result).strip()
    return result

def extract_table_with_links(table_element):
    """
    Extract table data while preserving hyperlinks
    """
    table_data = []
    rows = table_element.find_all('tr')
    
    for row in rows:
        cells = row.find_all(['th', 'td'])
        row_data = []
        
        for cell in cells:
            # Check if cell contains links
            links = cell.find_all('a')
            if links:
                cell_text = extract_text_with_links(cell)
            else:
                cell_text = cell.get_text(strip=True)
            
            row_data.append(cell_text)
        
        if row_data:  # Only add non-empty rows
            table_data.append(row_data)
    
    return table_data

def extract_list_with_links(list_element):
    """
    Extract list items while preserving hyperlinks
    """
    list_items = []
    
    for li in list_element.find_all('li'):
        links = li.find_all('a')
        if links:
            item_text = extract_text_with_links(li)
        else:
            item_text = li.get_text(strip=True)
        
        if item_text:
            list_items.append(item_text)
    
    return list_items

def parse_regions_faq(url):
    """
    Parse FAQ data from Regions Bank webpage
    """
    try:
        # Fetch the webpage
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        faq_data = []
        
        # Look for FAQ containers
        faq_containers = soup.find_all(['div'], class_=['regions-help-new-answer-content', 'regions-help-new-answer-container'])
        
        for container in faq_containers:
            # Find question
            question_elem = container.find(['h1', 'h2', 'h3'], class_=['regions-help-new-answer-title'])
            if not question_elem:
                question_elem = container.find(['h1', 'h2', 'h3'])
            
            # Find answer
            answer_elem = container.find('div', class_='regions-help-new-answer')
            
            if question_elem and answer_elem:
                question_text = question_elem.get_text(strip=True)
                
                # Extract answer text with hyperlinks
                answer_text = extract_text_with_links(answer_elem)
                
                # Extract tables if present with links
                tables = []
                table_elements = answer_elem.find_all('table')
                
                for table in table_elements:
                    table_data = extract_table_with_links(table)
                    if table_data:
                        tables.append(table_data)
                
                # Extract any lists with links
                lists = []
                list_elements = answer_elem.find_all(['ul', 'ol'])
                
                for list_elem in list_elements:
                    list_items = extract_list_with_links(list_elem)
                    if list_items:
                        lists.append(list_items)
                
                faq_item = {
                    'question': question_text,
                    'answer': answer_text,
                    'tables': tables,
                    'lists': lists
                }
                
                faq_data.append(faq_item)
        
        return faq_data
        
    except requests.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return []
    except Exception as e:
        print(f"Error parsing the webpage: {e}")
        return []

def save_to_json(data, filename='regions_faq.json'):
    """Save parsed data to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Data saved to {filename}")

def print_faq_data(faq_data):
    """Print FAQ data in a readable format"""
    for i, faq in enumerate(faq_data, 1):
        print(f"\n{'='*50}")
        print(f"FAQ {i}")
        print(f"{'='*50}")
        print(f"Question: {faq['question']}")
        print(f"\nAnswer: {faq['answer']}")
        
        if faq['tables']:
            print(f"\nTables ({len(faq['tables'])} found):")
            for j, table in enumerate(faq['tables'], 1):
                print(f"\n  Table {j}:")
                for row in table:
                    print(f"    {' | '.join(row)}")
        
        if faq['lists']:
            print(f"\nLists ({len(faq['lists'])} found):")
            for j, lst in enumerate(faq['lists'], 1):
                print(f"\n  List {j}:")
                for item in lst:
                    print(f"    • {item}")

# Main execution
if __name__ == "__main__":
    url = "https://www.regions.com/help/products-services/checking-accounts/deposits-and-wire-transfers/what-is-my-routing-transit-number"
    
    print("Fetching and parsing Regions Bank FAQ data...")
    faq_data = parse_regions_faq(url)
    
    if faq_data:
        print(f"\nFound {len(faq_data)} FAQ items")
        
        # Print the data
        print_faq_data(faq_data)
        
        # Save to JSON file
        save_to_json(faq_data)
        
        # Also save routing numbers table separately if found
        for faq in faq_data:
            if 'routing' in faq['question'].lower() and faq['tables']:
                routing_data = {'routing_numbers': faq['tables'][0]}
                with open('routing_numbers.json', 'w') as f:
                    json.dump(routing_data, f, indent=2)
                print("Routing numbers saved to routing_numbers.json")
                
    else:
        print("No FAQ data found or error occurred")

# Alternative simpler version for quick testing
def simple_parse(url):
    """Simplified version that just gets the main content"""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Get the main title
    title = soup.find('h2', class_='regions-help-new-answer-title')
    title_text = title.get_text(strip=True) if title else "No title found"
    
    # Get the answer content
    answer = soup.find('div', class_='regions-help-new-answer')
    answer_text = answer.get_text(strip=True) if answer else "No answer found"
    
    # Get table data (routing numbers)
    table = soup.find('table')
    table_data = []
    if table:
        rows = table.find_all('tr')
        for row in rows:
            cells = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
            table_data.append(cells)
    
    return {
        'question': title_text,
        'answer': answer_text,
        'table': table_data
    }

# Uncomment below to run the simple version
# print("\n" + "="*50)
# print("SIMPLE VERSION:")
# print("="*50)
# simple_data = simple_parse(url)
# print(f"Question: {simple_data['question']}")
# print(f"Answer: {simple_data['answer']}")
# print(f"Table: {simple_data['table']}")
