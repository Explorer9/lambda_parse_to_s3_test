import json
import boto3
import urllib.request
import urllib.parse
from html.parser import HTMLParser
import re
import os
from datetime import datetime

class FAQHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset_parser()
        
    def reset_parser(self):
        self.current_element = None
        self.current_attrs = {}
        self.text_buffer = []
        self.in_faq_container = False
        self.in_question = False
        self.in_answer = False
        self.in_table = False
        self.in_list = False
        self.current_question = ""
        self.current_answer = ""
        self.tables = []
        self.lists = []
        self.current_table = []
        self.current_row = []
        self.current_list = []
        self.faqs = []
        self.link_stack = []
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.current_element = tag
        self.current_attrs = attrs_dict
        
        # Check for FAQ containers
        if tag == 'div':
            class_attr = attrs_dict.get('class', '')
            if 'regions-help-new-answer-content' in class_attr or 'regions-help-new-answer-container' in class_attr:
                self.in_faq_container = True
        
        # Check for question headers
        if tag in ['h1', 'h2', 'h3'] and self.in_faq_container:
            class_attr = attrs_dict.get('class', '')
            if 'regions-help-new-answer-title' in class_attr or not class_attr:
                self.in_question = True
                self.text_buffer = []
        
        # Check for answer div
        if tag == 'div' and self.in_faq_container:
            class_attr = attrs_dict.get('class', '')
            if 'regions-help-new-answer' in class_attr:
                self.in_answer = True
                self.text_buffer = []
        
        # Handle links
        if tag == 'a' and attrs_dict.get('href'):
            href = attrs_dict['href']
            if href.startswith('/'):
                href = 'https://www.regions.com' + href
            elif href.startswith('#'):
                href = 'https://www.regions.com/help/products-services/checking-accounts/deposits-and-wire-transfers/what-is-my-routing-transit-number' + href
            self.link_stack.append(href)
        
        # Handle tables
        if tag == 'table' and self.in_answer:
            self.in_table = True
            self.current_table = []
        
        if tag == 'tr' and self.in_table:
            self.current_row = []
        
        # Handle lists
        if tag in ['ul', 'ol'] and self.in_answer:
            self.in_list = True
            self.current_list = []
        
        if tag == 'li' and self.in_list:
            self.text_buffer = []
    
    def handle_endtag(self, tag):
        if tag == 'div':
            if self.in_faq_container and self.current_question and self.current_answer:
                # Save the current FAQ
                faq_item = {
                    'question': self.current_question.strip(),
                    'answer': self.current_answer.strip(),
                    'tables': self.tables.copy(),
                    'lists': self.lists.copy()
                }
                self.faqs.append(faq_item)
                
                # Reset for next FAQ
                self.current_question = ""
                self.current_answer = ""
                self.tables = []
                self.lists = []
            
            if self.in_answer:
                self.current_answer = ' '.join(self.text_buffer)
                self.in_answer = False
            
            self.in_faq_container = False
        
        if tag in ['h1', 'h2', 'h3'] and self.in_question:
            self.current_question = ' '.join(self.text_buffer)
            self.in_question = False
        
        if tag == 'a' and self.link_stack:
            link_url = self.link_stack.pop()
            if self.text_buffer:
                link_text = self.text_buffer[-1] if self.text_buffer else ""
                self.text_buffer[-1] = f"{link_text} ({link_url})"
        
        if tag == 'table' and self.in_table:
            if self.current_table:
                self.tables.append(self.current_table)
            self.in_table = False
        
        if tag == 'tr' and self.in_table:
            if self.current_row:
                self.current_table.append(self.current_row)
            self.current_row = []
        
        if tag in ['ul', 'ol'] and self.in_list:
            if self.current_list:
                self.lists.append(self.current_list)
            self.in_list = False
        
        if tag == 'li' and self.in_list:
            item_text = ' '.join(self.text_buffer).strip()
            if item_text:
                self.current_list.append(item_text)
            self.text_buffer = []
    
    def handle_data(self, data):
        if self.in_question or self.in_answer or (self.in_table and self.current_element in ['td', 'th']) or self.in_list:
            clean_data = re.sub(r'\s+', ' ', data).strip()
            if clean_data:
                if self.in_table and self.current_element in ['td', 'th']:
                    self.current_row.append(clean_data)
                else:
                    self.text_buffer.append(clean_data)

def fetch_webpage(url):
    """Fetch webpage content using urllib"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read().decode('utf-8')
            return content
    except Exception as e:
        print(f"Error fetching webpage {url}: {str(e)}")
        return None

def parse_faq_content(html_content):
    """Parse FAQ content from HTML using custom parser"""
    parser = FAQHTMLParser()
    parser.feed(html_content)
    return parser.faqs

def create_faq_text_content(faq_item):
    """Create formatted text content for FAQ"""
    content = f"Question: {faq_item['question']}\n\n"
    content += f"Answer: {faq_item['answer']}\n\n"
    
    if faq_item.get('tables'):
        content += "Tables:\n"
        for i, table in enumerate(faq_item['tables'], 1):
            content += f"\nTable {i}:\n"
            for row in table:
                content += f"  {' | '.join(row)}\n"
        content += "\n"
    
    if faq_item.get('lists'):
        content += "Lists:\n"
        for i, lst in enumerate(faq_item['lists'], 1):
            content += f"\nList {i}:\n"
            for item in lst:
                content += f"  • {item}\n"
        content += "\n"
    
    return content

def save_to_s3(s3_client, bucket_name, key, content, metadata):
    """Save content to S3 with metadata"""
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=content.encode('utf-8'),
            ContentType='text/plain',
            Metadata=metadata
        )
        return True
    except Exception as e:
        print(f"Error saving to S3: {str(e)}")
        return False

def generate_filename(question, url, index):
    """Generate a safe filename from question"""
    # Clean the question for filename
    safe_question = re.sub(r'[^\w\s-]', '', question)
    safe_question = re.sub(r'\s+', '_', safe_question)
    safe_question = safe_question[:50]  # Limit length
    
    # Extract domain from URL for prefix
    domain = urllib.parse.urlparse(url).netloc.replace('.', '_')
    
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    
    return f"{domain}/faq_{index}_{safe_question}_{timestamp}.txt"

def handler(event, context):
    """Main Lambda handler function"""
    
    # Get parameters from event
    urls = event.get('urls', [])
    bucket_name = event.get('bucket_name', os.environ.get('S3_BUCKET_NAME'))
    
    if not urls:
        return {
            'statusCode': 400,
            'body': json.dumps({
                'error': 'No URLs provided',
                'message': 'Please provide a list of URLs in the event'
            })
        }
    
    if not bucket_name:
        return {
            'statusCode': 400,
            'body': json.dumps({
                'error': 'No S3 bucket specified',
                'message': 'Please provide bucket_name in event or set S3_BUCKET_NAME environment variable'
            })
        }
    
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    results = []
    total_faqs = 0
    
    for url in urls:
        try:
            print(f"Processing URL: {url}")
            
            # Fetch webpage content
            html_content = fetch_webpage(url)
            if not html_content:
                results.append({
                    'url': url,
                    'status': 'error',
                    'message': 'Failed to fetch webpage'
                })
                continue
            
            # Parse FAQ content
            faqs = parse_faq_content(html_content)
            
            if not faqs:
                results.append({
                    'url': url,
                    'status': 'warning',
                    'message': 'No FAQs found on this page'
                })
                continue
            
            # Save each FAQ to S3
            saved_files = []
            for i, faq in enumerate(faqs, 1):
                try:
                    # Create text content
                    text_content = create_faq_text_content(faq)
                    
                    # Generate filename
                    filename = generate_filename(faq['question'], url, i)
                    
                    # Prepare metadata
                    metadata = {
                        'question': faq['question'][:1000],  # S3 metadata limit
                        'source_url': url,
                        'faq_index': str(i),
                        'processed_date': datetime.utcnow().isoformat(),
                        'has_tables': str(bool(faq.get('tables'))),
                        'has_lists': str(bool(faq.get('lists')))
                    }
                    
                    # Save to S3
                    if save_to_s3(s3_client, bucket_name, filename, text_content, metadata):
                        saved_files.append(filename)
                        total_faqs += 1
                    else:
                        print(f"Failed to save FAQ {i} from {url}")
                
                except Exception as e:
                    print(f"Error processing FAQ {i} from {url}: {str(e)}")
            
            results.append({
                'url': url,
                'status': 'success',
                'faqs_found': len(faqs),
                'faqs_saved': len(saved_files),
                'files': saved_files
            })
            
        except Exception as e:
            results.append({
                'url': url,
                'status': 'error',
                'message': str(e)
            })
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'Processing complete. {total_faqs} FAQs saved to S3.',
            'bucket': bucket_name,
            'results': results,
            'summary': {
                'total_urls_processed': len(urls),
                'total_faqs_saved': total_faqs,
                'successful_urls': len([r for r in results if r['status'] == 'success'])
            }
        }, indent=2)
    }

# For local testing
if __name__ == "__main__":
    # Test event
    test_event = {
        'urls': [
            'https://www.regions.com/help/products-services/checking-accounts/deposits-and-wire-transfers/what-is-my-routing-transit-number'
        ],
        'bucket_name': 'your-test-bucket-name'
    }
    
    # Mock context
    class MockContext:
        def __init__(self):
            self.function_name = 'test-function'
            self.memory_limit_in_mb = 128
            self.invoked_function_arn = 'arn:aws:lambda:us-east-1:123456789012:function:test-function'
            self.aws_request_id = 'test-request-id'
    
    result = lambda_handler(test_event, MockContext())
    print(json.dumps(result, indent=2))
