def format_bedrock_results_for_llm(bedrock_response, user_query=None):
    """
    Format Bedrock retrieval results for LLM context with special handling for verbatim_items.
    
    Args:
        bedrock_response (dict): Response from Bedrock retriever
        user_query (str, optional): User's question to include in context
    
    Returns:
        str: Formatted context string for LLM
    """
    
    if 'retrievalResults' not in bedrock_response:
        return "No retrieval results found."
    
    results = bedrock_response['retrievalResults']
    
    if not results:
        return "No relevant documents found."
    
    # Sort by relevance score (highest first)
    sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
    
    context_parts = ["Context Information:\n"]
    
    for idx, result in enumerate(sorted_results, 1):
        # Extract text content
        text_content = extract_text_content(result.get('content', {}))
        if not text_content:
            continue
            
        # Extract location information
        location_info = extract_location_info(result.get('location', {}))
        
        # Extract and format metadata, especially verbatim_items
        metadata_info = format_metadata(result.get('metadata', {}))
        
        # Format the chunk
        context_parts.append(f"Document {idx}:")
        
        if location_info:
            context_parts.append(f"Source: {location_info}")
        
        if metadata_info:
            context_parts.append(f"Metadata: {metadata_info}")
            
        context_parts.append(f"Content: {text_content}")
        context_parts.append("")  # Empty line for separation
    
    # Add user query if provided
    if user_query:
        context_parts.append("---")
        context_parts.append(f"Question: {user_query}")
    
    return "\n".join(context_parts)


def extract_text_content(content_dict):
    """Extract text content from the content dictionary."""
    if not content_dict:
        return ""
    
    content_type = content_dict.get('type', 'TEXT')
    
    if content_type == 'TEXT' and 'text' in content_dict:
        return content_dict['text']
    elif 'byteContent' in content_dict:
        # Handle byte content if needed (decode if it's text)
        try:
            return content_dict['byteContent']
        except:
            return "[Binary content]"
    elif content_type == 'ROW' and 'row' in content_dict:
        # Handle row data by combining column values
        row_data = content_dict['row']
        row_text = []
        for col in row_data:
            if col.get('type') == 'STRING':
                row_text.append(f"{col.get('columnName', '')}: {col.get('columnValue', '')}")
        return " | ".join(row_text)
    
    return ""


def extract_location_info(location_dict):
    """Extract location information in a readable format."""
    if not location_dict:
        return ""
    
    location_type = location_dict.get('type', '')
    
    if location_type == 'S3' and 's3Location' in location_dict:
        return location_dict['s3Location'].get('uri', '')
    elif location_type == 'WEB' and 'webLocation' in location_dict:
        return location_dict['webLocation'].get('url', '')
    elif location_type == 'CONFLUENCE' and 'confluenceLocation' in location_dict:
        return location_dict['confluenceLocation'].get('url', '')
    elif location_type == 'SALESFORCE' and 'salesforceLocation' in location_dict:
        return location_dict['salesforceLocation'].get('url', '')
    elif location_type == 'SHAREPOINT' and 'sharePointLocation' in location_dict:
        return location_dict['sharePointLocation'].get('url', '')
    elif location_type == 'CUSTOM' and 'customDocumentLocation' in location_dict:
        return f"Document ID: {location_dict['customDocumentLocation'].get('id', '')}"
    elif location_type == 'KENDRA' and 'kendraDocumentLocation' in location_dict:
        return location_dict['kendraDocumentLocation'].get('uri', '')
    elif location_type == 'SQL' and 'sqlLocation' in location_dict:
        return f"SQL Query: {location_dict['sqlLocation'].get('query', '')}"
    
    return f"{location_type} location"


def format_metadata(metadata_dict):
    """Format metadata with special handling for verbatim_items."""
    if not metadata_dict:
        return ""
    
    metadata_parts = []
    
    # Handle verbatim_items specially (combine disclosures)
    if 'verbatim_items' in metadata_dict:
        verbatim_items = metadata_dict['verbatim_items']
        if isinstance(verbatim_items, list) and verbatim_items:
            disclosures = " | ".join([str(item) for item in verbatim_items if item])
            metadata_parts.append(f"Disclosures: {disclosures}")
    
    # Handle other metadata fields
    for key, value in metadata_dict.items():
        if key == 'verbatim_items':
            continue  # Already handled above
            
        if value is not None:
            # Format different value types
            if isinstance(value, list):
                formatted_value = ", ".join([str(v) for v in value])
            elif isinstance(value, dict):
                # For nested dictionaries, create a simple representation
                formatted_value = str(value)
            else:
                formatted_value = str(value)
            
            metadata_parts.append(f"{key}: {formatted_value}")
    
    return " | ".join(metadata_parts)


def format_bedrock_results_xml_style(bedrock_response, user_query=None):
    """
    Alternative XML-style formatting for structured prompts.
    """
    if 'retrievalResults' not in bedrock_response:
        return "<context>No retrieval results found.</context>"
    
    results = bedrock_response['retrievalResults']
    sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
    
    xml_parts = ["<context>"]
    
    for idx, result in enumerate(sorted_results, 1):
        text_content = extract_text_content(result.get('content', {}))
        if not text_content:
            continue
            
        location_info = extract_location_info(result.get('location', {}))
        metadata = result.get('metadata', {})
        
        xml_parts.append(f'  <document id="{idx}" score="{result.get("score", 0):.3f}">')
        
        if location_info:
            xml_parts.append(f'    <source>{location_info}</source>')
        
        # Handle verbatim_items in XML
        if 'verbatim_items' in metadata and metadata['verbatim_items']:
            xml_parts.append('    <disclosures>')
            for item in metadata['verbatim_items']:
                if item:
                    xml_parts.append(f'      <disclosure>{item}</disclosure>')
            xml_parts.append('    </disclosures>')
        
        xml_parts.append(f'    <content>{text_content}</content>')
        xml_parts.append('  </document>')
    
    xml_parts.append("</context>")
    
    if user_query:
        xml_parts.append("")
        xml_parts.append(f"<query>{user_query}</query>")
    
    return "\n".join(xml_parts)


# Example usage
if __name__ == "__main__":
    # Sample Bedrock response for testing
    sample_response = {
        'guardrailAction': 'NONE',
        'retrievalResults': [
            {
                'content': {
                    'text': 'This document contains important financial disclosures about risk management.',
                    'type': 'TEXT'
                },
                'location': {
                    'type': 'S3',
                    's3Location': {
                        'uri': 's3://my-bucket/financial-report-2023.pdf'
                    }
                },
                'metadata': {
                    'document_type': 'financial_report',
                    'year': 2023,
                    'verbatim_items': [
                        'Risk disclosure: Market volatility may affect returns',
                        'Regulatory disclosure: Subject to SEC regulations',
                        'Legal disclosure: Past performance does not guarantee future results'
                    ]
                },
                'score': 0.92
            },
            {
                'content': {
                    'text': 'Additional compliance information and regulatory requirements.',
                    'type': 'TEXT'
                },
                'location': {
                    'type': 'WEB',
                    'webLocation': {
                        'url': 'https://example.com/compliance-guide'
                    }
                },
                'metadata': {
                    'category': 'compliance',
                    'verbatim_items': [
                        'Compliance disclosure: All activities must comply with applicable laws'
                    ]
                },
                'score': 0.85
            }
        ]
    }
    
    # Test the formatter
    formatted_context = format_bedrock_results_for_llm(
        sample_response, 
        "What are the key risk disclosures in the financial documents?"
    )
    
    print("=== Standard Format ===")
    print(formatted_context)
    
    print("\n=== XML Format ===")
    xml_formatted = format_bedrock_results_xml_style(
        sample_response,
        "What are the key risk disclosures in the financial documents?"
    )
    print(xml_formatted)
