def format_bedrock_chunks_xml(bedrock_response, user_query=None):
    """
    Single function to format Bedrock retrieval results in XML style with combined disclosures.
    
    Args:
        bedrock_response (dict): Response from Bedrock retriever
        user_query (str, optional): User's question to include in context
    
    Returns:
        str: XML-formatted context string for LLM with combined disclosures
    """
    
    if 'retrievalResults' not in bedrock_response or not bedrock_response['retrievalResults']:
        return "<context>No retrieval results found.</context>"
    
    results = bedrock_response['retrievalResults']
    
    # Sort by relevance score (highest first)
    sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
    
    # Collect all unique disclosures
    all_disclosures = []
    seen_disclosures = set()
    
    for result in sorted_results:
        metadata = result.get('metadata', {})
        verbatim_items = metadata.get('verbatim_items', [])
        
        if isinstance(verbatim_items, list):
            for item in verbatim_items:
                if item and str(item) not in seen_disclosures:
                    all_disclosures.append(str(item))
                    seen_disclosures.add(str(item))
    
    # Build XML structure
    xml_parts = ["<context>"]
    
    # Add combined disclosures section if any exist
    if all_disclosures:
        xml_parts.append("  <disclosures>")
        for disclosure in all_disclosures:
            xml_parts.append(f"    <item>{disclosure}</item>")
        xml_parts.append("  </disclosures>")
    
    # Add documents section
    xml_parts.append("  <documents>")
    
    for idx, result in enumerate(sorted_results, 1):
        # Extract text content
        content = result.get('content', {})
        text_content = ""
        
        if content.get('type') == 'TEXT' and 'text' in content:
            text_content = content['text']
        elif 'byteContent' in content:
            text_content = content['byteContent']
        elif content.get('type') == 'ROW' and 'row' in content:
            row_data = content['row']
            row_text = []
            for col in row_data:
                if col.get('type') == 'STRING':
                    row_text.append(f"{col.get('columnName', '')}: {col.get('columnValue', '')}")
            text_content = " | ".join(row_text)
        
        if not text_content:
            continue
        
        # Extract location
        location = result.get('location', {})
        location_type = location.get('type', '')
        source_info = ""
        
        if location_type == 'S3' and 's3Location' in location:
            source_info = location['s3Location'].get('uri', '')
        elif location_type == 'WEB' and 'webLocation' in location:
            source_info = location['webLocation'].get('url', '')
        elif location_type == 'CONFLUENCE' and 'confluenceLocation' in location:
            source_info = location['confluenceLocation'].get('url', '')
        elif location_type == 'SALESFORCE' and 'salesforceLocation' in location:
            source_info = location['salesforceLocation'].get('url', '')
        elif location_type == 'SHAREPOINT' and 'sharePointLocation' in location:
            source_info = location['sharePointLocation'].get('url', '')
        elif location_type == 'CUSTOM' and 'customDocumentLocation' in location:
            source_info = f"Document ID: {location['customDocumentLocation'].get('id', '')}"
        elif location_type == 'KENDRA' and 'kendraDocumentLocation' in location:
            source_info = location['kendraDocumentLocation'].get('uri', '')
        elif location_type == 'SQL' and 'sqlLocation' in location:
            source_info = f"SQL Query: {location['sqlLocation'].get('query', '')}"
        else:
            source_info = f"{location_type} location" if location_type else "Unknown source"
        
        # Extract other metadata (excluding verbatim_items)
        metadata = result.get('metadata', {})
        other_metadata = []
        for key, value in metadata.items():
            if key != 'verbatim_items' and value is not None:
                if isinstance(value, list):
                    formatted_value = ", ".join([str(v) for v in value])
                else:
                    formatted_value = str(value)
                other_metadata.append(f"{key}: {formatted_value}")
        
        # Build document XML
        score = result.get('score', 0)
        xml_parts.append(f'    <document id="{idx}" score="{score:.3f}">')
        xml_parts.append(f'      <source>{source_info}</source>')
        
        if other_metadata:
            xml_parts.append('      <metadata>')
            for meta_item in other_metadata:
                xml_parts.append(f'        <item>{meta_item}</item>')
            xml_parts.append('      </metadata>')
        
        xml_parts.append(f'      <content>{text_content}</content>')
        xml_parts.append('    </document>')
    
    xml_parts.append("  </documents>")
    xml_parts.append("</context>")
    
    # Add query if provided
    if user_query:
        xml_parts.append("")
        xml_parts.append(f"<query>{user_query}</query>")
    
    return "\n".join(xml_parts)


# Example usage
if __name__ == "__main__":
    # Sample test data
    sample_response = {
        'retrievalResults': [
            {
                'content': {
                    'text': 'Financial risk information and market analysis.',
                    'type': 'TEXT'
                },
                'location': {
                    'type': 'S3',
                    's3Location': {
                        'uri': 's3://docs/financial-report.pdf'
                    }
                },
                'metadata': {
                    'document_type': 'financial',
                    'verbatim_items': ['d1', 'd2', 'd3']
                },
                'score': 0.95
            },
            {
                'content': {
                    'text': 'Additional compliance and regulatory content.',
                    'type': 'TEXT'
                },
                'location': {
                    'type': 'WEB',
                    'webLocation': {
                        'url': 'https://example.com/compliance'
                    }
                },
                'metadata': {
                    'category': 'compliance',
                    'verbatim_items': ['d2', 'd4']  # d2 is duplicate, will be filtered
                },
                'score': 0.87
            }
        ]
    }
    
    # Test the function
    result = format_bedrock_chunks_xml(
        sample_response, 
        "What are the key disclosures?"
    )
    
    print(result)
