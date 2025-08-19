def format_bedrock_chunks_hybrid(bedrock_response, user_query=None):
    """
    Hybrid format: Plain text disclosures + minimal XML for documents.
    Optimized for token efficiency while maintaining structure.
    
    Args:
        bedrock_response (dict): Response from Bedrock retriever
        user_query (str, optional): User's question to include in context
    
    Returns:
        str: Hybrid-formatted context string for LLM
    """
    
    if 'retrievalResults' not in bedrock_response or not bedrock_response['retrievalResults']:
        return "No retrieval results found."
    
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
    
    # Start building the context
    context_parts = []
    
    # Add combined disclosures section (plain text)
    if all_disclosures:
        context_parts.append("DISCLOSURES:")
        disclosure_text = " | ".join(all_disclosures)
        context_parts.append(disclosure_text)
        context_parts.append("")  # Empty line
    
    # Add documents in minimal XML
    valid_docs = []
    for idx, result in enumerate(sorted_results, 1):
        # Extract text content
        content = result.get('content', {})
        text_content = _extract_text_content(content)
        
        if not text_content:
            continue
        
        # Extract location
        location = result.get('location', {})
        source_info = _extract_location_info(location)
        
        # Extract other metadata (excluding verbatim_items)
        metadata = result.get('metadata', {})
        other_metadata = _format_other_metadata(metadata)
        
        score = result.get('score', 0)
        
        # Build minimal XML for this document
        doc_attrs = [f'id="{idx}"', f'score="{score:.2f}"']
        if source_info:
            doc_attrs.append(f'src="{source_info}"')
        
        doc_line = f'<doc {" ".join(doc_attrs)}>'
        
        # Add metadata if exists
        if other_metadata:
            doc_line += f'<meta>{other_metadata}</meta>'
        
        # Add content and close tag
        doc_line += f'{text_content}</doc>'
        
        valid_docs.append(doc_line)
    
    # Add all documents
    if valid_docs:
        context_parts.extend(valid_docs)
    
    # Add user query if provided
    if user_query:
        context_parts.append("")
        context_parts.append(f"QUERY: {user_query}")
    
    return "\n".join(context_parts)


def _extract_text_content(content_dict):
    """Extract text content from the content dictionary."""
    if not content_dict:
        return ""
    
    content_type = content_dict.get('type', 'TEXT')
    
    if content_type == 'TEXT' and 'text' in content_dict:
        return content_dict['text']
    elif 'byteContent' in content_dict:
        return content_dict['byteContent']
    elif content_type == 'ROW' and 'row' in content_dict:
        row_data = content_dict['row']
        row_text = []
        for col in row_data:
            if col.get('type') == 'STRING':
                row_text.append(f"{col.get('columnName', '')}: {col.get('columnValue', '')}")
        return " | ".join(row_text)
    
    return ""


def _extract_location_info(location_dict):
    """Extract location information in a compact format."""
    if not location_dict:
        return ""
    
    location_type = location_dict.get('type', '')
    
    if location_type == 'S3' and 's3Location' in location_dict:
        uri = location_dict['s3Location'].get('uri', '')
        # Shorten S3 URIs for token efficiency
        return uri.split('/')[-1] if '/' in uri else uri
    elif location_type == 'WEB' and 'webLocation' in location_dict:
        return location_dict['webLocation'].get('url', '')
    elif location_type == 'CONFLUENCE' and 'confluenceLocation' in location_dict:
        return location_dict['confluenceLocation'].get('url', '')
    elif location_type == 'SALESFORCE' and 'salesforceLocation' in location_dict:
        return location_dict['salesforceLocation'].get('url', '')
    elif location_type == 'SHAREPOINT' and 'sharePointLocation' in location_dict:
        return location_dict['sharePointLocation'].get('url', '')
    elif location_type == 'CUSTOM' and 'customDocumentLocation' in location_dict:
        return location_dict['customDocumentLocation'].get('id', '')
    elif location_type == 'KENDRA' and 'kendraDocumentLocation' in location_dict:
        return location_dict['kendraDocumentLocation'].get('uri', '')
    elif location_type == 'SQL' and 'sqlLocation' in location_dict:
        return f"SQL:{location_dict['sqlLocation'].get('query', '')[:50]}..."
    
    return location_type


def _format_other_metadata(metadata_dict):
    """Format non-verbatim metadata in a compact way."""
    if not metadata_dict:
        return ""
    
    metadata_parts = []
    for key, value in metadata_dict.items():
        if key != 'verbatim_items' and value is not None:
            if isinstance(value, list):
                formatted_value = ",".join([str(v) for v in value])
            else:
                formatted_value = str(value)
            metadata_parts.append(f"{key}:{formatted_value}")
    
    return " | ".join(metadata_parts)


# Enhanced version with even more token optimization
def format_bedrock_chunks_ultra_compact(bedrock_response, user_query=None):
    """
    Ultra-compact hybrid format for maximum token efficiency.
    """
    if 'retrievalResults' not in bedrock_response or not bedrock_response['retrievalResults']:
        return "No results."
    
    results = bedrock_response['retrievalResults']
    sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
    
    # Collect unique disclosures
    disclosures = []
    seen = set()
    for result in sorted_results:
        items = result.get('metadata', {}).get('verbatim_items', [])
        for item in items:
            if item and str(item) not in seen:
                disclosures.append(str(item))
                seen.add(str(item))
    
    parts = []
    
    # Compact disclosures
    if disclosures:
        parts.append(f"DISC: {' | '.join(disclosures)}\n")
    
    # Ultra-compact documents
    for i, result in enumerate(sorted_results, 1):
        text = _extract_text_content(result.get('content', {}))
        if not text:
            continue
            
        src = _extract_location_info(result.get('location', {}))
        score = result.get('score', 0)
        
        # One-line format: [id:score:src] content
        src_part = f":{src}" if src else ""
        parts.append(f"[{i}:{score:.2f}{src_part}] {text}")
    
    if user_query:
        parts.append(f"\nQ: {user_query}")
    
    return "\n".join(parts)


# Example usage
if __name__ == "__main__":
    # Sample test data
    sample_response = {
        'retrievalResults': [
            {
                'content': {
                    'text': 'Financial risk information and market analysis details.',
                    'type': 'TEXT'
                },
                'location': {
                    'type': 'S3',
                    's3Location': {
                        'uri': 's3://financial-docs/annual-report-2023.pdf'
                    }
                },
                'metadata': {
                    'document_type': 'financial',
                    'year': 2023,
                    'verbatim_items': ['Risk disclosure: Market volatility', 'Legal: Past performance disclaimer']
                },
                'score': 0.95
            },
            {
                'content': {
                    'text': 'Compliance requirements and regulatory framework overview.',
                    'type': 'TEXT'
                },
                'location': {
                    'type': 'WEB',
                    'webLocation': {
                        'url': 'https://compliance.example.com/guide'
                    }
                },
                'metadata': {
                    'category': 'compliance',
                    'verbatim_items': ['Regulatory: Must comply with all laws']
                },
                'score': 0.87
            }
        ]
    }
    
    print("=== HYBRID FORMAT ===")
    hybrid_result = format_bedrock_chunks_hybrid(sample_response, "What are the compliance requirements?")
    print(hybrid_result)
    
    print("\n=== ULTRA-COMPACT FORMAT ===")
    compact_result = format_bedrock_chunks_ultra_compact(sample_response, "What are the compliance requirements?")
    print(compact_result)
