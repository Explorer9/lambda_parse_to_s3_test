import json
import boto3
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import jwt
import time
from datetime import datetime, timedelta

def lambda_handler(event, context):
    """
    Fetch SharePoint list items using Azure AD app-only authentication
    """
    
    # Configuration - replace with your actual values
    TENANT_ID = "your-tenant-id"
    SITE_URL = "https://yourdomain.sharepoint.com/sites/yoursite"
    LIST_NAME = "YourListName"  # or use list GUID
    SECRET_NAME = "your-secret-manager-secret-name"
    CERTIFICATE_S3_BUCKET = "your-bucket-name"
    CERTIFICATE_S3_KEY = "path/to/sharepoint.crt"
    
    try:
        # 1. Get secrets from AWS Secrets Manager
        secrets_client = boto3.client('secretsmanager')
        secret_response = secrets_client.get_secret_value(SecretId=SECRET_NAME)
        secrets = json.loads(secret_response['SecretString'])
        
        client_id = secrets['client_id']
        private_key_pem = secrets['private_key']
        
        # 2. Get certificate from S3 (if needed for verification)
        s3_client = boto3.client('s3')
        # cert_obj = s3_client.get_object(Bucket=CERTIFICATE_S3_BUCKET, Key=CERTIFICATE_S3_KEY)
        # certificate = cert_obj['Body'].read().decode('utf-8')
        
        # 3. Get Access Token using OAuth2 with certificate
        access_token = get_access_token(tenant_id=TENANT_ID, 
                                       client_id=client_id, 
                                       private_key_pem=private_key_pem,
                                       site_url=SITE_URL)
        
        # 4. Fetch SharePoint List
        list_items = get_sharepoint_list(site_url=SITE_URL,
                                        list_name=LIST_NAME,
                                        access_token=access_token)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Successfully fetched SharePoint list',
                'itemCount': len(list_items),
                'items': list_items
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }


def get_access_token(tenant_id, client_id, private_key_pem, site_url):
    """
    Get access token using certificate-based authentication
    """
    # Extract domain from site URL
    domain = site_url.split('/')[2]
    resource = f"https://{domain}"
    
    # Create JWT assertion
    now = int(time.time())
    payload = {
        'aud': f'https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token',
        'iss': client_id,
        'sub': client_id,
        'jti': str(now),
        'nbf': now,
        'exp': now + 3600,
    }
    
    # Load private key
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode(),
        password=None,
        backend=default_backend()
    )
    
    # Sign JWT
    assertion = jwt.encode(
        payload,
        private_key,
        algorithm='RS256',
        headers={'x5t': get_thumbprint_from_key(private_key_pem)}  # Optional
    )
    
    # Request token
    token_url = f'https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token'
    
    data = {
        'client_id': client_id,
        'client_assertion_type': 'urn:ietf:params:oauth:client-assertion-type:jwt-bearer',
        'client_assertion': assertion,
        'scope': f'{resource}/.default',
        'grant_type': 'client_credentials'
    }
    
    response = requests.post(token_url, data=data)
    response.raise_for_status()
    
    token_response = response.json()
    return token_response['access_token']


def get_thumbprint_from_key(private_key_pem):
    """
    Optional: Calculate certificate thumbprint if needed
    You might need to adjust this based on your certificate
    """
    import hashlib
    import base64
    # This is a simplified version - you may need the actual certificate for thumbprint
    return base64.b64encode(hashlib.sha1(private_key_pem.encode()).digest()).decode()


def get_sharepoint_list(site_url, list_name, access_token):
    """
    Fetch items from SharePoint list using REST API
    """
    # Build API endpoint
    api_url = f"{site_url}/_api/web/lists/getbytitle('{list_name}')/items"
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json;odata=verbose',
        'Content-Type': 'application/json;odata=verbose'
    }
    
    # Optional: Add query parameters for filtering, selecting specific columns, etc.
    params = {
        '$top': 5000,  # Adjust as needed
        # '$select': 'Title,ID,Created,Modified',  # Specify columns you want
        # '$filter': "Status eq 'Active'",  # Add filters if needed
    }
    
    all_items = []
    
    while api_url:
        response = requests.get(api_url, headers=headers, params=params if api_url == f"{site_url}/_api/web/lists/getbytitle('{list_name}')/items" else None)
        response.raise_for_status()
        
        data = response.json()
        items = data['d']['results']
        all_items.extend(items)
        
        # Handle pagination
        api_url = data['d'].get('__next')
        params = None  # Clear params for subsequent requests
    
    return all_items


{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "secretsmanager:GetSecretValue"
            ],
            "Resource": "arn:aws:secretsmanager:region:account:secret:your-secret-name"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject"
            ],
            "Resource": "arn:aws:s3:::your-bucket/path/to/sharepoint.crt"
        }
    ]
}





import json
import boto3
import os
from typing import Dict, List, Any

# Initialize Kendra client
kendra = boto3.client('kendra')

def lambda_handler(event, context):
    """
    Lambda function to query Kendra index and retrieve results with metadata
    
    Expected event structure:
    {
        "query": "your search query",
        "index_id": "your-kendra-index-id",  # Optional if set as env variable
        "max_results": 10  # Optional, default is 10
    }
    """
    
    try:
        # Extract parameters from event
        query = event.get('query')
        index_id = event.get('index_id', os.environ.get('KENDRA_INDEX_ID'))
        max_results = event.get('max_results', 10)
        
        # Validate required parameters
        if not query:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Query parameter is required'})
            }
        
        if not index_id:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Kendra index_id is required'})
            }
        
        # Query Kendra
        response = kendra.query(
            IndexId=index_id,
            QueryText=query,
            PageSize=max_results,
            AttributeFilter={
                'AndAllFilters': []  # Add filters here if needed
            }
        )
        
        # Process results
        results = process_kendra_results(response)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'query': query,
                'total_results': len(results),
                'results': results
            }, indent=2, default=str)
        }
        
    except Exception as e:
        print(f"Error querying Kendra: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }


def process_kendra_results(response: Dict) -> List[Dict[str, Any]]:
    """
    Process Kendra query results and extract all metadata
    
    Args:
        response: Kendra query response
        
    Returns:
        List of processed results with metadata
    """
    processed_results = []
    
    # Process query result items
    if 'ResultItems' in response:
        for idx, item in enumerate(response['ResultItems']):
            result = {
                'rank': idx + 1,
                'id': item.get('Id'),
                'type': item.get('Type'),  # DOCUMENT or QUESTION_ANSWER or ANSWER
                'score': item.get('ScoreAttributes', {}).get('ScoreConfidence'),
                'document_title': item.get('DocumentTitle', {}).get('Text'),
                'document_excerpt': item.get('DocumentExcerpt', {}).get('Text'),
                'document_uri': item.get('DocumentURI'),
                'feedback_token': item.get('FeedbackToken'),
                'metadata': {}
            }
            
            # Extract all document attributes (metadata/custom fields)
            if 'DocumentAttributes' in item:
                for attr in item['DocumentAttributes']:
                    key = attr.get('Key')
                    value = attr.get('Value')
                    
                    # Handle different value types
                    if value:
                        if 'StringValue' in value:
                            result['metadata'][key] = value['StringValue']
                        elif 'StringListValue' in value:
                            result['metadata'][key] = value['StringListValue']
                        elif 'LongValue' in value:
                            result['metadata'][key] = value['LongValue']
                        elif 'DateValue' in value:
                            result['metadata'][key] = value['DateValue']
            
            # Extract highlights (relevant text excerpts)
            if 'DocumentExcerpt' in item and 'Highlights' in item['DocumentExcerpt']:
                result['highlights'] = []
                for highlight in item['DocumentExcerpt']['Highlights']:
                    result['highlights'].append({
                        'begin_offset': highlight.get('BeginOffset'),
                        'end_offset': highlight.get('EndOffset'),
                        'top_answer': highlight.get('TopAnswer', False),
                        'type': highlight.get('Type')
                    })
            
            # Additional excerpt information if available
            if 'AdditionalAttributes' in item:
                result['additional_attributes'] = []
                for add_attr in item['AdditionalAttributes']:
                    result['additional_attributes'].append({
                        'key': add_attr.get('Key'),
                        'value': add_attr.get('Value', {}).get('TextWithHighlightsValue', {}).get('Text')
                    })
            
            processed_results.append(result)
    
    return processed_results


# Example usage for testing locally
if __name__ == "__main__":
    # Test event
    test_event = {
        "query": "What is the project status?",
        "index_id": "your-index-id-here",
        "max_results": 5
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2, default=str))
