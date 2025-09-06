import json
import urllib3
from typing import Dict, Any

# Create a PoolManager instance
http = urllib3.PoolManager()

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function to proxy requests to ngrok tunnel and handle CORS
    """
    
    # Your ngrok URL - update this when ngrok URL changes
    NGROK_URL = "https://0eccb5a667ac.ngrok-free.app"
    
    # Default CORS headers
    cors_headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,Authorization,ngrok-skip-browser-warning,Accept,Cache-Control',
        'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS',
        'Access-Control-Max-Age': '86400'
    }
    
    try:
        # Handle both API Gateway and Function URL event formats
        if 'requestContext' in event and 'http' in event['requestContext']:
            # Function URL format
            method = event['requestContext']['http']['method']
            path = event['requestContext']['http']['path']
            query_params = event.get('queryStringParameters', {}) or {}
            headers = event.get('headers', {})
            body = event.get('body', '')
        else:
            # API Gateway format
            method = event.get('httpMethod', 'GET')
            path = event.get('path', '/')
            query_params = event.get('queryStringParameters', {}) or {}
            headers = event.get('headers', {})
            body = event.get('body', '')
        
        print(f"Detected method: {method}, path: {path}")
        
        # Handle OPTIONS preflight request
        if method == 'OPTIONS':
            print("Handling OPTIONS preflight request")
            return {
                'statusCode': 200,
                'headers': cors_headers,
                'body': json.dumps({'status': 'ok'})
            }
        
        # Add ngrok bypass header
        proxy_headers = {
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'any',
            'User-Agent': 'MTGenesis-Lambda-Proxy/1.0'
        }
        
        # Forward original headers (except Host)
        for key, value in headers.items():
            if key.lower() not in ['host', 'content-length']:
                proxy_headers[key] = value
        
        # Build target URL
        target_url = f"{NGROK_URL}{path}"
        if query_params:
            query_string = "&".join([f"{k}={v}" for k, v in query_params.items()])
            target_url += f"?{query_string}"
        
        print(f"Proxying {method} request to: {target_url}")
        print(f"Request body size: {len(body) if body else 0} bytes")
        
        # Only handle POST requests (OPTIONS handled above)
        if method != 'POST':
            print(f"Unsupported method: {method}")
            return {
                'statusCode': 405,
                'headers': cors_headers,
                'body': json.dumps({'error': 'Method not allowed. Only POST requests are supported.'})
            }
        
        # Make POST request to ngrok
        response = http.request(
            'POST',
            target_url,
            body=body,
            headers=proxy_headers,
            timeout=300.0,  # 5 minute timeout
            retries=urllib3.Retry(total=0)  # Disable retries for faster failures
        )
        
        print(f"Response status: {response.status}")
        print(f"Response size: {len(response.data)} bytes")
        
        # Get response body
        response_body = response.data.decode('utf-8')
        
        # Try to parse as JSON for pretty logging
        try:
            parsed_body = json.loads(response_body)
            print(f"Response JSON keys: {list(parsed_body.keys()) if isinstance(parsed_body, dict) else 'Not a dict'}")
        except:
            print("Response is not JSON")
        
        # Return successful response with CORS headers
        return {
            'statusCode': response.status,
            'headers': {**cors_headers, 'Content-Type': 'application/json'},
            'body': response_body
        }
        
    except urllib3.exceptions.TimeoutError:
        print("Request timed out")
        return {
            'statusCode': 504,
            'headers': cors_headers,
            'body': json.dumps({
                'error': 'Gateway timeout - request took too long',
                'details': 'The backend server did not respond within the timeout period'
            })
        }
        
    except Exception as e:
        print(f"Proxy error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        return {
            'statusCode': 502,
            'headers': cors_headers,
            'body': json.dumps({
                'error': 'Bad Gateway - proxy error',
                'details': str(e),
                'type': type(e).__name__
            })
        }