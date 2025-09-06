# Lambda Proxy Deployment Guide

## Quick Setup

1. **Update ngrok URL** in `lambda_function.py`:
   ```python
   NGROK_URL = "https://your-current-ngrok-url.ngrok-free.app"
   ```

2. **Create Lambda function**:
   - Go to AWS Lambda console
   - Create new function: "mtgenesis-proxy"
   - Runtime: Python 3.11
   - Copy `lambda_function.py` content into the code editor

3. **Set up API Gateway**:
   - Create new REST API
   - Create resource: `{proxy+}` 
   - Create method: `ANY` on `{proxy+}`
   - Integration type: Lambda Function
   - Enable Lambda Proxy integration
   - Deploy API to stage (e.g., "prod")

4. **Update your Angular environment**:
   ```typescript
   apiUrl: 'https://your-api-gateway-url.amazonaws.com/prod'
   ```

## Benefits
- ✅ Proper CORS handling
- ✅ No browser/ngrok compatibility issues  
- ✅ Handles timeouts gracefully
- ✅ Works with any HTTPS frontend
- ✅ Easy to update ngrok URL

## Testing
Once deployed, test with:
```bash
curl -X POST https://your-api-gateway-url.amazonaws.com/prod/api/v1/create_card \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "cardData": {"name": "Test"}}'
```