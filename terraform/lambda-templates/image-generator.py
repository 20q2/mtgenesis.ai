import json
import boto3
import os
import logging
import base64
import uuid
from typing import Dict, Any
from urllib.parse import urlparse
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_secrets():
    """Retrieve API keys from AWS Secrets Manager"""
    secrets_client = boto3.client('secretsmanager')
    secrets_arn = os.environ['SECRETS_ARN']
    
    try:
        response = secrets_client.get_secret_value(SecretId=secrets_arn)
        secrets = json.loads(response['SecretString'])
        return secrets
    except Exception as e:
        logger.error(f"Error retrieving secrets: {str(e)}")
        return {}

def upload_to_s3(image_data: bytes, filename: str) -> str:
    """Upload image to S3 and return public URL"""
    s3_client = boto3.client('s3')
    bucket_name = os.environ['S3_BUCKET']
    
    try:
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=f"cards/{filename}",
            Body=image_data,
            ContentType='image/png',
            ACL='public-read'
        )
        
        # Return public URL
        return f"https://{bucket_name}.s3.amazonaws.com/cards/{filename}"
    except Exception as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        raise

def generate_with_stability_ai(prompt: str, api_key: str) -> bytes:
    """Generate image using Stability AI API"""
    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    data = {
        "text_prompts": [
            {
                "text": prompt,
                "weight": 1
            }
        ],
        "cfg_scale": 7,
        "height": 1024,
        "width": 1024,
        "samples": 1,
        "steps": 30,
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        response_data = response.json()
        image_data = base64.b64decode(response_data["artifacts"][0]["base64"])
        return image_data
    except Exception as e:
        logger.error(f"Stability AI API error: {str(e)}")
        raise

def generate_with_openai_dalle(prompt: str, api_key: str) -> bytes:
    """Generate image using OpenAI DALL-E API"""
    try:
        import openai
        openai.api_key = api_key
        
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024",
            response_format="url"
        )
        
        image_url = response['data'][0]['url']
        
        # Download the image
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        
        return image_response.content
    except Exception as e:
        logger.error(f"OpenAI DALL-E API error: {str(e)}")
        raise

def generate_with_bedrock(prompt: str) -> bytes:
    """Generate image using AWS Bedrock (Titan Image Generator)"""
    try:
        bedrock = boto3.client('bedrock-runtime')
        
        body = json.dumps({
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt,
                "negativeText": "blurry, low quality, distorted"
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": 1024,
                "width": 1024,
                "cfgScale": 7.0,
                "seed": 42
            }
        })
        
        response = bedrock.invoke_model(
            body=body,
            modelId="amazon.titan-image-generator-v1",
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get('body').read())
        image_data = base64.b64decode(response_body.get('images')[0])
        return image_data
    except Exception as e:
        logger.error(f"Bedrock API error: {str(e)}")
        raise

def build_art_prompt(card_data: Dict[str, Any]) -> str:
    """Build an art prompt from card data"""
    name = card_data.get('name', 'Magic Card')
    card_type = card_data.get('type', '')
    colors = card_data.get('colors', [])
    description = card_data.get('description', '')
    art_prompt = card_data.get('artPrompt', '')
    
    # Base prompt
    if art_prompt:
        prompt = art_prompt
    else:
        prompt = f"Fantasy art of {name}"
    
    # Add color themes
    color_themes = {
        'W': 'bright, holy, peaceful, white and gold tones',
        'U': 'mystical, intellectual, blue and silver tones, arcane symbols',
        'B': 'dark, sinister, black and purple tones, shadowy',
        'R': 'fiery, aggressive, red and orange tones, chaotic energy',
        'G': 'natural, wild, green tones, forest and nature themes'
    }
    
    if colors:
        themes = [color_themes.get(color, '') for color in colors if color in color_themes]
        if themes:
            prompt += f", {', '.join(themes)}"
    
    # Add type-specific elements
    if 'creature' in card_type.lower():
        prompt += ", detailed character design, dynamic pose"
    elif 'instant' in card_type.lower() or 'sorcery' in card_type.lower():
        prompt += ", magical spell effects, energy swirls"
    elif 'artifact' in card_type.lower():
        prompt += ", mechanical design, metallic textures"
    elif 'enchantment' in card_type.lower():
        prompt += ", ethereal effects, magical aura"
    elif 'land' in card_type.lower():
        prompt += ", landscape view, environmental scene"
    
    # Quality modifiers
    prompt += ", high quality, detailed, professional fantasy art, Magic: The Gathering style, dramatic lighting"
    
    return prompt

def handler(event, context):
    """Lambda handler for card image generation"""
    try:
        # Parse request body
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event
        
        # Build art prompt
        art_prompt = build_art_prompt(body)
        logger.info(f"Generated art prompt: {art_prompt}")
        
        # Get secrets
        secrets = get_secrets()
        
        # Generate image based on available services
        image_data = None
        use_bedrock = os.environ.get('USE_BEDROCK', 'false').lower() == 'true'
        
        if use_bedrock:
            image_data = generate_with_bedrock(art_prompt)
        elif secrets.get('stability_api_key'):
            image_data = generate_with_stability_ai(art_prompt, secrets['stability_api_key'])
        elif secrets.get('openai_api_key'):
            image_data = generate_with_openai_dalle(art_prompt, secrets['openai_api_key'])
        else:
            raise ValueError("No valid image generation service available")
        
        # Generate unique filename
        card_name = body.get('name', 'card').lower().replace(' ', '-')
        filename = f"{card_name}-{uuid.uuid4().hex[:8]}.png"
        
        # Upload to S3
        image_url = upload_to_s3(image_data, filename)
        
        response = {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps({
                'imageUrl': image_url,
                'filename': filename,
                'prompt': art_prompt,
                'success': True
            })
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in image generation: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e),
                'success': False
            })
        }

