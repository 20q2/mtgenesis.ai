import json
import boto3
import os
import logging
from typing import Dict, Any

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

def generate_with_openai(prompt: str, api_key: str) -> str:
    """Generate card text using OpenAI API"""
    try:
        import openai
        openai.api_key = api_key
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert Magic: The Gathering card designer. Generate creative and balanced card text that follows MTG rules and conventions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.8
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise

def generate_with_anthropic(prompt: str, api_key: str) -> str:
    """Generate card text using Anthropic Claude API"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=500,
            temperature=0.8,
            system="You are an expert Magic: The Gathering card designer. Generate creative and balanced card text that follows MTG rules and conventions.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text.strip()
    except Exception as e:
        logger.error(f"Anthropic API error: {str(e)}")
        raise

def generate_with_bedrock(prompt: str) -> str:
    """Generate card text using AWS Bedrock"""
    try:
        bedrock = boto3.client('bedrock-runtime')
        
        body = json.dumps({
            "prompt": f"\n\nHuman: You are an expert Magic: The Gathering card designer. {prompt}\n\nAssistant:",
            "max_tokens_to_sample": 500,
            "temperature": 0.8,
            "top_p": 0.9
        })
        
        response = bedrock.invoke_model(
            body=body,
            modelId="anthropic.claude-v2",
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get('body').read())
        return response_body.get('completion', '').strip()
    except Exception as e:
        logger.error(f"Bedrock API error: {str(e)}")
        raise

def handler(event, context):
    """Lambda handler for card text generation"""
    try:
        # Parse request body
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event
        
        # Extract parameters
        card_name = body.get('name', '')
        card_type = body.get('type', '')
        mana_cost = body.get('manaCost', '')
        colors = body.get('colors', [])
        rarity = body.get('rarity', 'common')
        subtype = body.get('subtype', '')
        
        # Build prompt
        prompt = f"""Generate Magic: The Gathering card text for a card with these parameters:
        
Name: {card_name}
Type: {card_type}
Subtype: {subtype}
Mana Cost: {mana_cost}
Colors: {', '.join(colors)}
Rarity: {rarity}
        
Generate appropriate card text that:
1. Fits the card's theme and colors
2. Is balanced for the mana cost and rarity
3. Follows proper MTG templating
4. Is creative but not overpowered
        
Return only the card text without any additional explanation."""
        
        # Get secrets
        secrets = get_secrets()
        
        # Try different AI services based on availability
        generated_text = ""
        use_bedrock = os.environ.get('USE_BEDROCK', 'false').lower() == 'true'
        
        if use_bedrock:
            generated_text = generate_with_bedrock(prompt)
        elif secrets.get('openai_api_key'):
            generated_text = generate_with_openai(prompt, secrets['openai_api_key'])
        elif secrets.get('anthropic_api_key'):
            generated_text = generate_with_anthropic(prompt, secrets['anthropic_api_key'])
        else:
            raise ValueError("No valid AI service available")
        
        # Generate power/toughness for creatures
        power = None
        toughness = None
        if 'creature' in card_type.lower():
            # Simple power/toughness generation based on mana cost
            cmc = body.get('cmc', 0)
            if cmc <= 1:
                power, toughness = "1", "1"
            elif cmc <= 3:
                power, toughness = str(cmc), str(cmc)
            else:
                power, toughness = str(cmc - 1), str(cmc)
        
        response = {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps({
                'description': generated_text,
                'power': power,
                'toughness': toughness,
                'success': True
            })
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in text generation: {str(e)}")
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

