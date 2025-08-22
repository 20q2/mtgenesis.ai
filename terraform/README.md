# MTGenesis.ai Terraform Infrastructure

This Terraform configuration deploys the AI infrastructure needed for generating Magic: The Gathering card text and artwork.

## Features

- **Text Generation**: Lambda function supporting OpenAI GPT-4, Anthropic Claude, or AWS Bedrock
- **Image Generation**: Lambda function supporting Stability AI, OpenAI DALL-E, or AWS Bedrock
- **Secure Storage**: AWS Secrets Manager for API keys
- **Image Storage**: S3 bucket for generated card images with public access
- **Scalable**: Serverless architecture that scales automatically

## Prerequisites

1. **AWS CLI** configured with appropriate credentials
2. **Terraform** (>= 1.0) installed
3. **API Keys** for at least one of the following:
   - OpenAI API key
   - Anthropic API key
   - Stability AI API key
   - AWS Bedrock access (if using AWS-only solution)

## Quick Start

1. **Clone and navigate to the terraform directory**:
   ```bash
   cd terraform
   ```

2. **Copy the example variables file**:
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   ```

3. **Edit terraform.tfvars with your configuration**:
   ```bash
   # Edit the file with your preferred editor
   notepad terraform.tfvars  # Windows
   nano terraform.tfvars     # Linux/Mac
   ```

4. **Initialize Terraform**:
   ```bash
   terraform init
   ```

5. **Plan the deployment**:
   ```bash
   terraform plan
   ```

6. **Deploy the infrastructure**:
   ```bash
   terraform apply
   ```

7. **Note the outputs** - you'll need the Lambda function URLs for your application.

## Configuration Options

### AI Service Options

- **AWS Bedrock** (Recommended): No external API keys needed, stays within AWS
- **OpenAI**: Requires OpenAI API key, supports GPT-4 and DALL-E
- **Anthropic**: Requires Anthropic API key, supports Claude models
- **Stability AI**: Requires Stability AI API key, supports Stable Diffusion

### Infrastructure Options

- **enable_bedrock**: Use AWS Bedrock for AI models (recommended)
- **enable_gpu_instance**: Deploy GPU instances for self-hosted models (advanced)
- **lambda_timeout**: Maximum execution time for Lambda functions
- **lambda_memory**: Memory allocation for Lambda functions

## API Usage

After deployment, you'll have two REST API endpoints:

### Text Generation

**POST** to the text generation URL with:

```json
{
  "name": "Lightning Bolt",
  "type": "Instant",
  "colors": ["R"],
  "manaCost": "{R}",
  "rarity": "common",
  "subtype": "",
  "cmc": 1
}
```

**Response**:

```json
{
  "description": "Lightning Bolt deals 3 damage to any target.",
  "power": null,
  "toughness": null,
  "success": true
}
```

### Image Generation

**POST** to the image generation URL with:

```json
{
  "name": "Lightning Bolt",
  "type": "Instant",
  "colors": ["R"],
  "artPrompt": "A bolt of lightning striking through stormy clouds"
}
```

**Response**:

```json
{
  "imageUrl": "https://your-bucket.s3.amazonaws.com/cards/lightning-bolt-abc123.png",
  "filename": "lightning-bolt-abc123.png",
  "prompt": "A bolt of lightning striking through stormy clouds, fiery, aggressive, red and orange tones, chaotic energy, magical spell effects, energy swirls, high quality, detailed, professional fantasy art, Magic: The Gathering style, dramatic lighting",
  "success": true
}
```

## Cost Optimization

- **AWS Bedrock**: Most cost-effective for moderate usage
- **External APIs**: Pay per request, good for development
- **GPU Instances**: Only enable for high-volume production use

## Security

- API keys are stored securely in AWS Secrets Manager
- Lambda functions have minimal required permissions
- S3 bucket allows public read access for generated images only

## Updating API Keys

To update your API keys after deployment:

1. Go to AWS Secrets Manager in the console
2. Find the secret named `mtgenesis-ai-api-keys-*`
3. Edit the secret values
4. The Lambda functions will use the new keys immediately

## Cleanup

To destroy all resources:

```bash
terraform destroy
```

**Warning**: This will delete all generated images in S3!

## Troubleshooting

### Common Issues

1. **"No valid AI service available"**: Check that your API keys are correctly set in Secrets Manager
2. **Lambda timeout errors**: Increase `lambda_timeout` in terraform.tfvars
3. **S3 access denied**: Ensure the bucket policy is correctly applied

### Monitoring

Check CloudWatch logs for Lambda function execution details:
- `/aws/lambda/mtgenesis-ai-text-generator-*`
- `/aws/lambda/mtgenesis-ai-image-generator-*`

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Angular App   │───▶│  Lambda Function │───▶│   AI Services   │
│                 │    │  (Text/Image)    │    │ (OpenAI/Claude/ │
└─────────────────┘    └──────────────────┘    │  Stability/etc) │
                                ▼               └─────────────────┘
                       ┌──────────────────┐
                       │   S3 Bucket      │
                       │ (Generated Imgs) │
                       └──────────────────┘
                                ▲
                       ┌──────────────────┐
                       │ Secrets Manager  │
                       │   (API Keys)     │
                       └──────────────────┘
```

## Contributing

To add support for additional AI services:

1. Add the API integration to the appropriate Lambda function
2. Add any new required environment variables
3. Update the IAM policies if needed
4. Test thoroughly before deployment

## License

This infrastructure code is part of the MTGenesis.ai project.

