# Lambda function URLs for API endpoints
output "text_generator_url" {
  description = "URL for the text generation Lambda function"
  value       = aws_lambda_function_url.text_generator.function_url
}

output "image_generator_url" {
  description = "URL for the image generation Lambda function"
  value       = aws_lambda_function_url.image_generator.function_url
}

# S3 bucket information
output "card_images_bucket" {
  description = "S3 bucket name for card images"
  value       = aws_s3_bucket.card_images.bucket
}

output "card_images_bucket_url" {
  description = "S3 bucket URL for card images"
  value       = "https://${aws_s3_bucket.card_images.bucket}.s3.amazonaws.com"
}

# Secrets Manager ARN
output "secrets_arn" {
  description = "ARN of the Secrets Manager secret containing API keys"
  value       = aws_secretsmanager_secret.api_keys.arn
  sensitive   = true
}

# Lambda function ARNs
output "text_generator_arn" {
  description = "ARN of the text generation Lambda function"
  value       = aws_lambda_function.text_generator.arn
}

output "image_generator_arn" {
  description = "ARN of the image generation Lambda function"
  value       = aws_lambda_function.image_generator.arn
}

# Example usage information
output "usage_instructions" {
  description = "Instructions for using the deployed AI services"
  value = <<EOF
Your MTGenesis AI infrastructure is now deployed!

API Endpoints:
- Text Generation: ${aws_lambda_function_url.text_generator.function_url}
- Image Generation: ${aws_lambda_function_url.image_generator.function_url}

Next steps:
1. Update your API keys in AWS Secrets Manager: ${aws_secretsmanager_secret.api_keys.name}
2. Test the endpoints with a POST request containing card data
3. Generated images will be stored in: ${aws_s3_bucket.card_images.bucket}

Example request body:
{
  "name": "Lightning Bolt",
  "type": "Instant",
  "colors": ["R"],
  "manaCost": "{R}",
  "rarity": "common",
  "artPrompt": "A lightning bolt striking through dark clouds"
}
EOF
}

