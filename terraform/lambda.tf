# Lambda function for card text generation
resource "aws_lambda_function" "text_generator" {
  function_name    = "${local.project_name}-text-generator-${random_string.suffix.result}"
  role            = aws_iam_role.lambda_role.arn
  handler         = "index.handler"
  runtime         = "python3.11"
  timeout         = var.lambda_timeout
  memory_size     = var.lambda_memory

  # Placeholder code - will be updated with actual implementation
  filename         = "../lambda/text-generator.zip"
  source_code_hash = data.archive_file.text_generator_zip.output_base64sha256

  environment {
    variables = {
      SECRETS_ARN    = aws_secretsmanager_secret.api_keys.arn
      ENVIRONMENT    = var.environment
      USE_BEDROCK    = var.enable_bedrock
      AWS_REGION     = var.aws_region
    }
  }

  tags = local.common_tags
}

# Lambda function for card image generation
resource "aws_lambda_function" "image_generator" {
  function_name    = "${local.project_name}-image-generator-${random_string.suffix.result}"
  role            = aws_iam_role.lambda_role.arn
  handler         = "index.handler"
  runtime         = "python3.11"
  timeout         = var.lambda_timeout
  memory_size     = var.lambda_memory

  # Placeholder code - will be updated with actual implementation
  filename         = "../lambda/image-generator.zip"
  source_code_hash = data.archive_file.image_generator_zip.output_base64sha256

  environment {
    variables = {
      SECRETS_ARN       = aws_secretsmanager_secret.api_keys.arn
      S3_BUCKET         = aws_s3_bucket.card_images.bucket
      ENVIRONMENT       = var.environment
      USE_BEDROCK       = var.enable_bedrock
      AWS_REGION        = var.aws_region
    }
  }

  tags = local.common_tags
}

# Data source for creating Lambda deployment packages
data "archive_file" "text_generator_zip" {
  type        = "zip"
  output_path = "../lambda/text-generator.zip"
  source {
    content = templatefile("${path.module}/lambda-templates/text-generator.py", {
      # Template variables can be added here
    })
    filename = "index.py"
  }
  source {
    content  = file("${path.module}/lambda-templates/requirements.txt")
    filename = "requirements.txt"
  }
}

data "archive_file" "image_generator_zip" {
  type        = "zip"
  output_path = "../lambda/image-generator.zip"
  source {
    content = templatefile("${path.module}/lambda-templates/image-generator.py", {
      # Template variables can be added here
    })
    filename = "index.py"
  }
  source {
    content  = file("${path.module}/lambda-templates/requirements.txt")
    filename = "requirements.txt"
  }
}

# Lambda function URLs for HTTP access
resource "aws_lambda_function_url" "text_generator" {
  function_name      = aws_lambda_function.text_generator.function_name
  authorization_type = "NONE"

  cors {
    allow_credentials = false
    allow_origins     = ["*"]
    allow_methods     = ["POST"]
    allow_headers     = ["date", "keep-alive", "content-type"]
    expose_headers    = ["date", "keep-alive"]
    max_age          = 86400
  }
}

resource "aws_lambda_function_url" "image_generator" {
  function_name      = aws_lambda_function.image_generator.function_name
  authorization_type = "NONE"

  cors {
    allow_credentials = false
    allow_origins     = ["*"]
    allow_methods     = ["POST"]
    allow_headers     = ["date", "keep-alive", "content-type"]
    expose_headers    = ["date", "keep-alive"]
    max_age          = 86400
  }
}

