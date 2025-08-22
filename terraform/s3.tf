# S3 bucket for storing generated card images
resource "aws_s3_bucket" "card_images" {
  bucket = var.s3_bucket_name != "" ? var.s3_bucket_name : "${local.project_name}-images-${random_string.suffix.result}"
  tags   = local.common_tags
}

resource "aws_s3_bucket_versioning" "card_images" {
  bucket = aws_s3_bucket.card_images.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "card_images" {
  bucket = aws_s3_bucket.card_images.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "card_images" {
  bucket = aws_s3_bucket.card_images.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

# S3 bucket policy to allow public read access to card images
resource "aws_s3_bucket_policy" "card_images" {
  bucket = aws_s3_bucket.card_images.id
  depends_on = [aws_s3_bucket_public_access_block.card_images]

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "PublicReadGetObject"
        Effect    = "Allow"
        Principal = "*"
        Action    = "s3:GetObject"
        Resource  = "${aws_s3_bucket.card_images.arn}/*"
      }
    ]
  })
}

# S3 bucket for storing Lambda deployment packages
resource "aws_s3_bucket" "lambda_deployments" {
  bucket = "${local.project_name}-lambda-${random_string.suffix.result}"
  tags   = local.common_tags
}

resource "aws_s3_bucket_versioning" "lambda_deployments" {
  bucket = aws_s3_bucket.lambda_deployments.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "lambda_deployments" {
  bucket = aws_s3_bucket.lambda_deployments.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

