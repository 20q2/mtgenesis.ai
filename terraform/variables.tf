variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "openai_api_key" {
  description = "OpenAI API key for text generation"
  type        = string
  sensitive   = true
  default     = ""
}

variable "anthropic_api_key" {
  description = "Anthropic API key for text generation"
  type        = string
  sensitive   = true
  default     = ""
}

variable "stability_api_key" {
  description = "Stability AI API key for image generation"
  type        = string
  sensitive   = true
  default     = ""
}

variable "enable_gpu_instance" {
  description = "Enable GPU instance for self-hosted AI models"
  type        = bool
  default     = false
}

variable "gpu_instance_type" {
  description = "EC2 instance type for GPU workloads"
  type        = string
  default     = "g4dn.xlarge"
}

variable "enable_bedrock" {
  description = "Enable AWS Bedrock for AI models"
  type        = bool
  default     = true
}

variable "lambda_timeout" {
  description = "Lambda function timeout in seconds"
  type        = number
  default     = 300
}

variable "lambda_memory" {
  description = "Lambda function memory in MB"
  type        = number
  default     = 1024
}

variable "s3_bucket_name" {
  description = "S3 bucket name for storing generated images (leave empty for auto-generated)"
  type        = string
  default     = ""
}

