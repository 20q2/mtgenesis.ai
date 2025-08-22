# AWS Secrets Manager secret for API keys
resource "aws_secretsmanager_secret" "api_keys" {
  name                    = "${local.project_name}-api-keys-${random_string.suffix.result}"
  description             = "API keys for MTGenesis AI services"
  recovery_window_in_days = 7
  tags                    = local.common_tags
}

# Secret version with API keys
resource "aws_secretsmanager_secret_version" "api_keys" {
  secret_id = aws_secretsmanager_secret.api_keys.id
  secret_string = jsonencode({
    openai_api_key     = var.openai_api_key
    anthropic_api_key  = var.anthropic_api_key
    stability_api_key  = var.stability_api_key
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}

