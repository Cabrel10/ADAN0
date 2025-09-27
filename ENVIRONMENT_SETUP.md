# Environment Variables Setup Guide

## Overview

This guide explains how to securely configure API keys using environment variables instead of hardcoded files.

## Required Environment Variables

### Binance
```bash
export BINANCE_API_KEY="your_binance_api_key"
export BINANCE_API_SECRET="your_binance_api_secret"
export BINANCE_SANDBOX="true"  # Set to "false" for production
```

### Binance Futures
```bash
export BINANCE_FUTURES_API_KEY="your_binance_futures_api_key"
export BINANCE_FUTURES_API_SECRET="your_binance_futures_api_secret"
export BINANCE_FUTURES_SANDBOX="true"  # Set to "false" for production
```

### Bitget
```bash
export BITGET_API_KEY="your_bitget_api_key"
export BITGET_API_SECRET="your_bitget_api_secret"
export BITGET_PASSPHRASE="your_bitget_passphrase"
export BITGET_SANDBOX="true"  # Set to "false" for production
```

### Bybit
```bash
export BYBIT_API_KEY="your_bybit_api_key"
export BYBIT_API_SECRET="your_bybit_api_secret"
export BYBIT_SANDBOX="true"  # Set to "false" for production
```

### OKEx
```bash
export OKEX_API_KEY="your_okex_api_key"
export OKEX_API_SECRET="your_okex_api_secret"
export OKEX_PASSPHRASE="your_okex_passphrase"
export OKEX_SANDBOX="true"  # Set to "false" for production
```

### Kraken
```bash
export KRAKEN_API_KEY="your_kraken_api_key"
export KRAKEN_API_SECRET="your_kraken_api_secret"
export KRAKEN_SANDBOX="true"  # Set to "false" for production
```

## Setup Methods

### Method 1: .env File (Recommended for Development)

1. Create a `.env` file in your project root:
```bash
# .env file
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
BINANCE_SANDBOX=true
```

2. Add `.env` to your `.gitignore` file to prevent committing secrets:
```bash
echo ".env" >> .gitignore
```

3. Load the .env file in your application:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Method 2: System Environment Variables

#### Linux/macOS
Add to your `~/.bashrc` or `~/.zshrc`:
```bash
export BINANCE_API_KEY="your_binance_api_key"
export BINANCE_API_SECRET="your_binance_api_secret"
```

Then reload your shell:
```bash
source ~/.bashrc  # or ~/.zshrc
```

#### Windows
```cmd
setx BINANCE_API_KEY "your_binance_api_key"
setx BINANCE_API_SECRET "your_binance_api_secret"
```

### Method 3: Docker Environment Variables

```dockerfile
ENV BINANCE_API_KEY=your_binance_api_key
ENV BINANCE_API_SECRET=your_binance_api_secret
```

Or using docker-compose:
```yaml
environment:
  - BINANCE_API_KEY=your_binance_api_key
  - BINANCE_API_SECRET=your_binance_api_secret
```

## Security Best Practices

1. **Never commit API keys to version control**
2. **Use different keys for development and production**
3. **Regularly rotate your API keys**
4. **Set appropriate permissions on your exchange accounts**
5. **Monitor API key usage and set up alerts**
6. **Use sandbox/testnet environments for development**

## Verification

To verify your environment variables are set correctly:

```bash
# Check if variables are set (will show masked values)
echo $BINANCE_API_KEY | sed 's/./*/g'
echo $BINANCE_API_SECRET | sed 's/./*/g'
```

## Migration from Hardcoded Files

If you have existing hardcoded API keys:

1. Copy your API keys to environment variables using one of the methods above
2. Remove any files containing hardcoded keys (e.g., `gemini_api_keys.txt`)
3. Restart your application
4. Verify the application loads credentials from environment variables

## Troubleshooting

### Common Issues

1. **Environment variables not loaded**: Make sure to restart your terminal/application after setting variables
2. **Permission denied**: Check that your API keys have the correct permissions on the exchange
3. **Sandbox mode**: Ensure `SANDBOX` variables are set to "true" for testing

### Validation

The SecureAPIManager will automatically validate that no hardcoded API keys exist in your codebase at startup.