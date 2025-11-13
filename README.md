# Smartsheet to Wrike Migration Tool

A Python script to migrate data from Smartsheet to Wrike using direct REST APIs.

**No Azure or Graph API dependencies** - This tool uses only direct REST API calls with simple credential management.

## Features

- Direct REST API integration (no Graph API)
- Multiple credential management options (environment variables, .env file, config.json, interactive prompts)
- Custom field mapping support
- Dry-run mode for testing
- Comprehensive logging
- Rate limiting to be nice to APIs

## Prerequisites

- Python 3.9 or higher
- `requests` library
- Smartsheet API token
- Wrike API token

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Smike
```

2. Install dependencies:
```bash
pip install requests
```

## Getting API Tokens

### Smartsheet Token
1. Log in to Smartsheet
2. Go to Account → Apps & Integrations → API Access
3. Generate a new access token
4. Copy the token (you won't be able to see it again!)

### Wrike Token
1. Log in to Wrike
2. Go to Settings → Apps & Integrations → API
3. Create a new permanent token
4. Copy the token

## Configuration

This tool supports **four methods** for providing credentials (in priority order):

### Method 1: Environment Variables (Recommended for CI/CD)

```bash
export SMARTSHEET_ACCESS_TOKEN="your_smartsheet_token"
export WRIKE_ACCESS_TOKEN="your_wrike_token"
```

### Method 2: .env File (Recommended for Local Development)

1. Copy the example file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your tokens:
```
SMARTSHEET_ACCESS_TOKEN=your_smartsheet_token
WRIKE_ACCESS_TOKEN=your_wrike_token
```

**Important:** `.env` is in `.gitignore` and will NOT be committed to version control.

### Method 3: config.json File

1. Copy the example file:
```bash
cp config.json.example config.json
```

2. Edit `config.json` and add your tokens:
```json
{
  "smartsheet_access_token": "your_smartsheet_token",
  "wrike_access_token": "your_wrike_token"
}
```

**Important:** `config.json` is in `.gitignore` and will NOT be committed to version control.

### Method 4: Interactive Prompt

If no credentials are found in the above methods, the script will prompt you to enter them:

```bash
python smartsheet_wrike_migrator.py --sheet-id <id> --wrike-folder <id> --title-column <name>
# You'll be prompted to enter tokens
```

To disable interactive prompts (useful in automated environments), use `--no-interactive`.

## Usage

### Basic Usage

```bash
python smartsheet_wrike_migrator.py \
  --sheet-id <smartsheet_id> \
  --wrike-folder <wrike_folder_id> \
  --title-column "Task Name"
```

### Dry Run (Test Without Creating Tasks)

```bash
python smartsheet_wrike_migrator.py \
  --sheet-id <smartsheet_id> \
  --wrike-folder <wrike_folder_id> \
  --title-column "Task Name" \
  --dry-run
```

### Limit Number of Rows

```bash
python smartsheet_wrike_migrator.py \
  --sheet-id <smartsheet_id> \
  --wrike-folder <wrike_folder_id> \
  --title-column "Task Name" \
  --max-rows 10
```

### Verbose Logging

```bash
python smartsheet_wrike_migrator.py \
  --sheet-id <smartsheet_id> \
  --wrike-folder <wrike_folder_id> \
  --title-column "Task Name" \
  --verbose
```

### All Options

```
Options:
  --sheet-id SHEET_ID       Smartsheet sheet ID (required)
  --wrike-folder FOLDER_ID  Wrike folder ID (required)
  --title-column COLUMN     Column name to use as task title (required)
  --dry-run                 Test run without creating tasks
  --verbose                 Enable verbose logging
  --max-rows N              Limit processing to first N rows
  --no-interactive          Disable interactive credential prompts
  -h, --help                Show help message
```

## How It Works

1. **Fetch Smartsheet Data**: Retrieves sheet data via Smartsheet REST API
2. **Get Wrike Custom Fields**: Fetches existing custom fields from Wrike
3. **Map Columns**: Maps Smartsheet columns to Wrike custom fields by name
4. **Create Tasks**: Creates tasks in Wrike with mapped custom field values
5. **Log Results**: Comprehensive logging of all operations

## Custom Field Mapping

The script automatically maps Smartsheet columns to Wrike custom fields by matching column names:

- Smartsheet column "Status" → Wrike custom field "Status" (if it exists)
- Smartsheet column "Priority" → Wrike custom field "Priority" (if it exists)

Custom fields must already exist in Wrike before migration. The script will not create new custom fields.

## Security Best Practices

1. **Never commit credentials** to version control
2. Use `.env` or `config.json` for local development (both are gitignored)
3. Use environment variables for production/CI/CD
4. Rotate your API tokens regularly
5. Use the minimum required permissions for API tokens

## Logging

The script creates detailed log files named `migration_fixed_YYYYMMDD_HHMMSS.log` with:
- All API requests and responses
- Task creation results
- Errors and warnings
- Migration statistics

## Troubleshooting

### "No credentials found" Error

Make sure you've set credentials using one of the four methods above. Check:
- Environment variables are exported in the current shell
- `.env` file exists and has correct format
- `config.json` file exists and has valid JSON
- Interactive mode is enabled (don't use `--no-interactive`)

### "Failed to fetch Smartsheet" Error

- Verify your Smartsheet token is valid
- Check that the sheet ID is correct
- Ensure you have permission to access the sheet

### "Failed to fetch custom fields" Error

- Verify your Wrike token is valid
- Check that you have permission to access the Wrike workspace

### "Task creation failed" Error

- Verify the Wrike folder ID is correct
- Check that you have permission to create tasks in that folder
- Review the log file for detailed error messages

## Architecture

This tool is designed to be **completely independent** of Azure and Microsoft Graph API:

- **Direct REST API calls** to Smartsheet and Wrike
- **No Azure Key Vault** dependency
- **No Microsoft Graph API** usage
- **Simple authentication** using Bearer tokens
- **Standard Python libraries** only (no Azure SDKs)

## API Endpoints Used

- Smartsheet: `https://api.smartsheet.com/2.0`
- Wrike: `https://www.wrike.com/api/v4`

## License

[Your License Here]

## Contributing

Contributions are welcome! Please submit issues or pull requests.
