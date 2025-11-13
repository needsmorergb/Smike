# Excel to Forms Response Ingestion

This tool analyzes Excel data and ingests it into Google Forms or Microsoft Forms responses.

## Features

- **Excel Data Analysis**: Reads and analyzes Excel files (.xlsx, .xls, .csv, .xlsm)
- **Multiple Platforms**: Supports Google Forms and Microsoft Forms
- **Column Mapping**: Map Excel columns to form fields using JSON configuration
- **Data Validation**: Validates data structure and required columns
- **Comprehensive Logging**: Detailed logs with timestamps and error tracking
- **Dry Run Mode**: Test ingestion without actually submitting data
- **JSON Export**: Export Excel data to JSON format for custom integrations
- **Statistics Tracking**: Track successful, failed, and skipped records

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For Google Forms only
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

# For Microsoft Forms only
pip install msal requests

# For Excel processing (required)
pip install pandas openpyxl xlrd
```

## Usage

### 1. Analyze Excel File

Analyze the structure of your Excel file:

```bash
python excel_forms_ingestion.py --excel data.xlsx --analyze --verbose
```

This will show:
- Total rows and columns
- Column names and data types
- Missing value percentages
- Unique value counts

### 2. Export to JSON

Export Excel data to JSON format:

```bash
python excel_forms_ingestion.py --excel data.xlsx --export output.json
```

With column mapping:

```bash
python excel_forms_ingestion.py --excel data.xlsx --export output.json --mapping column_mapping_example.json
```

### 3. Ingest to Google Forms

**Dry run (recommended first):**

```bash
python excel_forms_ingestion.py \
  --excel data.xlsx \
  --google-form YOUR_FORM_ID \
  --credentials google_credentials.json \
  --dry-run \
  --verbose
```

**Live ingestion:**

```bash
python excel_forms_ingestion.py \
  --excel data.xlsx \
  --google-form YOUR_FORM_ID \
  --credentials google_credentials.json \
  --mapping column_mapping_example.json
```

### 4. Ingest to Microsoft Forms

**Setup environment variables:**

```bash
export MS_CLIENT_ID="your-client-id"
export MS_CLIENT_SECRET="your-client-secret"
export MS_TENANT_ID="your-tenant-id"
```

**Dry run:**

```bash
python excel_forms_ingestion.py \
  --excel data.xlsx \
  --ms-form YOUR_FORM_ID \
  --ms-user YOUR_USER_ID \
  --dry-run
```

**Live ingestion:**

```bash
python excel_forms_ingestion.py \
  --excel data.xlsx \
  --ms-form YOUR_FORM_ID \
  --ms-user YOUR_USER_ID \
  --mapping column_mapping_example.json
```

## Column Mapping

Create a JSON file to map Excel columns to form fields:

```json
{
  "Excel Column Name": "Form Field Name",
  "Name": "full_name",
  "Email": "email_address",
  "Phone": "phone_number",
  "Company": "company_name",
  "Message": "feedback_message"
}
```

Save this as `column_mapping.json` and use with `--mapping column_mapping.json`.

## Google Forms Setup

### 1. Enable Google Forms API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Google Forms API
4. Create credentials (Service Account)
5. Download the JSON credentials file

### 2. Share Form

Share your Google Form with the service account email address (found in credentials JSON).

## Microsoft Forms Setup

### 1. Register App in Azure AD

1. Go to [Azure Portal](https://portal.azure.com/)
2. Navigate to "Azure Active Directory" > "App registrations"
3. Click "New registration"
4. Note the Application (client) ID and Directory (tenant) ID
5. Create a client secret under "Certificates & secrets"

### 2. Grant Permissions

Grant the following Microsoft Graph API permissions:
- `Forms.Read.All`
- `Forms.ReadWrite.All`

## Excel File Format

Your Excel file should have:
- **Header row**: Column names in the first row
- **Data rows**: Each row represents one form response
- **Clean data**: Remove extra headers, footers, or formatting

Example structure:

| Name | Email | Phone | Company | Message |
|------|-------|-------|---------|---------|
| John Doe | john@example.com | 555-1234 | Acme Inc | Great service! |
| Jane Smith | jane@example.com | 555-5678 | Tech Co | Need help with... |

## Command-Line Options

```
--excel EXCEL           Path to Excel file (required)
--sheet SHEET           Sheet name (if Excel has multiple sheets)
--analyze              Analyze Excel structure only
--export FILE          Export to JSON file
--google-form FORM_ID  Google Form ID for ingestion
--ms-form FORM_ID      Microsoft Form ID for ingestion
--credentials FILE     Path to Google credentials JSON
--ms-user USER_ID      Microsoft user ID
--ms-client-id ID      Microsoft client ID
--ms-client-secret SEC Microsoft client secret
--ms-tenant-id ID      Microsoft tenant ID
--mapping FILE         JSON file with column mapping
--dry-run             Test run without actual submission
--verbose             Verbose logging
```

## Logging

The tool creates detailed log files:
- Filename: `forms_ingestion_YYYYMMDD_HHMMSS.log`
- Contains timestamps, function names, and detailed messages
- Use `--verbose` for debug-level logging

## Error Handling

The tool handles various error scenarios:
- Missing or invalid Excel files
- Authentication failures
- Network errors
- Invalid data formats
- Missing required columns

All errors are logged with detailed information for troubleshooting.

## Statistics

After completion, you'll see statistics:
- Total rows processed
- Successful submissions
- Failed submissions
- Skipped rows

## Limitations

### Google Forms
- Direct API submission requires Apps Script deployment or beta API access
- Current implementation demonstrates the structure; you may need to use Apps Script for actual submission

### Microsoft Forms
- Microsoft Forms API has limitations for direct response submission
- Consider using Power Automate or Microsoft Graph API beta features

## Examples

### Example 1: Analyze Survey Data

```bash
python excel_forms_ingestion.py --excel survey_responses.xlsx --analyze
```

### Example 2: Export Registration Data

```bash
python excel_forms_ingestion.py \
  --excel registrations.xlsx \
  --export registrations.json \
  --mapping reg_mapping.json
```

### Example 3: Test Google Forms Integration

```bash
python excel_forms_ingestion.py \
  --excel customer_feedback.xlsx \
  --google-form 1a2b3c4d5e6f7g8h9i0j \
  --credentials service_account.json \
  --dry-run \
  --verbose
```

## Troubleshooting

### "File not found" error
- Check the file path is correct
- Use absolute paths if needed

### Authentication errors
- Verify credentials file exists
- Check credentials have proper permissions
- Ensure form is shared with service account

### Column mapping errors
- Verify JSON syntax is valid
- Check column names match Excel exactly
- Use `--analyze` to see available columns

### Import errors
- Install all required dependencies
- Use virtual environment for isolation

## Support

For issues or questions:
1. Check the log file for detailed error messages
2. Use `--verbose` flag for more information
3. Verify credentials and permissions
4. Test with `--dry-run` first

## License

This tool is part of the Smike project.
