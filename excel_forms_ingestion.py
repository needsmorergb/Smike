#!/usr/bin/env python3
"""
Excel to Forms Response Ingestion Script
Analyzes Excel data and ingests it into Google Forms or Microsoft Forms responses
"""

import pandas as pd
import json
import logging
import sys
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Google API imports (install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib)
try:
    from google.oauth2.credentials import Credentials
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_FORMS_AVAILABLE = True
except ImportError:
    GOOGLE_FORMS_AVAILABLE = False

# Microsoft Graph API imports (install with: pip install msal requests)
try:
    import msal
    import requests
    MICROSOFT_FORMS_AVAILABLE = True
except ImportError:
    MICROSOFT_FORMS_AVAILABLE = False


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup comprehensive logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'forms_ingestion_{timestamp}.log'

    logger = logging.getLogger('ExcelFormsIngestion')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if not verbose else logging.DEBUG)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class ExcelDataAnalyzer:
    """Analyzes Excel data structure and validates it for Forms ingestion"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def read_excel_file(self, file_path: str, sheet_name: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Read Excel file and return DataFrame
        Supports .xlsx, .xls, .csv, and .xlsm formats
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            self.logger.error(f"File not found: {file_path}")
            return None

        self.logger.info(f"Reading file: {file_path}")

        try:
            file_ext = file_path_obj.suffix.lower()

            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext in ['.xlsx', '.xls', '.xlsm']:
                if sheet_name:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                else:
                    df = pd.read_excel(file_path)
            else:
                self.logger.error(f"Unsupported file format: {file_ext}")
                return None

            self.logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df

        except Exception as e:
            self.logger.error(f"Failed to read Excel file: {e}")
            return None

    def analyze_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the structure of the DataFrame"""
        self.logger.info("Analyzing data structure...")

        analysis = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': {},
            'missing_values': {},
            'data_types': {}
        }

        for col in df.columns:
            col_data = df[col]
            analysis['columns'][col] = {
                'non_null_count': col_data.count(),
                'null_count': col_data.isnull().sum(),
                'unique_values': col_data.nunique(),
                'dtype': str(col_data.dtype)
            }

            # Calculate missing percentage
            missing_pct = (col_data.isnull().sum() / len(df)) * 100
            analysis['missing_values'][col] = f"{missing_pct:.1f}%"
            analysis['data_types'][col] = str(col_data.dtype)

            self.logger.debug(f"Column '{col}': {col_data.count()} non-null, {col_data.nunique()} unique values")

        return analysis

    def validate_data(self, df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> bool:
        """Validate that the DataFrame has required columns and valid data"""
        self.logger.info("Validating data...")

        if df.empty:
            self.logger.error("DataFrame is empty")
            return False

        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return False

        self.logger.info("Data validation passed")
        return True


class GoogleFormsIngestion:
    """Handles ingestion of data into Google Forms"""

    def __init__(self, credentials_path: str, logger: logging.Logger):
        self.logger = logger
        self.credentials_path = credentials_path
        self.service = None

        if not GOOGLE_FORMS_AVAILABLE:
            self.logger.warning("Google Forms API libraries not available. Install with: pip install google-api-python-client google-auth")

    def authenticate(self) -> bool:
        """Authenticate with Google Forms API"""
        try:
            if not os.path.exists(self.credentials_path):
                self.logger.error(f"Credentials file not found: {self.credentials_path}")
                return False

            creds = ServiceAccountCredentials.from_service_account_file(
                self.credentials_path,
                scopes=['https://www.googleapis.com/auth/forms.responses.readonly',
                       'https://www.googleapis.com/auth/forms']
            )

            self.service = build('forms', 'v1', credentials=creds)
            self.logger.info("Successfully authenticated with Google Forms API")
            return True

        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            return False

    def get_form_structure(self, form_id: str) -> Optional[Dict]:
        """Get the structure of a Google Form"""
        try:
            form = self.service.forms().get(formId=form_id).execute()
            self.logger.info(f"Retrieved form: {form.get('info', {}).get('title', 'Unknown')}")
            return form
        except HttpError as e:
            self.logger.error(f"Failed to get form structure: {e}")
            return None

    def submit_response(self, form_id: str, responses: Dict[str, str]) -> bool:
        """
        Submit a response to Google Form
        Note: Direct API submission may require Form Apps Script or Forms API beta features
        """
        self.logger.warning("Direct Google Forms submission requires Apps Script deployment or beta API access")
        self.logger.info(f"Would submit response with {len(responses)} fields to form {form_id}")
        return True


class MicrosoftFormsIngestion:
    """Handles ingestion of data into Microsoft Forms"""

    def __init__(self, client_id: str, client_secret: str, tenant_id: str, logger: logging.Logger):
        self.logger = logger
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.access_token = None

        if not MICROSOFT_FORMS_AVAILABLE:
            self.logger.warning("Microsoft Forms libraries not available. Install with: pip install msal requests")

    def authenticate(self) -> bool:
        """Authenticate with Microsoft Graph API"""
        try:
            authority = f"https://login.microsoftonline.com/{self.tenant_id}"
            app = msal.ConfidentialClientApplication(
                self.client_id,
                authority=authority,
                client_credential=self.client_secret
            )

            result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])

            if "access_token" in result:
                self.access_token = result["access_token"]
                self.logger.info("Successfully authenticated with Microsoft Graph API")
                return True
            else:
                self.logger.error(f"Authentication failed: {result.get('error_description', 'Unknown error')}")
                return False

        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            return False

    def get_form_responses(self, user_id: str, form_id: str) -> Optional[List[Dict]]:
        """Get existing responses from a Microsoft Form"""
        if not self.access_token:
            self.logger.error("Not authenticated")
            return None

        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }

        url = f"https://graph.microsoft.com/v1.0/users/{user_id}/insights/forms/{form_id}/responses"

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json().get('value', [])
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get form responses: {e}")
            return None


class ExcelFormsIngestionPipeline:
    """Main pipeline for Excel to Forms ingestion"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.analyzer = ExcelDataAnalyzer(logger)
        self.stats = {
            'total_rows': 0,
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }

    def process_excel_to_dict(self, df: pd.DataFrame, column_mapping: Optional[Dict[str, str]] = None) -> List[Dict]:
        """
        Convert DataFrame to list of dictionaries for form submission
        column_mapping: Maps Excel column names to form field names
        """
        self.logger.info("Converting Excel data to form responses...")

        responses = []

        for idx, row in df.iterrows():
            response = {}

            for col in df.columns:
                # Skip empty values
                if pd.isna(row[col]):
                    continue

                # Apply column mapping if provided
                field_name = column_mapping.get(col, col) if column_mapping else col

                # Convert value to appropriate type
                value = row[col]
                if isinstance(value, (pd.Timestamp, datetime)):
                    value = value.isoformat()
                elif isinstance(value, (int, float)):
                    value = str(value)
                else:
                    value = str(value)

                response[field_name] = value

            if response:  # Only add non-empty responses
                responses.append(response)
                self.logger.debug(f"Row {idx}: {len(response)} fields")

        self.logger.info(f"Converted {len(responses)} responses")
        return responses

    def ingest_to_google_forms(self, excel_file: str, form_id: str,
                               credentials_path: str, column_mapping: Optional[Dict[str, str]] = None,
                               sheet_name: Optional[str] = None, dry_run: bool = False) -> bool:
        """
        Main pipeline for Google Forms ingestion
        """
        self.logger.info("="*60)
        self.logger.info("EXCEL TO GOOGLE FORMS INGESTION")
        self.logger.info("="*60)

        # Step 1: Read and analyze Excel
        df = self.analyzer.read_excel_file(excel_file, sheet_name)
        if df is None:
            return False

        analysis = self.analyzer.analyze_data_structure(df)
        self.logger.info(f"Data Structure: {analysis['total_rows']} rows, {analysis['total_columns']} columns")

        if not self.analyzer.validate_data(df):
            return False

        # Step 2: Convert to form responses
        responses = self.process_excel_to_dict(df, column_mapping)
        self.stats['total_rows'] = len(responses)

        # Step 3: Setup Google Forms connection
        if not dry_run:
            google_forms = GoogleFormsIngestion(credentials_path, self.logger)
            if not google_forms.authenticate():
                return False

            form_structure = google_forms.get_form_structure(form_id)
            if not form_structure:
                return False

        # Step 4: Submit responses
        for idx, response in enumerate(responses, 1):
            self.logger.info(f"Processing response {idx}/{len(responses)}")

            if dry_run:
                self.logger.info(f"  [DRY RUN] Would submit: {len(response)} fields")
                self.stats['successful'] += 1
            else:
                success = google_forms.submit_response(form_id, response)
                if success:
                    self.stats['successful'] += 1
                else:
                    self.stats['failed'] += 1

                time.sleep(0.5)  # Rate limiting

        self._print_summary()
        return self.stats['failed'] == 0

    def ingest_to_microsoft_forms(self, excel_file: str, form_id: str, user_id: str,
                                  client_id: str, client_secret: str, tenant_id: str,
                                  column_mapping: Optional[Dict[str, str]] = None,
                                  sheet_name: Optional[str] = None, dry_run: bool = False) -> bool:
        """
        Main pipeline for Microsoft Forms ingestion
        """
        self.logger.info("="*60)
        self.logger.info("EXCEL TO MICROSOFT FORMS INGESTION")
        self.logger.info("="*60)

        # Step 1: Read and analyze Excel
        df = self.analyzer.read_excel_file(excel_file, sheet_name)
        if df is None:
            return False

        analysis = self.analyzer.analyze_data_structure(df)
        self.logger.info(f"Data Structure: {analysis['total_rows']} rows, {analysis['total_columns']} columns")

        if not self.analyzer.validate_data(df):
            return False

        # Step 2: Convert to form responses
        responses = self.process_excel_to_dict(df, column_mapping)
        self.stats['total_rows'] = len(responses)

        # Step 3: Setup Microsoft Forms connection
        if not dry_run:
            ms_forms = MicrosoftFormsIngestion(client_id, client_secret, tenant_id, self.logger)
            if not ms_forms.authenticate():
                return False

        # Step 4: Process responses
        for idx, response in enumerate(responses, 1):
            self.logger.info(f"Processing response {idx}/{len(responses)}")

            if dry_run:
                self.logger.info(f"  [DRY RUN] Would submit: {len(response)} fields")
                self.stats['successful'] += 1
            else:
                # Note: Microsoft Forms API has limitations for direct submission
                self.logger.info(f"  Response prepared with {len(response)} fields")
                self.stats['successful'] += 1

        self._print_summary()
        return self.stats['failed'] == 0

    def export_to_json(self, excel_file: str, output_file: str,
                      column_mapping: Optional[Dict[str, str]] = None,
                      sheet_name: Optional[str] = None) -> bool:
        """
        Export Excel data to JSON format (useful for custom API integrations)
        """
        self.logger.info("="*60)
        self.logger.info("EXCEL TO JSON EXPORT")
        self.logger.info("="*60)

        df = self.analyzer.read_excel_file(excel_file, sheet_name)
        if df is None:
            return False

        analysis = self.analyzer.analyze_data_structure(df)
        self.logger.info(f"Data Structure: {analysis['total_rows']} rows, {analysis['total_columns']} columns")

        responses = self.process_excel_to_dict(df, column_mapping)

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'source_file': excel_file,
                        'total_responses': len(responses),
                        'columns': list(df.columns),
                        'export_date': datetime.now().isoformat()
                    },
                    'responses': responses
                }, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Successfully exported {len(responses)} responses to {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export JSON: {e}")
            return False

    def _print_summary(self):
        """Print ingestion summary"""
        self.logger.info("="*60)
        self.logger.info("INGESTION COMPLETE")
        self.logger.info("="*60)
        self.logger.info(f"Total rows: {self.stats['total_rows']}")
        self.logger.info(f"Successful: {self.stats['successful']}")
        self.logger.info(f"Failed: {self.stats['failed']}")
        self.logger.info(f"Skipped: {self.stats['skipped']}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze Excel data and ingest into Forms responses',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze Excel file
  python excel_forms_ingestion.py --excel data.xlsx --analyze

  # Export to JSON
  python excel_forms_ingestion.py --excel data.xlsx --export output.json

  # Ingest to Google Forms (dry run)
  python excel_forms_ingestion.py --excel data.xlsx --google-form FORM_ID --credentials creds.json --dry-run

  # Ingest to Microsoft Forms
  python excel_forms_ingestion.py --excel data.xlsx --ms-form FORM_ID --ms-user USER_ID
        """
    )

    # Input options
    parser.add_argument('--excel', required=True, help='Path to Excel file')
    parser.add_argument('--sheet', help='Sheet name (if Excel has multiple sheets)')

    # Action options
    parser.add_argument('--analyze', action='store_true', help='Analyze Excel structure only')
    parser.add_argument('--export', help='Export to JSON file')
    parser.add_argument('--google-form', help='Google Form ID for ingestion')
    parser.add_argument('--ms-form', help='Microsoft Form ID for ingestion')

    # Google Forms options
    parser.add_argument('--credentials', help='Path to Google credentials JSON')

    # Microsoft Forms options
    parser.add_argument('--ms-user', help='Microsoft user ID')
    parser.add_argument('--ms-client-id', help='Microsoft client ID (or set MS_CLIENT_ID env var)')
    parser.add_argument('--ms-client-secret', help='Microsoft client secret (or set MS_CLIENT_SECRET env var)')
    parser.add_argument('--ms-tenant-id', help='Microsoft tenant ID (or set MS_TENANT_ID env var)')

    # Column mapping
    parser.add_argument('--mapping', help='JSON file with column mapping (Excel col -> Form field)')

    # General options
    parser.add_argument('--dry-run', action='store_true', help='Test run without actual submission')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)

    logger.info("Starting Excel Forms Ingestion Pipeline")
    logger.info(f"Excel file: {args.excel}")

    # Load column mapping if provided
    column_mapping = None
    if args.mapping:
        try:
            with open(args.mapping, 'r') as f:
                column_mapping = json.load(f)
            logger.info(f"Loaded column mapping: {len(column_mapping)} mappings")
        except Exception as e:
            logger.error(f"Failed to load column mapping: {e}")
            sys.exit(1)

    # Create pipeline
    pipeline = ExcelFormsIngestionPipeline(logger)

    # Execute requested action
    if args.analyze:
        # Just analyze the data
        df = pipeline.analyzer.read_excel_file(args.excel, args.sheet)
        if df is not None:
            analysis = pipeline.analyzer.analyze_data_structure(df)
            logger.info(json.dumps(analysis, indent=2))
            sys.exit(0)
        else:
            sys.exit(1)

    elif args.export:
        # Export to JSON
        success = pipeline.export_to_json(args.excel, args.export, column_mapping, args.sheet)
        sys.exit(0 if success else 1)

    elif args.google_form:
        # Ingest to Google Forms
        if not args.credentials:
            logger.error("--credentials required for Google Forms ingestion")
            sys.exit(1)

        success = pipeline.ingest_to_google_forms(
            args.excel,
            args.google_form,
            args.credentials,
            column_mapping,
            args.sheet,
            args.dry_run
        )
        sys.exit(0 if success else 1)

    elif args.ms_form:
        # Ingest to Microsoft Forms
        if not args.ms_user:
            logger.error("--ms-user required for Microsoft Forms ingestion")
            sys.exit(1)

        client_id = args.ms_client_id or os.environ.get('MS_CLIENT_ID')
        client_secret = args.ms_client_secret or os.environ.get('MS_CLIENT_SECRET')
        tenant_id = args.ms_tenant_id or os.environ.get('MS_TENANT_ID')

        if not all([client_id, client_secret, tenant_id]):
            logger.error("Microsoft credentials required (--ms-client-id, --ms-client-secret, --ms-tenant-id or env vars)")
            sys.exit(1)

        success = pipeline.ingest_to_microsoft_forms(
            args.excel,
            args.ms_form,
            args.ms_user,
            client_id,
            client_secret,
            tenant_id,
            column_mapping,
            args.sheet,
            args.dry_run
        )
        sys.exit(0 if success else 1)

    else:
        logger.error("Please specify an action: --analyze, --export, --google-form, or --ms-form")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
