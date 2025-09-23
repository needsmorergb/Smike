#!/usr/bin/env python3
"""
Smartsheet to Wrike Migration Script - FIXED VERSION
Uses the exact API call format that was proven to work in testing
"""

import requests
import json
import logging
import sys
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

# Setup logging
def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup comprehensive logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'migration_fixed_{timestamp}.log'
    
    logger = logging.getLogger('SmartsheetWrikeMigrator')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Console handler with UTF-8 encoding for Windows
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if not verbose else logging.DEBUG)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # Fix encoding for Windows console
    if sys.platform == 'win32':
        import codecs
        console_handler.stream = codecs.getwriter('utf-8')(console_handler.stream.buffer, 'replace')
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class SmartsheetWrikeMigrator:
    """Fixed migration class using proven API call format"""
    
    def __init__(self, smartsheet_token: str, wrike_token: str, logger: logging.Logger):
        self.logger = logger
        self.smartsheet_token = smartsheet_token
        self.wrike_token = wrike_token
        
        # API endpoints
        self.smartsheet_base = "https://api.smartsheet.com/2.0"
        self.wrike_base = "https://www.wrike.com/api/v4"
        
        # Stats
        self.stats = {
            'total_rows': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }
        
    def get_smartsheet_data(self, sheet_id: str) -> Optional[Dict]:
        """Fetch Smartsheet data"""
        self.logger.info(f"Fetching Smartsheet {sheet_id}")
        
        headers = {
            "Authorization": f"Bearer {self.smartsheet_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(
                f"{self.smartsheet_base}/sheets/{sheet_id}",
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            self.logger.info(f"Successfully fetched sheet: {data.get('name', 'Unknown')}")
            return data
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch Smartsheet: {e}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"Response: {e.response.text}")
            return None
    
    def get_wrike_custom_fields(self) -> Dict[str, str]:
        """Get existing Wrike custom fields"""
        self.logger.info("Fetching Wrike custom fields")
        
        headers = {
            "Authorization": f"Bearer {self.wrike_token}"
        }
        
        try:
            response = requests.get(
                f"{self.wrike_base}/customfields",
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            fields = {}
            for field in response.json().get('data', []):
                fields[field['title']] = field['id']
                self.logger.debug(f"Found CF: {field['title']} -> {field['id']}")
            
            self.logger.info(f"Found {len(fields)} custom fields in Wrike")
            return fields
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch custom fields: {e}")
            return {}
    
    def create_wrike_task_simple(self, folder_id: str, title: str, 
                                 custom_fields: Optional[List[Dict]] = None) -> Optional[str]:
        """
        Create a task using the EXACT format that worked in testing
        This uses form-encoded data with explicit Content-Type
        """
        
        url = f"{self.wrike_base}/folders/{folder_id}/tasks"
        
        # Build headers - using explicit Content-Type like in successful Test 1
        headers = {
            "Authorization": f"Bearer {self.wrike_token}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        # Build form data
        data = {
            "title": title,
            "description": f"Migrated from Smartsheet on {datetime.now().isoformat()}"
        }
        
        # Add custom fields if present
        if custom_fields:
            data["customFields"] = json.dumps(custom_fields)
            self.logger.debug(f"Adding {len(custom_fields)} custom fields")
        
        try:
            # Make the API call using the exact same format as the successful test
            response = requests.post(
                url,
                headers=headers,
                data=data,  # Form-encoded data
                timeout=30
            )
            
            # Check response
            if response.status_code == 200:
                result = response.json()
                if 'data' in result and result['data']:
                    task_id = result['data'][0]['id']
                    self.logger.debug(f"Successfully created task: {title} (ID: {task_id})")
                    return task_id
                else:
                    self.logger.error(f"Unexpected response structure: {result}")
                    return None
            else:
                self.logger.error(f"Task creation failed with status {response.status_code}")
                self.logger.error(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for task '{title}': {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error creating task '{title}': {e}")
            return None
    
    def migrate(self, sheet_id: str, wrike_folder_id: str, 
                title_column: str, dry_run: bool = False,
                max_rows: Optional[int] = None) -> bool:
        """Main migration function"""
        
        self.logger.info("="*60)
        self.logger.info("STARTING MIGRATION")
        self.logger.info("="*60)
        
        # Step 1: Fetch Smartsheet data
        sheet_data = self.get_smartsheet_data(sheet_id)
        if not sheet_data:
            self.logger.error("Failed to fetch Smartsheet data")
            return False
        
        sheet_name = sheet_data.get('name', 'Unknown')
        rows = sheet_data.get('rows', [])
        columns = sheet_data.get('columns', [])
        
        self.logger.info(f"Sheet: {sheet_name}")
        self.logger.info(f"Rows: {len(rows)}")
        self.logger.info(f"Columns: {len(columns)}")
        
        # Step 2: Find title column
        columns_by_id = {str(col['id']): col for col in columns}
        columns_by_title = {col['title']: col for col in columns}
        
        if title_column not in columns_by_title:
            self.logger.error(f"Title column '{title_column}' not found in sheet")
            self.logger.info(f"Available columns: {list(columns_by_title.keys())}")
            return False
        
        title_column_id = str(columns_by_title[title_column]['id'])
        self.logger.info(f"Using title column: {title_column} (ID: {title_column_id})")
        
        # Step 3: Get Wrike custom fields
        wrike_fields = self.get_wrike_custom_fields()
        
        # Step 4: Process rows
        if max_rows:
            rows = rows[:max_rows]
            self.logger.info(f"Processing first {max_rows} rows only")
        
        self.stats['total_rows'] = len(rows)
        
        for idx, row in enumerate(rows, 1):
            self.logger.info(f"Processing row {idx}/{len(rows)}")
            
            # Extract data from row
            task_title = None
            custom_field_values = []
            
            for cell in row.get('cells', []):
                col_id = str(cell.get('columnId'))
                
                if col_id not in columns_by_id:
                    continue
                
                col = columns_by_id[col_id]
                col_title = col['title']
                
                # Get cell value (prefer displayValue over value)
                value = cell.get('displayValue') or cell.get('value')
                if not value:
                    continue
                
                # Check if this is the title column
                if col_id == title_column_id:
                    task_title = str(value)
                    self.logger.debug(f"  Title: {task_title}")
                # Otherwise, if it's a known custom field, add it
                elif col_title in wrike_fields:
                    custom_field_values.append({
                        'id': wrike_fields[col_title],
                        'value': str(value)
                    })
                    self.logger.debug(f"  CF: {col_title} = {value}")
            
            # Skip if no title
            if not task_title:
                self.logger.warning(f"Row {idx}: No title found, skipping")
                self.stats['skipped'] += 1
                continue
            
            # Create task (or simulate in dry run)
            if dry_run:
                self.logger.info(f"  [DRY RUN] Would create: '{task_title}' with {len(custom_field_values)} CFs")
                self.stats['successful'] += 1
            else:
                self.logger.info(f"  Creating task: '{task_title}'")
                task_id = self.create_wrike_task_simple(
                    wrike_folder_id,
                    task_title,
                    custom_field_values if custom_field_values else None
                )
                
                if task_id:
                    self.logger.info(f"  ✓ Created task ID: {task_id}")
                    self.stats['successful'] += 1
                else:
                    self.logger.error(f"  ✗ Failed to create task")
                    self.stats['failed'] += 1
            
            # Rate limiting (be nice to the API)
            if not dry_run:
                time.sleep(0.5)
        
        # Summary
        self.logger.info("="*60)
        self.logger.info("MIGRATION COMPLETE")
        self.logger.info("="*60)
        self.logger.info(f"Total rows: {self.stats['total_rows']}")
        self.logger.info(f"Successful: {self.stats['successful']}")
        self.logger.info(f"Failed: {self.stats['failed']}")
        self.logger.info(f"Skipped: {self.stats['skipped']}")
        
        return self.stats['failed'] == 0

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate Smartsheet to Wrike (COMPLETE)')
    parser.add_argument('--sheet-id', required=True, help='Smartsheet sheet ID')
    parser.add_argument('--wrike-folder', required=True, help='Wrike folder ID')
    parser.add_argument('--title-column', required=True, help='Column to use as task title')
    parser.add_argument('--dry-run', action='store_true', help='Test run without creating tasks')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--max-rows', type=int, help='Maximum number of rows to process')
    
    args = parser.parse_args()
    
    # Get tokens from environment
    smartsheet_token = os.environ.get('SMARTSHEET_ACCESS_TOKEN')
    wrike_token = os.environ.get('WRIKE_ACCESS_TOKEN')
    
    if not smartsheet_token:
        print("ERROR: Please set SMARTSHEET_ACCESS_TOKEN environment variable")
        sys.exit(1)
    
    if not wrike_token:
        print("ERROR: Please set WRIKE_ACCESS_TOKEN environment variable")
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    logger.info("Starting Smartsheet to Wrike Migration")
    logger.info(f"Sheet ID: {args.sheet_id}")
    logger.info(f"Wrike Folder: {args.wrike_folder}")
    logger.info(f"Title Column: {args.title_column}")
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    
    # Create migrator
    migrator = SmartsheetWrikeMigrator(smartsheet_token, wrike_token, logger)
    
    # Run migration
    success = migrator.migrate(
        args.sheet_id,
        args.wrike_folder,
        args.title_column,
        args.dry_run,
        args.max_rows
    )
    
    if success:
        logger.info("[SUCCESS] Migration completed successfully!")
        sys.exit(0)
    else:
        logger.error("[ERROR] Migration completed with errors")
        sys.exit(1)

if __name__ == "__main__":
    main()
