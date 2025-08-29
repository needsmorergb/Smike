#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Smartsheet -> Wrike Migration Tool with Comprehensive Audit

Features:
- Creates/reuses Wrike Projects under destination folders
- Intelligent custom field creation (DropDown for PICKLIST types, Text for others)
- Creates tasks with custom field values, preserving Smartsheet data fidelity
- Comprehensive audit with actionable recommendations
- Streaming CSV writes for large datasets
- Enhanced error handling and logging
- Data validation and cleanup

Requirements:
  SMARTSHEET_TOKEN   (required)
  WRIKE_TOKEN        (required)

Usage:
  python smartsheet_wrike_migrator.py \
    --sheet-id 558272789931908 \
    --wrike-dest-id IEAGMPFKI5ST2COM \
    --title-column "Full Name of 2024 Mentor Champion" \
    --include-cols "Office,Full Name of 2024 Mentor Champion" \
    --limit-rows 25 \
    --dry-run
"""

from __future__ import annotations
import os
import json
import time
import argparse
import logging
import csv
import datetime
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import requests
import certifi

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# API Configuration
SS_API = "https://api.smartsheet.com/2.0"
WRIKE_API = "https://www.wrike.com/api/v4"

# Retry Configuration
MAX_RETRIES = 5
RETRY_BACKOFF_S = 2.0
MAX_CF_OPTIONS = 200
REQUEST_TIMEOUT = 60

# Logging setup
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"migration_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("smartsheet_wrike_migrator")

@dataclass
class MigrationStats:
    """Track migration statistics"""
    total_rows: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    custom_fields_created: int = 0
    custom_fields_reused: int = 0
    dropdown_fields: int = 0
    text_fields: int = 0
    
    def __str__(self) -> str:
        return (f"Migration Stats: {self.successful_tasks}/{self.total_rows} tasks created, "
                f"{self.custom_fields_created} CFs created, {self.custom_fields_reused} CFs reused")

class MigrationError(Exception):
    """Custom exception for migration-specific errors"""
    pass

# ------------------ Enhanced HTTP Handling ------------------

def _get_headers(service: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Get headers for API requests with proper token validation"""
    token_var = f"{service.upper()}_TOKEN"
    token = os.getenv(token_var)
    if not token:
        raise MigrationError(f"Environment variable {token_var} is required")
    
    headers = {"Authorization": f"Bearer {token}"}
    if extra:
        headers.update(extra)
    return headers

def _make_request(method: str, url: str, headers=None, params=None, data=None, 
                 description: str = "") -> requests.Response:
    """Enhanced HTTP request with proper retry logic and error handling"""
    last_exception = None
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.debug(f"Attempt {attempt + 1}/{MAX_RETRIES}: {method} {url}")
            if description:
                logger.debug(f"Request: {description}")
                
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=data,
                timeout=REQUEST_TIMEOUT,
                verify=certifi.where(),
                allow_redirects=True
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', RETRY_BACKOFF_S * (attempt + 1)))
                logger.warning(f"Rate limited. Waiting {retry_after}s before retry...")
                time.sleep(retry_after)
                continue
                
            # Handle server errors with retry
            if response.status_code >= 500:
                wait_time = RETRY_BACKOFF_S * (2 ** attempt)
                logger.warning(f"Server error {response.status_code}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
                
            return response
            
        except requests.RequestException as e:
            last_exception = e
            wait_time = RETRY_BACKOFF_S * (2 ** attempt)
            logger.warning(f"Request failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    if last_exception:
        raise MigrationError(f"Request failed after {MAX_RETRIES} attempts: {last_exception}")
    raise MigrationError(f"Request failed after {MAX_RETRIES} attempts with unknown error")

# ------------------ Smartsheet API ------------------

def get_smartsheet_sheet(sheet_id: str) -> dict:
    """Fetch Smartsheet data with comprehensive includes"""
    logger.info(f"Fetching Smartsheet data for sheet {sheet_id}")
    
    response = _make_request(
        "GET", 
        f"{SS_API}/sheets/{sheet_id}",
        headers=_get_headers("smartsheet"),
        params={"include": "attachments,discussions,format,objectValue"},
        description="Fetching Smartsheet sheet data"
    )
    
    if not response.ok:
        raise MigrationError(f"Failed to fetch Smartsheet {sheet_id}: {response.status_code} {response.text}")
    
    return response.json()

# ------------------ Wrike API: Folders & Projects ------------------

def get_wrike_folder(folder_id: str) -> Optional[dict]:
    """Get Wrike folder details"""
    try:
        response = _make_request(
            "GET", 
            f"{WRIKE_API}/folders/{folder_id}",
            headers=_get_headers("wrike"),
            description=f"Fetching Wrike folder {folder_id}"
        )
        
        if response.ok:
            data = response.json().get("data", [])
            return data[0] if data else None
    except Exception as e:
        logger.error(f"Error fetching Wrike folder {folder_id}: {e}")
    
    return None

def list_wrike_folders() -> List[dict]:
    """List all accessible Wrike folders"""
    response = _make_request(
        "GET",
        f"{WRIKE_API}/folders",
        headers=_get_headers("wrike"),
        description="Listing Wrike folders"
    )
    response.raise_for_status()
    return response.json().get("data", [])

def find_or_create_wrike_project(parent_id: str, title: str) -> dict:
    """Find existing project by title or create new one"""
    logger.info(f"Looking for existing project '{title}'")
    
    # Search for existing projects with matching title
    existing_projects = [
        folder for folder in list_wrike_folders() 
        if folder.get("title") == title and folder.get("project")
    ]
    
    if existing_projects:
        # Sort by creation date and use the most recent
        existing_projects.sort(key=lambda x: x.get("createdDate", ""), reverse=True)
        project = existing_projects[0]
        logger.info(f"Reusing existing project '{title}' ({project['id']})")
        return project
    
    # Create new project
    logger.info(f"Creating new project '{title}' under {parent_id}")
    payload = {
        "title": title,
        "parents": [parent_id],
        "project": {
            "ownerIds": [],
            "customStatusId": None
        }
    }
    
    response = _make_request(
        "POST",
        f"{WRIKE_API}/folders",
        headers=_get_headers("wrike", {"Content-Type": "application/json"}),
        data=json.dumps(payload),
        description=f"Creating project '{title}'"
    )
    response.raise_for_status()
    
    project = response.json()["data"][0]
    logger.info(f"Created project '{title}' ({project['id']})")
    return project

# ------------------ Wrike API: Custom Fields ------------------

def list_wrike_custom_fields() -> List[dict]:
    """Get all Wrike custom fields"""
    response = _make_request(
        "GET",
        f"{WRIKE_API}/customfields",
        headers=_get_headers("wrike"),
        description="Fetching Wrike custom fields"
    )
    response.raise_for_status()
    return response.json().get("data", [])

def find_wrike_custom_field(title: str) -> Optional[dict]:
    """Find custom field by exact title match"""
    for cf in list_wrike_custom_fields():
        if cf.get("title") == title:
            return cf
    return None

def create_wrike_dropdown_field(title: str, options: List[str]) -> Optional[dict]:
    """Create dropdown custom field with smart option handling"""
    if not options:
        logger.warning(f"No options provided for dropdown field '{title}'")
        return None
    
    # Clean and deduplicate options
    clean_options = []
    seen = set()
    
    for option in options:
        if not option:
            continue
        
        clean_option = str(option).strip()
        if not clean_option or clean_option in seen:
            continue
            
        seen.add(clean_option)
        clean_options.append(clean_option)
        
        if len(clean_options) >= MAX_CF_OPTIONS:
            logger.warning(f"Truncating dropdown options for '{title}' at {MAX_CF_OPTIONS}")
            break
    
    if not clean_options:
        logger.warning(f"No valid options found for dropdown field '{title}'")
        return None
    
    logger.info(f"Creating dropdown field '{title}' with {len(clean_options)} options")
    
    # Try different payload formats for Wrike API compatibility
    payloads = [
        {"title": title, "type": "DropDown", "settings": {"options": [{"title": opt} for opt in clean_options]}},
        {"title": title, "type": "DropDown", "settings": {"values": [{"title": opt} for opt in clean_options]}}
    ]
    
    for payload_type, payload in enumerate(payloads, 1):
        try:
            response = _make_request(
                "POST",
                f"{WRIKE_API}/customfields",
                headers=_get_headers("wrike", {"Content-Type": "application/json"}),
                data=json.dumps(payload),
                description=f"Creating dropdown field '{title}' (format {payload_type})"
            )
            
            if response.ok:
                cf_data = response.json().get("data", [])
                if cf_data:
                    logger.info(f"Successfully created dropdown field '{title}' ({cf_data[0].get('id')})")
                    return cf_data[0]
        except Exception as e:
            logger.debug(f"Dropdown creation attempt {payload_type} failed: {e}")
    
    return None

def create_wrike_text_field(title: str) -> dict:
    """Create text custom field"""
    logger.info(f"Creating text field '{title}'")
    
    payload = {"title": title, "type": "Text"}
    response = _make_request(
        "POST",
        f"{WRIKE_API}/customfields",
        headers=_get_headers("wrike", {"Content-Type": "application/json"}),
        data=json.dumps(payload),
        description=f"Creating text field '{title}'"
    )
    response.raise_for_status()
    
    cf_data = response.json()["data"][0]
    logger.info(f"Created text field '{title}' ({cf_data['id']})")
    return cf_data

def ensure_custom_field(title: str, field_type: str, options: Optional[List[str]] = None, 
                       stats: Optional[MigrationStats] = None) -> dict:
    """Ensure custom field exists, creating if necessary"""
    # Check for existing field
    existing_field = find_wrike_custom_field(title)
    if existing_field:
        logger.info(f"Reusing existing custom field '{title}' ({existing_field['id']})")
        if stats:
            stats.custom_fields_reused += 1
        return existing_field
    
    # Create new field based on type
    if field_type in ("PICKLIST", "MULTI_PICKLIST") and options:
        cf = create_wrike_dropdown_field(title, options)
        if cf:
            if stats:
                stats.custom_fields_created += 1
                stats.dropdown_fields += 1
            return cf
        else:
            logger.warning(f"Dropdown creation failed for '{title}', falling back to text field")
    
    # Fallback to text field
    cf = create_wrike_text_field(title)
    if stats:
        stats.custom_fields_created += 1
        stats.text_fields += 1
    return cf

# ------------------ Wrike API: Tasks ------------------

def create_wrike_task(title: str, parent_ids: List[str], description: Optional[str], 
                     custom_fields: List[dict]) -> dict:
    """Create Wrike task with custom fields"""
    payload = {
        "title": title,
        "parents": parent_ids
    }
    
    if description:
        payload["description"] = description
    
    if custom_fields:
        payload["customFields"] = custom_fields
    
    response = _make_request(
        "POST",
        f"{WRIKE_API}/tasks",
        headers=_get_headers("wrike", {"Content-Type": "application/json"}),
        data=json.dumps(payload),
        description=f"Creating task '{title}'"
    )
    response.raise_for_status()
    
    return response.json()["data"][0]

def get_wrike_task(task_id: str) -> Optional[dict]:
    """Fetch task details including custom fields"""
    try:
        response = _make_request(
            "GET",
            f"{WRIKE_API}/tasks/{task_id}",
            headers=_get_headers("wrike"),
            params={"fields": "[\"customFields\"]"},
            description=f"Fetching task {task_id}"
        )
        
        if response.ok:
            data = response.json().get("data", [])
            return data[0] if data else None
    except Exception as e:
        logger.error(f"Error fetching task {task_id}: {e}")
    
    return None

# ------------------ Data Processing ------------------

def extract_task_title(row: dict, title_column: Optional[str], 
                      col_id_to_title: Dict[str, str], sheet_name: str, row_index: int) -> str:
    """Extract task title from row with fallback logic"""
    # Try specified title column first
    if title_column:
        for cell in row.get("cells", []):
            col_title = col_id_to_title.get(str(cell.get("columnId")))
            if col_title == title_column:
                value = cell.get("displayValue") or cell.get("value")
                if value not in (None, ""):
                    return str(value).strip()
    
    # Fallback to primary column (first column)
    columns = row.get("cells", [])
    if columns:
        first_cell = columns[0]
        value = first_cell.get("displayValue") or first_cell.get("value")
        if value not in (None, ""):
            return str(value).strip()
    
    # Final fallback with proper encoding
    return f"{sheet_name} - Row {row_index}"

def collect_dropdown_options(sheet: dict, include_columns: Optional[Set[str]] = None) -> Dict[str, Set[str]]:
    """Collect all possible dropdown options from column definitions and data"""
    dropdown_options: Dict[str, Set[str]] = {}
    col_id_to_title = {str(c["id"]): c["title"] for c in sheet.get("columns", [])}
    
    # Collect from column definitions
    for column in sheet.get("columns", []):
        title = column["title"]
        if include_columns and title not in include_columns:
            continue
            
        if column.get("type") in ("PICKLIST", "MULTI_PICKLIST"):
            dropdown_options[title] = set()
            
            # Add predefined options
            for option in column.get("options", []):
                if isinstance(option, dict):
                    opt_value = option.get("value") or option.get("title")
                else:
                    opt_value = str(option)
                
                if opt_value:
                    clean_value = str(opt_value).strip()
                    if clean_value:
                        dropdown_options[title].add(clean_value)
    
    # Collect from actual row data
    for row in sheet.get("rows", []):
        for cell in row.get("cells", []):
            col_title = col_id_to_title.get(str(cell.get("columnId")))
            if not col_title or col_title not in dropdown_options:
                continue
                
            value = cell.get("displayValue") or cell.get("value")
            if value not in (None, ""):
                clean_value = str(value).strip()
                if clean_value:
                    dropdown_options[col_title].add(clean_value)
    
    return dropdown_options

def get_audit_recommendation(reason: str) -> str:
    """Provide actionable recommendations for audit issues"""
    recommendations = {
        "Smartsheet cell is empty": "Consider adding default values in Smartsheet or update data source",
        "No Wrike CF mapping (field not created/mapped)": "Check if column should be included in migration or if CF creation failed",
        "Value not sent (internal mapping filter)": "Review include_cols filter or check for data type incompatibilities",
        "Wrike stored empty value (CF not attached to task or rejected silently)": "Verify CF permissions and task template requirements",
        "Mismatch (Wrike normalization or truncation)": "Check Wrike field length limits or format requirements",
        "Dropdown option not found": "Add missing options to Wrike dropdown field or use text field instead",
        "API Error": "Check Wrike API permissions and rate limits"
    }
    return recommendations.get(reason, "Review migration logs for specific error details")

# ------------------ Enhanced Migration Logic ------------------

def migrate_smartsheet_to_wrike(
    sheet_id: str,
    wrike_dest_id: str,
    title_column: Optional[str] = None,
    include_cols: Optional[List[str]] = None,
    limit_rows: Optional[int] = None,
    dry_run: bool = False
) -> dict:
    """
    Enhanced migration with comprehensive audit and statistics
    """
    stats = MigrationStats()
    
    # Fetch Smartsheet data
    try:
        sheet = get_smartsheet_sheet(sheet_id)
        sheet_name = sheet.get("name") or f"Sheet_{sheet_id}"
        logger.info(f"Loaded Smartsheet '{sheet_name}' with {len(sheet.get('rows', []))} rows")
    except Exception as e:
        raise MigrationError(f"Failed to load Smartsheet: {e}")
    
    # Build column mappings
    col_id_to_title = {str(c["id"]): c["title"] for c in sheet.get("columns", [])}
    col_id_to_type = {str(c["id"]): c.get("type", "TEXT") for c in sheet.get("columns", [])}
    
    # Process include columns
    include_columns = None
    if include_cols:
        include_columns = {col.strip() for col in include_cols if col.strip()}
        logger.info(f"Including columns: {sorted(include_columns)}")
        
        # Validate that included columns exist
        available_columns = set(col_id_to_title.values())
        missing_columns = include_columns - available_columns
        if missing_columns:
            logger.warning(f"Requested columns not found in sheet: {sorted(missing_columns)}")
    
    # Collect dropdown options
    logger.info("Analyzing column types and collecting dropdown options...")
    dropdown_options = collect_dropdown_options(sheet, include_columns)
    
    # Validate Wrike destination
    parent_folder = get_wrike_folder(wrike_dest_id)
    if not parent_folder:
        raise MigrationError(f"Wrike destination folder '{wrike_dest_id}' not found")
    
    logger.info(f"Using Wrike destination: {parent_folder.get('title')} ({wrike_dest_id})")
    
    if dry_run:
        logger.info("DRY RUN: Would create the following custom fields:")
        for col in sheet.get("columns", []):
            title = col["title"]
            if include_columns and title not in include_columns:
                continue
            col_type = col.get("type", "TEXT")
            if title in dropdown_options:
                logger.info(f"  DropDown: '{title}' with {len(dropdown_options[title])} options")
            else:
                logger.info(f"  Text: '{title}' (type: {col_type})")
        return {"dry_run": True, "stats": stats}
    
    # Create Wrike project
    project = find_or_create_wrike_project(parent_folder["id"], sheet_name)
    project_id = project["id"]
    
    # Ensure custom fields exist
    logger.info("Ensuring custom fields exist in Wrike...")
    cf_title_to_id: Dict[str, str] = {}
    
    for column in sheet.get("columns", []):
        title = column["title"]
        if include_columns and title not in include_columns:
            continue
            
        col_type = column.get("type", "TEXT")
        options = sorted(dropdown_options.get(title, set())) if title in dropdown_options else None
        
        try:
            cf = ensure_custom_field(title, col_type, options, stats)
            cf_title_to_id[title] = cf["id"]
            logger.info(f"Custom field ready: '{title}' -> {cf['id']} ({cf.get('type', 'Unknown')})")
        except Exception as e:
            logger.error(f"Failed to ensure custom field '{title}': {e}")
            raise MigrationError(f"Custom field creation failed for '{title}': {e}")
    
    # Process rows
    rows = sheet.get("rows", [])
    if limit_rows:
        rows = rows[:limit_rows]
        logger.info(f"Limited to first {limit_rows} rows")
    
    stats.total_rows = len(rows)
    
    # Setup audit file with streaming writes
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    audit_filename = f"audit_{sheet_id}_{timestamp}.csv"
    audit_path = Path(audit_filename)
    
    created_task_ids = []
    
    logger.info(f"Starting migration of {len(rows)} rows...")
    
    with open(audit_path, 'w', newline='', encoding='utf-8') as audit_file:
        audit_writer = csv.DictWriter(audit_file, fieldnames=[
            "row_index", "task_id", "task_title", "column_title",
            "smartsheet_value", "value_sent_to_wrike", "wrike_stored_value", 
            "reason", "recommendation"
        ])
        audit_writer.writeheader()
        
        for row_idx, row in enumerate(rows, 1):
            try:
                # Extract task title
                task_title = extract_task_title(row, title_column, col_id_to_title, sheet_name, row_idx)
                
                # Build custom field payload and audit data
                cf_payload = []
                audit_entries = []
                
                for cell in row.get("cells", []):
                    col_id = str(cell.get("columnId"))
                    col_title = col_id_to_title.get(col_id)
                    
                    if not col_title:
                        continue
                    if include_columns and col_title not in include_columns:
                        continue
                    
                    # Extract cell value (prefer displayValue)
                    ss_value = cell.get("displayValue")
                    if ss_value in (None, ""):
                        ss_value = cell.get("value")
                    
                    ss_value_str = "" if ss_value in (None, "") else str(ss_value).strip()
                    
                    # Prepare audit entry
                    audit_entry = {
                        "row_index": str(row_idx),
                        "task_title": task_title,
                        "column_title": col_title,
                        "smartsheet_value": ss_value_str,
                        "value_sent_to_wrike": "",
                        "wrike_stored_value": "",
                        "reason": "",
                        "recommendation": ""
                    }
                    
                    # Determine what to send to Wrike
                    wrike_cf_id = cf_title_to_id.get(col_title)
                    if not wrike_cf_id:
                        audit_entry["reason"] = "No Wrike CF mapping (field not created/mapped)"
                    elif ss_value_str == "":
                        audit_entry["reason"] = "Smartsheet cell is empty"
                    else:
                        audit_entry["value_sent_to_wrike"] = ss_value_str
                        cf_payload.append({"id": wrike_cf_id, "value": ss_value_str})
                    
                    audit_entries.append(audit_entry)
                
                # Create task
                logger.info(f"Creating task {row_idx}/{len(rows)}: '{task_title}' with {len(cf_payload)} custom fields")
                
                task = create_wrike_task(
                    title=task_title,
                    parent_ids=[project_id],
                    description=None,
                    custom_fields=cf_payload
                )
                
                task_id = task["id"]
                created_task_ids.append(task_id)
                stats.successful_tasks += 1
                
                # Verify what was actually stored
                time.sleep(0.5)  # Brief delay to ensure data consistency
                stored_task = get_wrike_task(task_id)
                stored_cf_values = {}
                
                if stored_task and stored_task.get("customFields"):
                    stored_cf_values = {
                        cf["id"]: cf.get("value", "") 
                        for cf in stored_task["customFields"]
                    }
                
                # Complete audit entries and write to CSV
                for audit_entry in audit_entries:
                    audit_entry["task_id"] = task_id
                    col_title = audit_entry["column_title"]
                    wrike_cf_id = cf_title_to_id.get(col_title)
                    
                    if wrike_cf_id:
                        stored_value = stored_cf_values.get(wrike_cf_id, "")
                        audit_entry["wrike_stored_value"] = str(stored_value) if stored_value is not None else ""
                        
                        # Determine final reason if not already set
                        if not audit_entry["reason"]:
                            sent_value = audit_entry["value_sent_to_wrike"]
                            if stored_value == "":
                                audit_entry["reason"] = "Wrike stored empty value (CF not attached to task or rejected silently)"
                            elif str(stored_value) != str(sent_value):
                                audit_entry["reason"] = "Mismatch (Wrike normalization or truncation)"
                            else:
                                audit_entry["reason"] = "OK"
                    
                    # Add recommendation
                    audit_entry["recommendation"] = get_audit_recommendation(audit_entry["reason"])
                    
                    # Write audit row immediately (streaming)
                    audit_writer.writerow(audit_entry)
                
            except Exception as e:
                logger.error(f"Failed to create task for row {row_idx}: {e}")
                stats.failed_tasks += 1
                
                # Write error audit entries
                error_audit = {
                    "row_index": str(row_idx),
                    "task_id": "ERROR",
                    "task_title": f"Failed: {extract_task_title(row, title_column, col_id_to_title, sheet_name, row_idx)}",
                    "column_title": "N/A",
                    "smartsheet_value": "",
                    "value_sent_to_wrike": "",
                    "wrike_stored_value": "",
                    "reason": f"Task creation failed: {str(e)}",
                    "recommendation": "Check logs for detailed error information"
                }
                audit_writer.writerow(error_audit)
    
    logger.info(f"Migration complete! {stats}")
    logger.info(f"Audit report written to: {audit_path}")
    
    # Generate summary report
    summary = generate_migration_summary(audit_path, stats)
    
    return {
        "projectId": project_id,
        "createdTaskIds": created_task_ids,
        "auditCsv": str(audit_path),
        "stats": stats,
        "summary": summary
    }

def generate_migration_summary(audit_path: Path, stats: MigrationStats) -> Dict[str, any]:
    """Generate summary statistics from audit file"""
    summary = {
        "total_audit_entries": 0,
        "successful_mappings": 0,
        "issues_by_reason": {},
        "columns_with_issues": set(),
        "success_rate": 0.0
    }
    
    try:
        with open(audit_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                summary["total_audit_entries"] += 1
                reason = row["reason"]
                
                if reason == "OK":
                    summary["successful_mappings"] += 1
                else:
                    summary["issues_by_reason"][reason] = summary["issues_by_reason"].get(reason, 0) + 1
                    summary["columns_with_issues"].add(row["column_title"])
        
        if summary["total_audit_entries"] > 0:
            summary["success_rate"] = summary["successful_mappings"]