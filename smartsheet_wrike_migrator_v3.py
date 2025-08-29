import logging
import configparser
import smartsheet
from wrike import Wrike
import re
from typing import Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    filename='migration.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_file: str = 'config.ini') -> configparser.ConfigParser:
    """Load and validate configuration from config.ini."""
    config = configparser.ConfigParser()
    config.read(config_file)
    required_sections = {
        'Smartsheet': ['access_token', 'sheet_id'],
        'Wrike': ['access_token', 'folder_id'],
        'Settings': ['reserved_columns']
    }
    for section, keys in required_sections.items():
        if section not in config:
            logger.error(f"Missing section '{section}' in config.ini")
            raise ValueError(f"Missing section '{section}' in config.ini")
        for key in keys:
            if key not in config[section] or not config[section][key]:
                logger.error(f"Missing or empty key '{key}' in section '{section}'")
                raise ValueError(f"Missing or empty key '{key}' in section '{section}'")
    return config

def get_column_id(sheet, column_name: str, column_map: Dict[str, int]) -> Optional[int]:
    """Get column ID by name from cached column map."""
    if column_name not in column_map:
        logger.error(f"Column '{column_name}' not found in sheet")
        raise ValueError(f"Required column '{column_name}' missing")
    return column_map[column_name]

def get_email(cell_value: str) -> List[str]:
    """Extract valid email addresses from a cell value."""
    if not cell_value:
        return []
    try:
        emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', cell_value)
        return emails if emails else []
    except Exception as e:
        logger.warning(f"Invalid email format in cell value '{cell_value}': {e}")
        return []

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_wrike_custom_fields(wrike) -> Dict[str, str]:
    """Retrieve existing Wrike custom fields and return a title-to-ID mapping."""
    try:
        response = wrike.customfields.get()
        custom_fields = response.get('data', [])
        return {field['title']: field['id'] for field in custom_fields}
    except Exception as e:
        logger.error(f"Failed to retrieve Wrike custom fields: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def create_wrike_custom_field(wrike, title: str) -> str:
    """Create a single Wrike custom field with retry logic."""
    try:
        response = wrike.customfields.create(title=title, type="Text")
        field_id = response['data'][0]['id']
        logger.info(f"Created Wrike custom field '{title}' with ID {field_id}")
        return field_id
    except Exception as e:
        logger.error(f"Failed to create Wrike custom field '{title}': {e}")
        raise

def create_wrike_custom_fields(wrike, sheet, reserved_columns: List[str]) -> Dict[str, str]:
    """Create Wrike custom fields for Smartsheet columns if they don't exist."""
    existing_fields = get_wrike_custom_fields(wrike)
    custom_field_map = {}
    
    for column in sheet.columns:
        if column.title not in reserved_columns:
            if column.title in existing_fields:
                logger.info(f"Custom field '{column.title}' already exists in Wrike with ID {existing_fields[column.title]}")
                custom_field_map[column.title] = existing_fields[column.title]
            else:
                field_id = create_wrike_custom_field(wrike, column.title)
                custom_field_map[column.title] = field_id
    return custom_field_map

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def create_wrike_project(wrike, folder_id: str, title: str) -> Dict:
    """Create a Wrike project with retry logic."""
    try:
        project = wrike.folders.create(folder_id, title=title, project=True)['data'][0]
        logger.info(f"Created Wrike project: {project['id']}")
        return project
    except Exception as e:
        logger.error(f"Failed to create Wrike project: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def create_wrike_task(wrike, project_id: str, task_data: Dict) -> None:
    """Create a single Wrike task with retry logic."""
    try:
        wrike.tasks.create(project_id, **task_data)
        logger.info(f"Created task: {task_data['title']}")
    except Exception as e:
        logger.error(f"Failed to create task '{task_data['title']}': {e}")
        raise

def migrate_data(smartsheet_client, wrike, config):
    """Migrate data from Smartsheet to Wrike with pagination and batch task creation."""
    page_size = 100  # Number of rows per page
    page = 1
    reserved_columns = [col.strip() for col in config['Settings']['reserved_columns'].split(',')]
    
    # Create Wrike project
    try:
        project = create_wrike_project(
            wrike,
            config['Wrike']['folder_id'],
            f"{config['Smartsheet']['sheet_id']} (Migrated from Smartsheet)"
        )
    except Exception:
        return

    # Process Smartsheet sheets with pagination
    while True:
        try:
            sheet = smartsheet_client.Sheets.get_sheet(
                config['Smartsheet']['sheet_id'],
                page_size=page_size,
                page=page
            )
            logger.info(f"Retrieved Smartsheet page {page} with {len(sheet.rows)} rows")
        except smartsheet.exceptions.RateLimitExceededError as e:
            logger.error(f"Smartsheet API rate limit exceeded: {e}")
            return
        except smartsheet.exceptions.HttpError as e:
            logger.error(f"Smartsheet API error: {e}")
            return

        if not sheet.rows:
            logger.info("No more rows to process")
            break

        # Cache column IDs
        column_map = {col.title: col.id for col in sheet.columns}
        
        # Validate required columns
        for col in reserved_columns:
            if col not in column_map:
                logger.error(f"Required column '{col}' missing in sheet")
                return

        # Create or map custom fields (once per migration)
        if page == 1:
            custom_field_map = create_wrike_custom_fields(wrike, sheet, reserved_columns)

        # Prepare tasks in batches
        batch_size = 50
        task_batch = []
        for row in sheet.rows:
            try:
                task_name = row.cells[column_map["Task Name"]].value
                if not task_name:
                    logger.warning(f"Skipping row {row.row_number}: Empty task name")
                    continue

                # Prepare custom field data
                custom_fields = []
                for col_title, col_id in column_map.items():
                    if col_title not in reserved_columns:
                        cell_value = row.cells[col_id].value
                        if cell_value:
                            custom_fields.append({
                                "id": custom_field_map[col_title],
                                "value": str(cell_value)
                            })

                task_data = {
                    "title": task_name,
                    "status": "Completed" if row.cells[column_map["Status"]].value == "Completed" else "Active",
                    "responsibles": get_email(row.cells[column_map["Assigned To"]].value),
                    "customFields": custom_fields
                }
                task_batch.append(task_data)

                # Process batch when full
                if len(task_batch) >= batch_size:
                    for task_data in task_batch:
                        create_wrike_task(wrike, project['id'], task_data)
                    task_batch = []

            except Exception as e:
                logger.error(f"Failed to process row {row.row_number}: {e}")
                continue

        # Process remaining tasks in batch
        for task_data in task_batch:
            create_wrike_task(wrike, project['id'], task_data)

        page += 1

def main():
    """Main function to initialize clients and start migration."""
    try:
        config = load_config()
        smartsheet_client = smartsheet.Smartsheet(os.getenv('SMARTSHEET_ACCESS_TOKEN', config['Smartsheet']['access_token']))
        wrike = Wrike(os.getenv('WRIKE_ACCESS_TOKEN', config['Wrike']['access_token']))
        migrate_data(smartsheet_client, wrike, config)
    except Exception as e:
        logger.error(f"Migration failed: {e}")

if __name__ == "__main__":
    main()