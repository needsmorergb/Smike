import asyncio
import logging
import configparser
import smartsheet
import PyWrike
import re
import os
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Generator, Tuple, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    filename='migration.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MigrationConfig:
    """Configuration data class for migration settings."""
    smartsheet_token: str
    smartsheet_sheet_id: str
    wrike_token: str
    wrike_folder_id: str
    reserved_columns: List[str]
    page_size: int = 100
    batch_size: int = 50
    max_concurrent_tasks: int = 10

    @classmethod
    def from_config_file(cls, config_file: str = 'config.ini', sheet_id: Optional[str] = None, folder_id: Optional[str] = None) -> 'MigrationConfig':
        """Load configuration from file with validation and optional CLI overrides."""
        config = configparser.ConfigParser()
        config.read(config_file)
        
        # Validate required sections (excluding IDs if provided via CLI)
        required_sections = {
            'Smartsheet': ['access_token'],
            'Wrike': ['access_token'],
            'Settings': ['reserved_columns']
        }
        
        # Add ID requirements only if not provided via CLI
        if sheet_id is None:
            required_sections['Smartsheet'].append('sheet_id')
        if folder_id is None:
            required_sections['Wrike'].append('folder_id')
        
        for section, keys in required_sections.items():
            if section not in config:
                raise ValueError(f"Missing section '{section}' in config.ini")
            for key in keys:
                if key not in config[section] or not config[section][key]:
                    raise ValueError(f"Missing or empty key '{key}' in section '{section}'")
        
        # Use CLI arguments if provided, otherwise fall back to config
        final_sheet_id = sheet_id or config['Smartsheet']['sheet_id']
        final_folder_id = folder_id or config['Wrike']['folder_id']
        
        return cls(
            smartsheet_token=os.getenv('SMARTSHEET_ACCESS_TOKEN', config['Smartsheet']['access_token']),
            smartsheet_sheet_id=final_sheet_id,
            wrike_token=os.getenv('WRIKE_ACCESS_TOKEN', config['Wrike']['access_token']),
            wrike_folder_id=final_folder_id,
            reserved_columns=[col.strip() for col in config['Settings']['reserved_columns'].split(',')],
            page_size=config.getint('Settings', 'page_size', fallback=100),
            batch_size=config.getint('Settings', 'batch_size', fallback=50),
            max_concurrent_tasks=config.getint('Settings', 'max_concurrent_tasks', fallback=10)
        )


@dataclass
class TaskData:
    """Data class for task information."""
    title: str
    status: str
    responsibles: List[str] = field(default_factory=list)
    custom_fields: List[Dict[str, str]] = field(default_factory=list)
    row_number: Optional[int] = None

    def __post_init__(self):
        if not self.title or not self.title.strip():
            raise ValueError(f"Task title cannot be empty (row {self.row_number})")
        self.title = self.title.strip()


@dataclass
class SheetPage:
    """Data class for paginated sheet data."""
    rows: List[Any]
    columns: List[Any]
    column_map: Dict[str, int]
    page_number: int


class EmailValidator:
    """Utility class for email validation and extraction."""
    
    EMAIL_PATTERN = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
    
    @classmethod
    def extract_emails(cls, cell_value: str) -> List[str]:
        """Extract and validate email addresses from a cell value."""
        if not cell_value:
            return []
        
        try:
            emails = cls.EMAIL_PATTERN.findall(str(cell_value))
            # Additional validation could be added here
            return [email.lower().strip() for email in emails if cls._is_valid_email(email)]
        except Exception as e:
            logger.warning(f"Error extracting emails from '{cell_value}': {e}")
            return []
    
    @staticmethod
    def _is_valid_email(email: str) -> bool:
        """Basic email validation."""
        return len(email) > 5 and '@' in email and '.' in email.split('@')[-1]


class WrikeCustomFieldManager:
    """Manager for Wrike custom fields operations using PyWrike."""
    
    def __init__(self, wrike_token: str):
        self.wrike_token = wrike_token
        self._field_cache: Optional[Dict[str, str]] = None
        # Set up PyWrike authentication
        PyWrike.wrike.WRIKE_ACCESS_TOKEN = wrike_token
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_existing_fields(self) -> Dict[str, str]:
        """Get existing custom fields with caching."""
        if self._field_cache is not None:
            return self._field_cache
            
        try:
            custom_fields = PyWrike.get_custom_fields()
            self._field_cache = {field['title']: field['id'] for field in custom_fields}
            logger.info(f"Retrieved {len(self._field_cache)} existing custom fields")
            return self._field_cache
        except Exception as e:
            logger.error(f"Failed to retrieve Wrike custom fields: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def create_field(self, title: str) -> str:
        """Create a single custom field."""
        try:
            field_id = PyWrike.create_custom_field(title, "Text")
            logger.info(f"Created custom field '{title}' with ID {field_id}")
            
            # Update cache
            if self._field_cache is not None:
                self._field_cache[title] = field_id
                
            return field_id
        except Exception as e:
            logger.error(f"Failed to create custom field '{title}': {e}")
            raise
    
    async def ensure_fields_exist(self, column_titles: List[str], reserved_columns: List[str]) -> Dict[str, str]:
        """Ensure all required custom fields exist, creating them if necessary."""
        existing_fields = await self.get_existing_fields()
        field_map = {}
        
        # Filter out reserved columns
        required_fields = [title for title in column_titles if title not in reserved_columns]
        
        # Create missing fields
        missing_fields = [title for title in required_fields if title not in existing_fields]
        
        if missing_fields:
            logger.info(f"Creating {len(missing_fields)} missing custom fields")
            # Create fields sequentially to avoid API rate limits
            for title in missing_fields:
                field_id = await self.create_field(title)
                field_map[title] = field_id
                # Small delay to prevent rate limiting
                await asyncio.sleep(0.1)
        
        # Add existing fields to map
        for title in required_fields:
            if title in existing_fields:
                field_map[title] = existing_fields[title]
        
        return field_map


class SmartsheetDataProvider:
    """Provider for paginated Smartsheet data."""
    
    def __init__(self, client, config: MigrationConfig):
        self.client = client
        self.config = config
    
    def get_paginated_data(self) -> Generator[SheetPage, None, None]:
        """Generator that yields paginated sheet data."""
        page = 1
        
        while True:
            try:
                sheet = self.client.Sheets.get_sheet(
                    self.config.smartsheet_sheet_id,
                    page_size=self.config.page_size,
                    page=page
                )
                logger.info(f"Retrieved page {page} with {len(sheet.rows)} rows")
                
                if not sheet.rows:
                    logger.info("No more rows to process")
                    break
                
                column_map = {col.title: col.id for col in sheet.columns}
                
                # Validate required columns on first page
                if page == 1:
                    self._validate_required_columns(column_map)
                
                yield SheetPage(
                    rows=sheet.rows,
                    columns=sheet.columns,
                    column_map=column_map,
                    page_number=page
                )
                
                page += 1
                
            except smartsheet.exceptions.RateLimitExceededError as e:
                logger.error(f"Smartsheet rate limit exceeded: {e}")
                raise
            except smartsheet.exceptions.HttpError as e:
                logger.error(f"Smartsheet API error: {e}")
                raise
    
    def _validate_required_columns(self, column_map: Dict[str, int]) -> None:
        """Validate that all required columns exist."""
        missing_columns = [col for col in self.config.reserved_columns if col not in column_map]
        if missing_columns:
            error_msg = f"Required columns missing: {missing_columns}"
            logger.error(error_msg)
            raise ValueError(error_msg)


class TaskDataProcessor:
    """Processor for converting Smartsheet rows to Wrike task data."""
    
    def __init__(self, config: MigrationConfig, custom_field_map: Dict[str, str]):
        self.config = config
        self.custom_field_map = custom_field_map
    
    def process_rows(self, rows: List[Any], column_map: Dict[str, int]) -> List[TaskData]:
        """Process multiple rows into TaskData objects."""
        tasks = []
        
        for row in rows:
            try:
                task_data = self._process_single_row(row, column_map)
                if task_data:
                    tasks.append(task_data)
            except Exception as e:
                logger.error(f"Failed to process row {row.row_number}: {e}")
                continue
        
        return tasks
    
    def _process_single_row(self, row: Any, column_map: Dict[str, int]) -> Optional[TaskData]:
        """Process a single row into TaskData."""
        # Get task name
        task_name_cell = row.cells.get(column_map.get("Task Name"))
        if not task_name_cell or not task_name_cell.value:
            logger.warning(f"Skipping row {row.row_number}: Empty task name")
            return None
        
        # Get status
        status_cell = row.cells.get(column_map.get("Status"))
        status = "Completed" if status_cell and status_cell.value == "Completed" else "Active"
        
        # Get assignees
        assigned_cell = row.cells.get(column_map.get("Assigned To"))
        responsibles = EmailValidator.extract_emails(assigned_cell.value if assigned_cell else "")
        
        # Prepare custom fields
        custom_fields = self._prepare_custom_fields(row, column_map)
        
        return TaskData(
            title=task_name_cell.value,
            status=status,
            responsibles=responsibles,
            custom_fields=custom_fields,
            row_number=row.row_number
        )
    
    def _prepare_custom_fields(self, row: Any, column_map: Dict[str, int]) -> List[Dict[str, str]]:
        """Prepare custom field data for a row."""
        custom_fields = []
        
        for col_title, col_id in column_map.items():
            if col_title not in self.config.reserved_columns and col_title in self.custom_field_map:
                cell = row.cells.get(col_id)
                if cell and cell.value:
                    custom_fields.append({
                        "id": self.custom_field_map[col_title],
                        "value": str(cell.value)
                    })
        
        return custom_fields


class WrikeTaskCreator:
    """Handler for creating Wrike tasks using PyWrike."""
    
    def __init__(self, wrike_token: str, config: MigrationConfig):
        self.wrike_token = wrike_token
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_tasks)
        # Set up PyWrike authentication
        PyWrike.wrike.WRIKE_ACCESS_TOKEN = wrike_token
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def create_task(self, project_id: str, task_data: TaskData) -> bool:
        """Create a single task with rate limiting."""
        async with self.semaphore:
            try:
                # Use PyWrike's create_task function
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    task_id = await loop.run_in_executor(
                        executor,
                        lambda: PyWrike.create_task_in_folder(
                            folder_id=project_id,
                            title=task_data.title,
                            status=task_data.status,
                            responsibles=task_data.responsibles,
                            custom_fields=task_data.custom_fields
                        )
                    )
                
                logger.info(f"Created task: {task_data.title}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to create task '{task_data.title}': {e}")
                raise
    
    async def create_tasks_batch(self, project_id: str, tasks: List[TaskData]) -> Tuple[int, int]:
        """Create multiple tasks concurrently."""
        logger.info(f"Creating batch of {len(tasks)} tasks")
        
        # Create tasks concurrently
        results = await asyncio.gather(
            *[self.create_task(project_id, task) for task in tasks],
            return_exceptions=True
        )
        
        # Count successes and failures
        successes = sum(1 for result in results if result is True)
        failures = len(results) - successes
        
        if failures > 0:
            logger.warning(f"Batch completed: {successes} successes, {failures} failures")
        else:
            logger.info(f"Batch completed successfully: {successes} tasks created")
        
        return successes, failures


class SmartsheetWrikeMigrator:
    """Main migrator class that orchestrates the migration process."""
    
    def __init__(self, config: MigrationConfig):
        self.config = config
        self.smartsheet_client = smartsheet.Smartsheet(config.smartsheet_token)
        
        # Set up PyWrike authentication
        PyWrike.wrike.WRIKE_ACCESS_TOKEN = config.wrike_token
        
        self.custom_field_manager = WrikeCustomFieldManager(config.wrike_token)
        self.task_creator = WrikeTaskCreator(config.wrike_token, config)
        self.stats = {
            'total_rows_processed': 0,
            'tasks_created': 0,
            'tasks_failed': 0,
            'pages_processed': 0
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def create_wrike_project(self) -> Dict[str, Any]:
        """Create the target Wrike project using PyWrike."""
        try:
            project_title = f"{self.config.smartsheet_sheet_id} (Migrated from Smartsheet)"
            
            # Try different PyWrike project creation methods
            try:
                # Try with parent folder
                project_id = PyWrike.create_wrike_project(
                    title=project_title,
                    parent_folder_id=self.config.wrike_folder_id
                )
            except TypeError:
                try:
                    # Try with just title and set folder separately
                    project_id = PyWrike.create_wrike_project(title=project_title)
                except TypeError:
                    try:
                        # Try create_folder_or_project instead
                        project_id = PyWrike.create_folder_or_project(
                            parent_id=self.config.wrike_folder_id,
                            title=project_title,
                            is_project=True
                        )
                    except (TypeError, AttributeError):
                        # Fallback to create_folder with project=True
                        project_id = PyWrike.create_folder(
                            parent_folder_id=self.config.wrike_folder_id,
                            title=project_title,
                            project=True
                        )
            
            project = {'id': project_id, 'title': project_title}
            logger.info(f"Created Wrike project: {project_id}")
            return project
        except Exception as e:
            logger.error(f"Failed to create Wrike project: {e}")
            raise
    
    async def migrate(self) -> None:
        """Execute the complete migration process."""
        logger.info("Starting migration process")
        start_time = time.time()
        
        try:
            # Create project
            project = self.create_wrike_project()
            
            # Setup data provider
            data_provider = SmartsheetDataProvider(self.smartsheet_client, self.config)
            
            # Process pages
            custom_field_map = None
            
            for page_data in data_provider.get_paginated_data():
                # Setup custom fields on first page
                if custom_field_map is None:
                    custom_field_map = await self._setup_custom_fields(page_data.columns)
                
                # Process this page
                await self._process_page(project['id'], page_data, custom_field_map)
            
            # Log final statistics
            duration = time.time() - start_time
            self._log_migration_summary(duration)
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
    
    async def _setup_custom_fields(self, columns: List[Any]) -> Dict[str, str]:
        """Setup custom fields for the migration."""
        logger.info("Setting up custom fields")
        column_titles = [col.title for col in columns]
        return await self.custom_field_manager.ensure_fields_exist(
            column_titles, 
            self.config.reserved_columns
        )
    
    async def _process_page(self, project_id: str, page_data: SheetPage, custom_field_map: Dict[str, str]) -> None:
        """Process a single page of data."""
        logger.info(f"Processing page {page_data.page_number}")
        
        # Convert rows to task data
        processor = TaskDataProcessor(self.config, custom_field_map)
        tasks = processor.process_rows(page_data.rows, page_data.column_map)
        
        if not tasks:
            logger.warning(f"No valid tasks found on page {page_data.page_number}")
            return
        
        # Create tasks in batches
        for i in range(0, len(tasks), self.config.batch_size):
            batch = tasks[i:i + self.config.batch_size]
            successes, failures = await self.task_creator.create_tasks_batch(project_id, batch)
            
            # Update statistics
            self.stats['tasks_created'] += successes
            self.stats['tasks_failed'] += failures
        
        self.stats['total_rows_processed'] += len(page_data.rows)
        self.stats['pages_processed'] += 1
    
    def _log_migration_summary(self, duration: float) -> None:
        """Log comprehensive migration summary."""
        logger.info("=" * 50)
        logger.info("MIGRATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Pages processed: {self.stats['pages_processed']}")
        logger.info(f"Total rows processed: {self.stats['total_rows_processed']}")
        logger.info(f"Tasks created successfully: {self.stats['tasks_created']}")
        logger.info(f"Tasks failed: {self.stats['tasks_failed']}")
        
        if self.stats['total_rows_processed'] > 0:
            success_rate = (self.stats['tasks_created'] / self.stats['total_rows_processed']) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        
        logger.info("=" * 50)


async def main():
    """Main async function to run the migration."""
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Migrate data from Smartsheet to Wrike',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python migrator.py --sheet-id 123456789 --folder-id IEAAAAAA
  python migrator.py -s 123456789 -f IEAAAAAA
  python migrator.py --sheet-id 123456789 --folder-id IEAAAAAA --config custom_config.ini
        """
    )
    
    parser.add_argument(
        '--sheet-id', '-s',
        type=str,
        help='Smartsheet sheet ID (overrides config.ini value)'
    )
    
    parser.add_argument(
        '--folder-id', '-f', 
        type=str,
        help='Wrike folder ID (overrides config.ini value)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.ini',
        help='Path to configuration file (default: config.ini)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging to console'
    )
    
    args = parser.parse_args()
    
    # Setup console logging if verbose
    if args.verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(console_handler)
    
    try:
        # Load configuration with CLI overrides
        config = MigrationConfig.from_config_file(
            config_file=args.config,
            sheet_id=args.sheet_id,
            folder_id=args.folder_id
        )
        
        logger.info("Configuration loaded successfully")
        logger.info(f"Sheet ID: {config.smartsheet_sheet_id}")
        logger.info(f"Folder ID: {config.wrike_folder_id}")
        
        # Create and run migrator
        migrator = SmartsheetWrikeMigrator(config)
        await migrator.migrate()
        
        logger.info("Migration completed successfully")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


def run_migration():
    """Synchronous entry point for the migration."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Migration interrupted by user")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        print(f"Migration failed. Check migration.log for details. Error: {e}")


if __name__ == "__main__":
    run_migration()
