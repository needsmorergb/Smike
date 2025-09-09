#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smartsheet → Wrike (Custom Fields aware)

What it does:
- Reads Smartsheet columns + rows
- Maps Smartsheet columns to Wrike Custom Fields by exact title
- Optionally auto-creates simple CFs (Text/Date/Checkbox/Number)
- Creates a Wrike Project under a given parent and creates tasks
- Writes CF values for each task via Wrike form-encoded 'customFields'

Notes:
- Wrike task creation must be application/x-www-form-urlencoded.
- Wrike 'customFields' parameter is a JSON array string: [{"id":"CFID","value":"..."}]
- For reliability under deadline, manually create Dropdown CFs in Wrike with titles
  that exactly match Smartsheet column names. Then the script will populate values.

Env:
  SMARTSHEET_ACCESS_TOKEN
  WRIKE_ACCESS_TOKEN
"""

import os
import sys
import json
import time
import math
import argparse
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import requests

# ---------------------------
# Logging
# ---------------------------
LOG = logging.getLogger("ss2wr_cf")
def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# ---------------------------
# Config / Constants
# ---------------------------
WRIKE_API = "https://www.wrike.com/api/v4"
SS_API    = "https://api.smartsheet.com/2.0"

DEFAULT_MAX_ROWS = 100

# ---------------------------
# HTTP helpers
# ---------------------------
def build_session() -> requests.Session:
    s = requests.Session()
    # ASCII ONLY — no Unicode arrow
    s.headers["User-Agent"] = "SS-to-Wrike CF Migrator/1.0"
    s.verify = True
    return s

def call_api(
    session: requests.Session,
    method: str,
    url: str,
    *,
    headers: Dict[str, str],
    params: Dict[str, Any] | None = None,
    data: Dict[str, Any] | None = None,
    json_body: Any | None = None,
    service: str = "",
    max_retries: int = 3,
    backoff: float = 1.0,
) -> requests.Response:
    """Small wrapper with light retry for 429/5xx."""
    for attempt in range(1, max_retries + 1):
        resp = session.request(
            method, url, headers=headers, params=params, data=data, json=json_body, timeout=30
        )
        if resp.status_code in (429, 500, 502, 503, 504):
            LOG.warning("[%s] %s %s -> %s; retrying in %.1fs",
                        service or "api", method, url, resp.status_code, backoff)
            time.sleep(backoff)
            backoff *= 2
            continue
        # If Wrike/SS returns 4xx other than 429, surface it immediately
        if 400 <= resp.status_code < 600:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            LOG.error("[%s] %s %s -> %s body=%s", service or "api", method, url, resp.status_code, body)
            resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp

# ---------------------------
# Smartsheet client
# ---------------------------
class SmartsheetClient:
    def __init__(self, token: str, sess: requests.Session):
        self.s = sess
        self.h = {"Authorization": f"Bearer {token}"}

    def whoami(self) -> Dict[str, Any]:
        r = call_api(self.s, "GET", f"{SS_API}/users/me", headers=self.h, service="smartsheet")
        rj = r.json()
        # handle both shapes: wrapped and unwrapped
        if isinstance(rj, dict) and "data" in rj:
            return rj["data"]
        return rj

    def get_sheet_meta(self, sheet_id: int) -> Dict[str, Any]:
        # Basic info incl. columns; this endpoint usually returns top-level sheet object
        r = call_api(
            self.s, "GET", f"{SS_API}/sheets/{sheet_id}?page=1&pageSize=1",
            headers=self.h, service="smartsheet"
        )
        rj = r.json()
        # Be defensive in case API returns wrapped list
        if isinstance(rj, dict) and "data" in rj:
            # sometimes it's a list under data
            d = rj["data"]
            if isinstance(d, list) and d:
                return d[0]
            if isinstance(d, dict):
                return d
        return rj

    def iter_rows(self, sheet_id: int, max_rows: int | None = None):
        """Yield rows page by page. Uses page+pageSize, which worked in your env."""
        page = 1
        fetched = 0
        page_size = 200
        while True:
            r = call_api(
                self.s, "GET",
                f"{SS_API}/sheets/{sheet_id}?page={page}&pageSize={page_size}",
                headers=self.h, service="smartsheet"
            ).json()
            rows = r.get("rows", [])
            if not rows:
                break
            for row in rows:
                yield row
                fetched += 1
                if max_rows and fetched >= max_rows:
                    return
            page += 1

# ---------------------------
# Wrike client
# ---------------------------
class WrikeClient:
    def __init__(self, token: str, sess: requests.Session):
        self.s = sess
        self.h = {"Authorization": f"Bearer {token}"}

    def get_folder(self, folder_id: str) -> Dict[str, Any]:
        r = call_api(self.s, "GET", f"{WRIKE_API}/folders/{folder_id}", headers=self.h, service="wrike")
        return r.json()["data"][0]

    def list_contacts(self) -> List[Dict[str, Any]]:
        # NOTE: Wrike /contacts does NOT accept pageSize
        r = call_api(self.s, "GET", f"{WRIKE_API}/contacts", headers=self.h, service="wrike")
        return r.json()["data"]

    def list_custom_fields(self) -> List[Dict[str, Any]]:
        r = call_api(self.s, "GET", f"{WRIKE_API}/customfields", headers=self.h, service="wrike")
        return r.json()["data"]

    def create_project(self, parent_folder_id: str, title: str) -> str:
        # Must be FORM data
        r = call_api(
            self.s, "POST", f"{WRIKE_API}/folders/{parent_folder_id}/folders",
            headers=self.h, data={"title": title}, service="wrike"
        )
        return r.json()["data"][0]["id"]

    def create_task(
        self,
        folder_id: str,
        *,
        title: str,
        description: Optional[str] = None,
        custom_fields: Optional[List[Dict[str, Any]]] = None,
        responsible_ids: Optional[List[str]] = None,
        due_date_iso: Optional[str] = None,
    ) -> str:
        data: Dict[str, Any] = {"title": title}

        if description:
            data["description"] = description

        # dates[due] must be YYYY-MM-DD
        if due_date_iso:
            data["dates[due]"] = due_date_iso

        if responsible_ids:
            data["responsibleIds"] = ",".join(responsible_ids)

        if custom_fields:
            # IMPORTANT: this must be JSON string for form-encoded payload
            data["customFields"] = json.dumps(custom_fields, ensure_ascii=False)

        r = call_api(
            self.s, "POST", f"{WRIKE_API}/folders/{folder_id}/tasks",
            headers=self.h, data=data, service="wrike"
        )
        return r.json()["data"][0]["id"]

    # (Optional) very conservative CF creation (safe types only)
    def create_custom_field(
        self,
        *,
        title: str,
        ftype: str,          # "Text"|"Number"|"Date"|"Checkbox"
        space_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Attempts to create a CF. We stick to simple, widely supported types to avoid 400s.
        If your token/role can't create CFs or Space scoping is restricted, this will 403/400.
        """
        payload = {"title": title, "type": ftype}
        if space_id:
            payload["spaceId"] = space_id

        # Wrike CF creation supports JSON body.
        r = call_api(self.s, "POST", f"{WRIKE_API}/customfields",
                     headers={**self.h, "Content-Type": "application/json"},
                     json_body=payload, service="wrike")
        try:
            return r.json()["data"][0]["id"]
        except Exception:
            return None

# ---------------------------
# Mapping helpers
# ---------------------------
# Smartsheet → Wrike simple type mapping (safe set)
SS_TO_WRIKE_SIMPLE = {
    "TEXT_NUMBER": "Text",
    "DATE":        "Date",
    "CHECKBOX":    "Checkbox",
    "PICKLIST":    "Text",      # recommend manual Dropdown create in Wrike; value still stored as text
    "CONTACT_LIST":"Text",      # store emails/names as text CF (Wrike CF "User" type is not used here)
}

def smartsheet_date_to_iso(cell: Dict[str, Any]) -> Optional[str]:
    v = cell.get("value")
    if not v:
        v = cell.get("displayValue")
    if not v:
        return None
    # Smartsheet dates are usually 'YYYY-MM-DD'
    try:
        # If already ISO date
        if isinstance(v, str) and len(v) >= 10:
            return v[:10]
    except Exception:
        return None
    return None

def smartsheet_checkbox_to_bool_str(cell: Dict[str, Any]) -> Optional[str]:
    v = cell.get("value")
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "1", "y"):
            return "true"
        if s in ("false", "no", "0", "n"):
            return "false"
    return None

def cell_text(cell: Dict[str, Any]) -> Optional[str]:
    # prefer displayValue then value, stringify
    v = cell.get("displayValue")
    if v is None:
        v = cell.get("value")
    if v is None:
        return None
    return str(v)

def build_cf_plan(
    ss_columns: List[Dict[str, Any]],
    existing_wrike_cfs: List[Dict[str, Any]]
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      - plan_by_title: {col_title: {"ss_type":..., "wr_type":..., "exists": bool, "wr_id": optional}}
      - missing: list of specs to create (safe types only)
    """
    wr_by_title = {cf["title"]: cf for cf in existing_wrike_cfs}

    plan: Dict[str, Dict[str, Any]] = {}
    missing: List[Dict[str, Any]] = []

    for col in ss_columns:
        title = col.get("title")
        ss_type = col.get("type")
        if not title or not ss_type:
            continue
        wr_type = SS_TO_WRIKE_SIMPLE.get(ss_type, "Text")
        entry = {
            "ss_type": ss_type,
            "wr_type": wr_type,
            "exists": title in wr_by_title,
            "wr_id": wr_by_title.get(title, {}).get("id"),
        }
        plan[title] = entry
        if not entry["exists"]:
            # only propose CF creation if it's a safe simple type
            if wr_type in ("Text", "Date", "Checkbox", "Number"):
                missing.append({"title": title, "wr_type": wr_type})
    return plan, missing

def row_to_wrike_custom_fields(
    row: Dict[str, Any],
    ss_columns: List[Dict[str, Any]],
    plan_by_title: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build Wrike CF list for this row based on plan_by_title (only fields that exist in Wrike)."""
    col_by_id = {c["id"]: c for c in ss_columns}
    cf_list: List[Dict[str, Any]] = []

    for cell in row.get("cells", []):
        col_id = cell.get("columnId")
        col = col_by_id.get(col_id)
        if not col:
            continue
        title = col.get("title")
        if not title:
            continue
        plan = plan_by_title.get(title)
        if not plan or not plan.get("exists") or not plan.get("wr_id"):
            continue

        wr_id = plan["wr_id"]
        ss_type = plan["ss_type"]
        wr_type = plan["wr_type"]

        value: Optional[str] = None
        if wr_type == "Date" or ss_type == "DATE":
            value = smartsheet_date_to_iso(cell)
        elif wr_type == "Checkbox" or ss_type == "CHECKBOX":
            value = smartsheet_checkbox_to_bool_str(cell)
        else:
            value = cell_text(cell)

        if value is not None and value != "":
            cf_list.append({"id": wr_id, "value": value})

    return cf_list

# ---------------------------
# Main migration
# ---------------------------
def migrate_sheet_to_wrike(
    ss: SmartsheetClient,
    wr: WrikeClient,
    *,
    sheet_id: int,
    wrike_parent_id: str,
    title_column: Optional[str],
    max_rows: int,
    plan_only: bool,
    auto_create_cf: bool,
    allow_dropdown_create: bool,
    verbose: bool,
) -> Dict[str, Any]:
    # 1) fetch sheet meta
    meta = ss.get_sheet_meta(sheet_id)
    sheet_name = meta.get("name") or f"Sheet {sheet_id}"
    columns = meta.get("columns", [])
    if not columns:
        raise RuntimeError("No columns found on Smartsheet sheet")

    # 2) gather Wrike CFs and make a plan
    wr_cfs = wr.list_custom_fields()
    plan_by_title, missing = build_cf_plan(columns, wr_cfs)

    if plan_only:
        LOG.info("Plan-only: Smartsheet → Wrike CF mapping for sheet '%s'", sheet_name)
        for t, spec in plan_by_title.items():
            LOG.info("  %-40s  SS=%-12s → Wrike=%-9s  %s%s",
                     t,
                     spec["ss_type"],
                     spec["wr_type"],
                     "EXISTS" if spec["exists"] else "MISSING",
                     f" (id={spec['wr_id']})" if spec.get("wr_id") else ""
                     )
        if missing:
            LOG.info("Missing CFs (safe to auto-create): %s",
                     ", ".join([f"{m['title']}:{m['wr_type']}" for m in missing]))
        else:
            LOG.info("No missing CFs (by exact title).")
        return {"mode": "plan", "sheet": sheet_name, "missing": missing}

    # 3) ensure project
    created_title = f"{sheet_name} (Migrated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')})"
    # If you already created a project manually, you can reuse its id by changing here.
    project_id = wr.create_project(wrike_parent_id, created_title)
    LOG.info("Created Wrike project '%s' (%s)", created_title, project_id)

    # 4) (optional) attempt CF creation if missing (safe types only)
    if auto_create_cf and missing:
        # try to discover the spaceId from the parent folder; if not present, create account-level
        space_id = None
        try:
            parent_obj = wr.get_folder(wrike_parent_id)
            space_id = parent_obj.get("spaceId")
        except Exception:
            space_id = None

        updated_wr_cfs = {cf["title"]: cf for cf in wr.list_custom_fields()}
        for m in missing:
            title = m["title"]
            wr_type = m["wr_type"]
            if wr_type not in ("Text", "Date", "Checkbox", "Number"):
                if wr_type == "Text":
                    pass
                elif wr_type == "PICKLIST" and not allow_dropdown_create:
                    LOG.warning("Skipping auto-create for '%s' (Dropdown). Create it manually in Wrike.", title)
                    continue
            if title in updated_wr_cfs:
                continue
            cfid = wr.create_custom_field(title=title, ftype=wr_type, space_id=space_id)
            if cfid:
                LOG.info("Auto-created CF '%s' (%s, %s)", title, wr_type, cfid)
                updated_wr_cfs[title] = {"id": cfid, "title": title, "type": wr_type}
            else:
                LOG.warning("Failed to auto-create CF '%s' (%s). Please create it manually.", title, wr_type)
        # refresh plan with any new CFs
        wr_cfs = list(updated_wr_cfs.values())
        plan_by_title, _ = build_cf_plan(columns, wr_cfs)

    # 5) fetch rows and create tasks with CFs
    # cache contacts (for potential assignee mapping later if needed)
    try:
        contacts = wr.list_contacts()
        LOG.info("Wrike contacts cached: %d", len(contacts))
    except Exception:
        contacts = []

    # Index Smartsheet columns for quick lookup
    col_by_title = {c["title"]: c for c in columns}
    title_col_id = None
    if title_column:
        if title_column not in col_by_title:
            raise RuntimeError(f"--title-column '{title_column}' not found on Smartsheet sheet.")
        title_col_id = col_by_title[title_column]["id"]

    created = 0
    for row in ss.iter_rows(sheet_id, max_rows=max_rows):
        # Build title
        row_title = None
        if title_col_id:
            for cell in row.get("cells", []):
                if cell.get("columnId") == title_col_id:
                    row_title = cell_text(cell)
                    break
        if not row_title:
            # fallback: first cell with displayValue
            for cell in row.get("cells", []):
                t = cell_text(cell)
                if t:
                    row_title = t
                    break
        if not row_title:
            row_title = f"Row {row.get('id')}"

        # Build CF values
        cf_values = row_to_wrike_custom_fields(row, columns, plan_by_title)

        # (Optional) Try a due date if there is a DATE column called "Due Date"
        due_iso = None
        due_col = col_by_title.get("Due Date")
        if due_col:
            for cell in row.get("cells", []):
                if cell.get("columnId") == due_col["id"]:
                    due_iso = smartsheet_date_to_iso(cell)
                    break

        # Create task in Wrike with CFs
        tid = wr.create_task(
            project_id,
            title=row_title[:255],
            description=None,
            custom_fields=cf_values if cf_values else None,
            responsible_ids=None,
            due_date_iso=due_iso,
        )
        created += 1
        if verbose:
            LOG.info("Created task '%s' (%s)%s", row_title[:60], tid, " with CFs" if cf_values else "")

    LOG.info("Done. Created %d task(s) in Wrike project %s", created, project_id)
    return {"project_id": project_id, "created_tasks": created}

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Smartsheet → Wrike (Custom Fields aware)")
    p.add_argument("--sheet-id", type=int, required=True, help="Smartsheet sheet ID")
    p.add_argument("--wrike-parent", type=str, required=True, help="Wrike parent folder ID to create the project under")
    p.add_argument("--title-column", type=str, default=None, help="Smartsheet column to use for Wrike task titles")
    p.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS, help="Max rows to migrate")
    p.add_argument("--plan-only", action="store_true", help="Only show CF plan; do not create anything")
    p.add_argument("--auto-create-cf", action="store_true", help="Attempt to auto-create missing CFs (safe types only)")
    p.add_argument("--allow-dropdown-create", action="store_true",
                   help="(Not recommended) also try to auto-create Dropdown CFs")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()

def main():
    args = parse_args()
    setup_logging(args.verbose)

    ss_token = os.environ.get("SMARTSHEET_ACCESS_TOKEN") or os.environ.get("SMARTSHEET_TOKEN")
    wr_token = os.environ.get("WRIKE_ACCESS_TOKEN")
    if not ss_token or not wr_token:
        LOG.error("Missing tokens. Set SMARTSHEET_ACCESS_TOKEN and WRIKE_ACCESS_TOKEN.")
        sys.exit(1)

    sess = build_session()
    ss = SmartsheetClient(ss_token, sess)
    wr = WrikeClient(wr_token, sess)

    # quick validations
    try:
        me_ss = ss.whoami()
        parent = wr.get_folder(args.wrike_parent)
        LOG.info("Setup validation OK.")
    except Exception as e:
        LOG.error("Validation failed: %s", e)
        sys.exit(1)

    try:
        out = migrate_sheet_to_wrike(
            ss, wr,
            sheet_id=args.sheet_id,
            wrike_parent_id=args.wrike_parent,
            title_column=args.title_column,
            max_rows=args.max_rows,
            plan_only=args.plan_only,
            auto_create_cf=args.auto_create_cf,
            allow_dropdown_create=args.allow_dropdown_create,
            verbose=args.verbose
        )
        if args.plan_only:
            LOG.info("Plan complete. See above for CFs to create (recommended for Dropdowns).")
        else:
            LOG.info("Migration complete: %s", json.dumps(out, indent=2))
    except Exception as e:
        LOG.exception("Fatal: %s", e)
        sys.exit(2)

if __name__ == "__main__":
    main()
