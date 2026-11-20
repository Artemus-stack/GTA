"""Google Sheet that tracks what hits.

Tabs (created automatically):
  * Posts     - one row per post; metrics are filled in as they arrive. You can
                also type Views/Likes/Comments/Shares by hand: the analyst reads
                them back for posts it has no API numbers for.
  * Formats   - which layout / caption style / length / hook / title style is winning
  * Streamers - hit rate per streamer, and how much the editor prioritises them

Auth: a Google Cloud service account JSON key, with the sheet shared to the
service account's email as an editor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("sheets")

POSTS_HEADER = [
    "Posted At (UTC)", "Post ID", "Clip ID", "Platform", "Account", "Streamer", "Source", "Moment Type",
    "Peak Msgs/Min", "Title", "Title Style", "Layout", "Captions", "Length (s)", "Hook", "Status",
    "Post URL", "Views", "Likes", "Comments", "Shares", "Viral Score", "Hit", "Metrics Updated",
    "Source VOD", "Scheduled For (UTC)",
]
FORMATS_HEADER = ["Platform", "Choice", "Option", "Posts", "Hits", "Hit Rate", "Avg Viral Score",
                  "Chance It's Best", "Status"]
STREAMERS_HEADER = ["Streamer", "Source", "Posts", "Hits", "Hit Rate", "Avg Viral Score", "Priority Boost"]
TABS = {"Posts": POSTS_HEADER, "Formats": FORMATS_HEADER, "Streamers": STREAMERS_HEADER}


class NullSheet:
    """Used when no sheet is configured: every call is a no-op."""

    enabled = False

    def ensure_tabs(self) -> None: ...
    def append_posts(self, rows: list[list]) -> None: ...
    def read_posts(self) -> list[dict]: return []
    def update_posts(self, updates: dict[int, dict[str, Any]]) -> None: ...
    def replace(self, name: str, rows: list[list]) -> None: ...


class Sheet:
    enabled = True

    def __init__(self, spreadsheet_id: str, credentials_file: str):
        import gspread

        self._gspread = gspread
        self.gc = gspread.service_account(filename=credentials_file)
        self.book = self.gc.open_by_key(spreadsheet_id)
        self._tabs: dict[str, Any] = {}

    def tab(self, name: str):
        if name in self._tabs:
            return self._tabs[name]
        header = TABS[name]
        try:
            ws = self.book.worksheet(name)
        except self._gspread.WorksheetNotFound:
            ws = self.book.add_worksheet(title=name, rows=1000, cols=len(header))
        if ws.row_values(1) != header:
            ws.update([header], "A1")
            ws.freeze(rows=1)
        self._tabs[name] = ws
        return ws

    def ensure_tabs(self) -> None:
        for name in TABS:
            self.tab(name)

    def append_posts(self, rows: list[list]) -> None:
        if rows:
            self.tab("Posts").append_rows(rows, value_input_option="USER_ENTERED")

    def read_posts(self) -> list[dict]:
        values = self.tab("Posts").get_all_values()
        if not values:
            return []
        header = values[0]
        return [{**dict(zip(header, row)), "_row": i + 2} for i, row in enumerate(values[1:])]

    def update_posts(self, updates: dict[int, dict[str, Any]]) -> None:
        """updates: sheet row number -> {column header: value}."""
        data = []
        for row, cols in updates.items():
            for col_name, value in cols.items():
                col = POSTS_HEADER.index(col_name) + 1
                data.append({"range": self._gspread.utils.rowcol_to_a1(row, col), "values": [[value]]})
        if data:
            self.tab("Posts").batch_update(data, value_input_option="USER_ENTERED")

    def replace(self, name: str, rows: list[list]) -> None:
        ws = self.tab(name)
        ws.clear()
        ws.update([TABS[name]] + rows, "A1", value_input_option="USER_ENTERED")


def open_sheet(spreadsheet_id: str, credentials_file: str) -> Sheet | NullSheet:
    if not spreadsheet_id:
        log.warning("no Google Sheet configured (GSHEET_ID); posts are only logged locally")
        return NullSheet()
    if not Path(credentials_file).is_file():
        log.warning("Google service account file %s not found; posts are only logged locally", credentials_file)
        return NullSheet()
    return Sheet(spreadsheet_id, credentials_file)
