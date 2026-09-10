from __future__ import annotations

from dataclasses import dataclass
import ast
import asyncio
import html
from html.parser import HTMLParser
import json
import ipaddress
import mimetypes
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence
from urllib.parse import parse_qs, unquote, urlparse
import xml.etree.ElementTree as ET
import zipfile


MAX_OBSERVATION_CHARS = 12000
SPREADSHEET_EXTENSIONS = {".xlsx", ".xls", ".csv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac",
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".mpeg", ".mpg",
}
_WHISPER_MODELS: dict[tuple[str, str, str], Any] = {}
_RAPID_OCR_ENGINE: Any | None = None


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._href: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            self._href = values.get("href") or ""
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            self.results.append({
                "title": html.unescape(" ".join(self._parts)).strip(),
                "url": self._href,
            })
            self._href = None
            self._parts = []


class _ReadableTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        elif not self._skip_depth and tag in {"p", "div", "li", "h1", "h2", "h3", "tr", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self.parts.append(html.unescape(data.strip()))

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", " ".join(self.parts)).strip()


class _BingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._result_depth = 0
        self._href: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if tag == "li" and "b_algo" in classes:
            self._result_depth = 1
        elif self._result_depth:
            self._result_depth += 1
            if tag == "a" and self._href is None and values.get("href"):
                self._href = values["href"]
                self._parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            self.results.append({
                "title": html.unescape(" ".join(self._parts)).strip(),
                "url": self._href,
            })
            self._href = None
            self._parts = []
        if self._result_depth:
            self._result_depth -= 1


@dataclass(frozen=True)
class ToolLoopResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    finish_reason: str | None
    tool_trace: tuple[dict[str, Any], ...]


class EpisodeWorkspace:
    """Per-candidate workspace with a copied attachment and explicit artifacts."""

    def __init__(self, metadata: dict[str, Any] | None, root: str | Path | None = None):
        base = Path(root).resolve() if root else None
        if base:
            base.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(
            prefix="mas_gaia_", dir=str(base) if base else None
        )
        self.root = Path(self._temporary.name).resolve()
        self.attachments = self.root / "attachments"
        self.artifacts = self.root / "artifacts"
        self.attachments.mkdir()
        self.artifacts.mkdir()
        self.metadata = metadata or {}
        source = str(self.metadata.get("attachment_source") or "").strip()
        self.attachment: Path | None = None
        if source:
            source_path = Path(source).resolve()
            if not source_path.is_file():
                raise FileNotFoundError(f"attachment not found: {source_path}")
            self.attachment = self.attachments / source_path.name
            shutil.copy2(source_path, self.attachment)
            self.attachment.chmod(0o444)

    def close(self) -> None:
        self._temporary.cleanup()

    def __enter__(self) -> "EpisodeWorkspace":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def resolve(self, value: str) -> Path:
        raw = value.strip()
        if not raw and self.attachment:
            return self.attachment
        candidate = (self.root / raw).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("path escapes episode workspace")
        if candidate.is_file():
            return candidate
        # Let models refer to the supplied attachment by basename.
        basename = Path(raw).name
        matches = list(self.attachments.glob(basename)) if basename else []
        if len(matches) == 1:
            return matches[0]
        raise FileNotFoundError(f"workspace file not found: {value}")


TOOL_DESCRIPTIONS = {
    "workspace_list_files": "List files in the isolated episode workspace. Arguments: {}.",
    "workspace_inspect_file": "Inspect attachment type, size, and basic image metadata. Arguments: {path}.",
    "workspace_read_document": "Automatically read TXT/JSON/XML/PDF/DOCX/PPTX/XLSX/CSV content. Arguments: {path, max_chars?, max_rows?, max_cols?}.",
    "workspace_read_text": "Read a UTF-8-like text attachment. Arguments: {path, max_chars?}.",
    "workspace_read_spreadsheet": "Read XLSX/CSV cells. Arguments: {path, sheet?, max_rows?, max_cols?}.",
    "workspace_read_pdf": "Extract PDF text. Arguments: {path, max_pages?}.",
    "workspace_list_archive": "Safely list ZIP members. Arguments: {path, max_members?}.",
    "workspace_read_archive_member": "Read one text member from a ZIP without extracting it. Arguments: {path, member, max_chars?}.",
    "workspace_extract_archive_member": "Safely copy one bounded ZIP member into episode artifacts so another tool can read it. Arguments: {path, member}.",
    "workspace_read_json": "Parse JSON or JSON-LD and return bounded structured content. Arguments: {path, max_chars?}.",
    "workspace_analyze_pdb": "Summarize PDB chains, residues, atoms, hetero residues, and amino-acid sequences. Arguments: {path, max_residues?}.",
    "workspace_inspect_spreadsheet": "Read XLS/XLSX cells with coordinates, values, formulas, fill and font colors. Arguments: {path, sheet?, max_rows?, max_cols?}.",
    "workspace_analyze_spreadsheet_grid": "Analyze orthogonal adjacency for cells sharing a fill color, including components, degrees, bipartition sizes, and necessary Hamiltonian-cycle conditions. Arguments: {path, fill_rgb, sheet?}.",
    "workspace_ocr_image": "Extract text and bounding boxes from an image with local RapidOCR. Arguments: {path, max_items?}.",
    "workspace_probe_media": "Inspect image/audio/video streams, duration, dimensions, codecs, and tags. Arguments: {path}.",
    "workspace_extract_video_frames": "Extract bounded JPEG frames from a video into episode artifacts. Arguments: {path, interval_seconds?, max_frames?}.",
    "workspace_transcribe_audio": "Transcribe an audio/video file with local faster-whisper. Arguments: {path, model?, language?, max_segments?}. The model is downloaded and cached on first use.",
    "python_calculate": "Run restricted pure Python calculation code. Arguments: {code, timeout?}.",
    "web_search": "Search the public web. Arguments: {query, max_results?}.",
    "web_open": "Open a public HTTP(S) page and extract readable text. Arguments: {url, max_chars?}.",
    "web_download": "Download a bounded public file into episode artifacts for another reader tool. Arguments: {url, filename?, max_bytes?}.",
    "web_find": "Find lines containing a pattern in supplied text. Arguments: {text, pattern, max_matches?}.",
    "wikipedia_search": "Search English Wikipedia through its API. Arguments: {query, max_results?}.",
    "arxiv_search": "Search arXiv metadata. Arguments: {query, max_results?}.",
}


def _clip(value: str, limit: int = MAX_OBSERVATION_CHARS) -> str:
    return value if len(value) <= limit else value[:limit] + "\n[TRUNCATED]"


def _int_arg(args: dict[str, Any], name: str, default: int, low: int, high: int) -> int:
    value = args.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return max(low, min(high, value))


def _read_text(path: Path, max_chars: int) -> str:
    return _clip(path.read_text(encoding="utf-8", errors="replace"), max_chars)


def _xlsx_column_index(reference: str) -> int:
    letters = "".join(char for char in reference if char.isalpha()).upper()
    index = 0
    for char in letters:
        index = index * 26 + ord(char) - ord("A") + 1
    return max(0, index - 1)


def _read_xlsx_stdlib(
    path: Path, requested: str, max_rows: int, max_cols: int
) -> tuple[str, list[str], list[list[Any]]]:
    """Small dependency-free XLSX reader for cached values and shared strings."""
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    package_rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{{{main_ns}}}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{{{main_ns}}}t")))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relation_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relation_targets = {
            relation.attrib["Id"]: relation.attrib["Target"]
            for relation in relation_root.findall(f"{{{package_rel_ns}}}Relationship")
        }
        sheets: list[tuple[str, str]] = []
        for sheet in workbook.findall(f".//{{{main_ns}}}sheet"):
            name = sheet.attrib["name"]
            relation_id = sheet.attrib[f"{{{rel_ns}}}id"]
            target = relation_targets[relation_id].lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target
            sheets.append((name, target))
        if not sheets:
            raise ValueError("XLSX workbook contains no sheets")
        selected = next((item for item in sheets if item[0] == requested), sheets[0] if not requested else None)
        if selected is None:
            raise ValueError(f"sheet not found: {requested}; available: {[name for name, _ in sheets]}")
        sheet_name, sheet_path = selected
        sheet_root = ET.fromstring(archive.read(sheet_path))
        rows: list[list[Any]] = []
        for row_node in sheet_root.findall(f".//{{{main_ns}}}row")[:max_rows]:
            values: list[Any] = [None] * max_cols
            for cell in row_node.findall(f"{{{main_ns}}}c"):
                column = _xlsx_column_index(cell.attrib.get("r", "A1"))
                if column >= max_cols:
                    continue
                kind = cell.attrib.get("t", "")
                value_node = cell.find(f"{{{main_ns}}}v")
                if kind == "inlineStr":
                    inline = cell.find(f"{{{main_ns}}}is")
                    value: Any = "" if inline is None else "".join(
                        node.text or "" for node in inline.iter(f"{{{main_ns}}}t")
                    )
                elif value_node is None:
                    value = None
                elif kind == "s":
                    value = shared[int(value_node.text or "0")]
                elif kind == "b":
                    value = value_node.text == "1"
                else:
                    raw = value_node.text or ""
                    try:
                        number = float(raw)
                        value = int(number) if number.is_integer() else number
                    except ValueError:
                        value = raw
                values[column] = value
            while values and values[-1] is None:
                values.pop()
            rows.append(values)
        return sheet_name, [name for name, _ in sheets], rows


def _read_office_xml(path: Path, kind: str, max_chars: int) -> str:
    with zipfile.ZipFile(path) as archive:
        if kind == "docx":
            names = ["word/document.xml"]
        else:
            names = sorted(
                name for name in archive.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            )
        parts: list[str] = []
        for name in names:
            if name not in archive.namelist():
                continue
            root = ET.fromstring(archive.read(name))
            text = " ".join(node.text or "" for node in root.iter() if node.tag.endswith("}t"))
            if text.strip():
                parts.append(f"[{name}]\n{text.strip()}")
            if sum(len(part) for part in parts) >= max_chars:
                break
        return _clip("\n\n".join(parts), max_chars)


def _safe_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("url must be public HTTP(S)")
    if parsed.username or parsed.password:
        raise ValueError("authenticated URLs are not allowed")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, None)}
    except socket.gaierror as exc:
        raise ValueError(f"cannot resolve URL host: {parsed.hostname}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("private, loopback, or link-local URLs are not allowed")


def _jsonable_metadata(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable_metadata(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_metadata(item) for key, item in value.items()}
    return str(value)


def execute_tool(
    name: str, arguments: dict[str, Any], workspace: EpisodeWorkspace
) -> dict[str, Any]:
    if name == "workspace_list_files":
        files = [
            str(path.relative_to(workspace.root))
            for path in sorted(workspace.root.rglob("*"))
            if path.is_file()
        ]
        return {"files": files}

    if name == "workspace_inspect_file":
        path = workspace.resolve(str(arguments.get("path") or ""))
        result: dict[str, Any] = {
            "path": str(path.relative_to(workspace.root)),
            "size_bytes": path.stat().st_size,
            "extension": path.suffix.lower(),
            "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        }
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            from PIL import Image

            with Image.open(path) as image:
                result["image"] = {
                    "format": image.format,
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "metadata": _jsonable_metadata(dict(image.info)),
                }
        return result

    if name == "workspace_read_document":
        path = workspace.resolve(str(arguments.get("path") or ""))
        extension = path.suffix.lower()
        delegated = dict(arguments)
        delegated["path"] = str(path.relative_to(workspace.root))
        if extension == ".pdf":
            return execute_tool("workspace_read_pdf", delegated, workspace)
        if extension in SPREADSHEET_EXTENSIONS | {".xlsm"}:
            return execute_tool("workspace_read_spreadsheet", delegated, workspace)
        max_chars = _int_arg(arguments, "max_chars", 20000, 1, 50000)
        if extension == ".docx":
            return {"path": path.name, "text": _read_office_xml(path, "docx", max_chars)}
        if extension == ".pptx":
            return {"path": path.name, "text": _read_office_xml(path, "pptx", max_chars)}
        if extension == ".zip":
            return execute_tool("workspace_list_archive", delegated, workspace)
        if extension in MEDIA_EXTENSIONS:
            return execute_tool("workspace_inspect_file", delegated, workspace)
        return execute_tool("workspace_read_text", delegated, workspace)

    if name == "workspace_read_text":
        path = workspace.resolve(str(arguments.get("path") or ""))
        limit = _int_arg(arguments, "max_chars", 12000, 1, 50000)
        return {"path": str(path.relative_to(workspace.root)), "text": _read_text(path, limit)}

    if name == "workspace_read_spreadsheet":
        path = workspace.resolve(str(arguments.get("path") or ""))
        max_rows = _int_arg(arguments, "max_rows", 100, 1, 500)
        max_cols = _int_arg(arguments, "max_cols", 30, 1, 100)
        if path.suffix.lower() == ".csv":
            import csv

            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                rows = [row[:max_cols] for _, row in zip(range(max_rows), csv.reader(handle))]
            return {"path": path.name, "sheet": None, "rows": rows}
        if path.suffix.lower() == ".xls":
            try:
                import xlrd
            except ImportError as exc:
                raise RuntimeError("XLS backend unavailable; install xlrd") from exc
            workbook = xlrd.open_workbook(str(path), on_demand=True)
            requested = str(arguments.get("sheet") or "").strip()
            sheet = workbook.sheet_by_name(requested) if requested else workbook.sheet_by_index(0)
            rows = [
                [sheet.cell_value(row, col) for col in range(min(sheet.ncols, max_cols))]
                for row in range(min(sheet.nrows, max_rows))
            ]
            return {"path": path.name, "sheet": sheet.name, "sheets": workbook.sheet_names(), "rows": rows}
        if path.suffix.lower() not in {".xlsx", ".xlsm"}:
            raise ValueError("spreadsheet tool supports CSV, XLS, XLSX, and XLSM")
        requested = str(arguments.get("sheet") or "").strip()
        try:
            import openpyxl
        except ImportError:
            sheet_name, sheet_names, rows = _read_xlsx_stdlib(
                path, requested, max_rows, max_cols
            )
        else:
            workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
            sheet = workbook[requested] if requested else workbook[workbook.sheetnames[0]]
            rows = [
                [cell for cell in row[:max_cols]]
                for _, row in zip(range(max_rows), sheet.iter_rows(values_only=True))
            ]
            sheet_name, sheet_names = sheet.title, workbook.sheetnames
        return {"path": path.name, "sheet": sheet_name, "sheets": sheet_names, "rows": rows}

    if name == "workspace_inspect_spreadsheet":
        path = workspace.resolve(str(arguments.get("path") or ""))
        max_rows = _int_arg(arguments, "max_rows", 100, 1, 500)
        max_cols = _int_arg(arguments, "max_cols", 30, 1, 100)
        requested = str(arguments.get("sheet") or "").strip()
        extension = path.suffix.lower()
        if extension == ".xls":
            try:
                import xlrd
            except ImportError as exc:
                raise RuntimeError("XLS backend unavailable; install xlrd") from exc
            workbook = xlrd.open_workbook(str(path), formatting_info=True, on_demand=True)
            sheet = workbook.sheet_by_name(requested) if requested else workbook.sheet_by_index(0)
            cells = []
            for row in range(min(sheet.nrows, max_rows)):
                for col in range(min(sheet.ncols, max_cols)):
                    cell = sheet.cell(row, col)
                    xf = workbook.xf_list[cell.xf_index]
                    background = workbook.colour_map.get(xf.background.pattern_colour_index)
                    cells.append({"row": row + 1, "column": col + 1, "value": cell.value, "fill_rgb": background})
            return {"path": path.name, "sheet": sheet.name, "sheets": workbook.sheet_names(), "cells": cells}
        if extension not in {".xlsx", ".xlsm"}:
            raise ValueError("styled spreadsheet tool supports XLS, XLSX, and XLSM")
        import openpyxl
        workbook = openpyxl.load_workbook(path, read_only=False, data_only=False)
        sheet = workbook[requested] if requested else workbook[workbook.sheetnames[0]]

        def color_value(color: Any) -> str | None:
            if color is None or not getattr(color, "type", None):
                return None
            value = getattr(color, color.type, None)
            return str(value) if value is not None else None

        cells = []
        for row in sheet.iter_rows(max_row=min(sheet.max_row, max_rows), max_col=min(sheet.max_column, max_cols)):
            for cell in row:
                cells.append({
                    "coordinate": cell.coordinate,
                    "value": cell.value,
                    "data_type": cell.data_type,
                    "fill_type": cell.fill.fill_type,
                    "fill_fg": color_value(cell.fill.fgColor),
                    "fill_bg": color_value(cell.fill.bgColor),
                    "font_color": color_value(cell.font.color),
                })
        return {"path": path.name, "sheet": sheet.title, "sheets": workbook.sheetnames, "cells": cells}

    if name == "workspace_analyze_spreadsheet_grid":
        path = workspace.resolve(str(arguments.get("path") or ""))
        if path.suffix.lower() not in {".xlsx", ".xlsm"}:
            raise ValueError("grid analysis currently supports XLSX and XLSM")
        requested = str(arguments.get("sheet") or "").strip()
        requested_color = str(arguments.get("fill_rgb") or "").strip().upper().lstrip("#")
        if len(requested_color) == 6:
            requested_color = "FF" + requested_color
        if not re.fullmatch(r"[0-9A-F]{8}", requested_color):
            raise ValueError("fill_rgb must be a 6- or 8-digit RGB/ARGB hex color")
        import openpyxl
        workbook = openpyxl.load_workbook(path, read_only=False, data_only=True)
        sheet = workbook[requested] if requested else workbook[workbook.sheetnames[0]]
        selected: set[tuple[int, int]] = set()
        for row in sheet.iter_rows():
            for cell in row:
                color = cell.fill.fgColor
                if color.type == "rgb" and str(color.rgb).upper()[-6:] == requested_color[-6:]:
                    selected.add((cell.row, cell.column))
        adjacency = {
            cell: [
                neighbor for neighbor in (
                    (cell[0] - 1, cell[1]), (cell[0] + 1, cell[1]),
                    (cell[0], cell[1] - 1), (cell[0], cell[1] + 1),
                ) if neighbor in selected
            ]
            for cell in selected
        }
        unseen = set(selected)
        component_sizes = []
        while unseen:
            start = unseen.pop()
            stack, size = [start], 0
            while stack:
                current = stack.pop(); size += 1
                for neighbor in adjacency[current]:
                    if neighbor in unseen:
                        unseen.remove(neighbor); stack.append(neighbor)
            component_sizes.append(size)
        partitions = [sum((row + col) % 2 == parity for row, col in selected) for parity in (0, 1)]
        degree_counts: dict[int, int] = {}
        for neighbors in adjacency.values():
            degree_counts[len(neighbors)] = degree_counts.get(len(neighbors), 0) + 1
        reasons = []
        if len(selected) % 2:
            reasons.append("odd vertex count in a bipartite grid")
        if len(component_sizes) != 1:
            reasons.append("selected cells are disconnected")
        if selected and min(map(len, adjacency.values())) < 2:
            reasons.append("at least one selected cell has degree below 2")
        if partitions[0] != partitions[1]:
            reasons.append("bipartition sizes are unequal")
        return {
            "path": path.name, "sheet": sheet.title, "fill_rgb": requested_color,
            "cell_count": len(selected), "coordinates": [sheet.cell(r, c).coordinate for r, c in sorted(selected)],
            "component_sizes": sorted(component_sizes, reverse=True),
            "degree_counts": degree_counts, "bipartition_sizes": partitions,
            "hamiltonian_cycle_ruled_out": bool(reasons),
            "necessary_condition_failures": reasons,
            "note": "Passing these necessary conditions does not prove that a Hamiltonian cycle exists.",
        }

    if name == "workspace_read_pdf":
        path = workspace.resolve(str(arguments.get("path") or ""))
        if path.suffix.lower() != ".pdf":
            raise ValueError("PDF tool requires a .pdf file")
        max_pages = _int_arg(arguments, "max_pages", 20, 1, 100)
        output = workspace.artifacts / f"{path.stem}.txt"
        result = subprocess.run(
            ["pdftotext", "-f", "1", "-l", str(max_pages), str(path), str(output)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"pdftotext exit {result.returncode}")
        return {"path": path.name, "text": _read_text(output, 30000)}

    if name == "workspace_list_archive":
        path = workspace.resolve(str(arguments.get("path") or ""))
        max_members = _int_arg(arguments, "max_members", 100, 1, 1000)
        with zipfile.ZipFile(path) as archive:
            members = []
            for info in archive.infolist()[:max_members]:
                members.append({
                    "name": info.filename,
                    "size_bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "is_directory": info.is_dir(),
                })
        return {"path": path.name, "members": members}

    if name == "workspace_read_archive_member":
        path = workspace.resolve(str(arguments.get("path") or ""))
        member = str(arguments.get("member") or "").strip()
        if not member:
            raise ValueError("member is required")
        member_path = Path(member)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError("unsafe archive member path")
        max_chars = _int_arg(arguments, "max_chars", 20000, 1, 50000)
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo(member)
            if info.is_dir():
                raise ValueError("archive member is a directory")
            if info.file_size > 5 * 1024 * 1024:
                raise ValueError("archive member exceeds 5 MiB")
            content = archive.read(info).decode("utf-8", errors="replace")
        return {"path": path.name, "member": member, "text": _clip(content, max_chars)}

    if name == "workspace_extract_archive_member":
        path = workspace.resolve(str(arguments.get("path") or ""))
        member = str(arguments.get("member") or "").strip()
        member_path = Path(member)
        if not member or member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError("unsafe archive member path")
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo(member)
            if info.is_dir():
                raise ValueError("archive member is a directory")
            if info.file_size > 25 * 1024 * 1024:
                raise ValueError("archive member exceeds 25 MiB")
            destination = workspace.artifacts / Path(member).name
            with archive.open(info) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        return {"path": str(destination.relative_to(workspace.root)), "member": member, "size_bytes": destination.stat().st_size}

    if name == "workspace_read_json":
        path = workspace.resolve(str(arguments.get("path") or ""))
        if path.suffix.lower() not in {".json", ".jsonld", ".geojson"}:
            raise ValueError("JSON tool requires JSON, JSON-LD, or GeoJSON")
        max_chars = _int_arg(arguments, "max_chars", 30000, 1, 50000)
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return {"path": path.name, "type": type(value).__name__, "content": _clip(json.dumps(value, ensure_ascii=False), max_chars)}

    if name == "workspace_analyze_pdb":
        path = workspace.resolve(str(arguments.get("path") or ""))
        if path.suffix.lower() not in {".pdb", ".ent"}:
            raise ValueError("PDB tool requires a .pdb or .ent file")
        max_residues = _int_arg(arguments, "max_residues", 5000, 1, 20000)
        amino = {
            "ALA":"A", "ARG":"R", "ASN":"N", "ASP":"D", "CYS":"C", "GLN":"Q", "GLU":"E",
            "GLY":"G", "HIS":"H", "ILE":"I", "LEU":"L", "LYS":"K", "MET":"M", "PHE":"F",
            "PRO":"P", "SER":"S", "THR":"T", "TRP":"W", "TYR":"Y", "VAL":"V", "SEC":"U",
        }
        chains: dict[str, list[tuple[str, str]]] = {}
        atoms = 0
        first_atoms: list[dict[str, Any]] = []
        hetero: set[tuple[str, str, str]] = set()
        seen: set[tuple[str, str, str]] = set()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            record = line[:6].strip()
            if record not in {"ATOM", "HETATM"}:
                continue
            atoms += 1
            residue, chain = line[17:20].strip(), line[21:22].strip() or "_"
            residue_id = line[22:27].strip()
            if len(first_atoms) < 20:
                try:
                    coordinates = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                except ValueError:
                    coordinates = []
                first_atoms.append({
                    "serial": line[6:11].strip(), "name": line[12:16].strip(),
                    "residue": residue, "chain": chain, "residue_id": residue_id,
                    "coordinates_angstrom": coordinates,
                })
            key = (chain, residue_id, residue)
            if record == "HETATM":
                hetero.add(key)
            elif key not in seen and sum(len(items) for items in chains.values()) < max_residues:
                seen.add(key)
                chains.setdefault(chain, []).append((residue_id, residue))
        first_distance = None
        if len(first_atoms) >= 2 and first_atoms[0]["coordinates_angstrom"] and first_atoms[1]["coordinates_angstrom"]:
            first_distance = sum(
                (left - right) ** 2
                for left, right in zip(first_atoms[0]["coordinates_angstrom"], first_atoms[1]["coordinates_angstrom"])
            ) ** 0.5
        return {
            "path": path.name, "atom_count": atoms,
            "first_atoms": first_atoms,
            "first_atom_distance_angstrom": first_distance,
            "first_atom_distance_angstrom_rounded_to_picometer": (
                round(first_distance, 3) if first_distance is not None else None
            ),
            "chains": {
                chain: {"residue_count": len(items), "sequence": "".join(amino.get(name, "X") for _, name in items),
                        "first_residue": items[0][0] if items else None, "last_residue": items[-1][0] if items else None}
                for chain, items in chains.items()
            },
            "hetero_residues": [{"chain": c, "residue_id": i, "name": n} for c, i, n in sorted(hetero)],
        }

    if name == "workspace_ocr_image":
        path = workspace.resolve(str(arguments.get("path") or ""))
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError("OCR tool requires an image file")
        if path.stat().st_size > 50 * 1024 * 1024:
            raise ValueError("OCR image exceeds 50 MiB")
        from PIL import Image
        with Image.open(path) as image:
            if image.width * image.height > 50_000_000:
                raise ValueError("OCR image exceeds 50 megapixels")
        max_items = _int_arg(arguments, "max_items", 200, 1, 1000)
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "OCR backend unavailable; install rapidocr-onnxruntime"
            ) from exc
        global _RAPID_OCR_ENGINE
        if _RAPID_OCR_ENGINE is None:
            _RAPID_OCR_ENGINE = RapidOCR()
        raw, elapsed = _RAPID_OCR_ENGINE(str(path))
        items = []
        for entry in (raw or [])[:max_items]:
            box, text, confidence = entry
            items.append({
                "text": str(text),
                "confidence": float(confidence),
                "box": [[float(value) for value in point] for point in box],
            })
        elapsed_values = (
            [float(value) for value in elapsed if isinstance(value, (int, float))]
            if isinstance(elapsed, (list, tuple))
            else [float(elapsed)] if isinstance(elapsed, (int, float)) else []
        )
        return {
            "path": path.name,
            "text": "\n".join(item["text"] for item in items),
            "items": items,
            "elapsed_seconds": sum(elapsed_values),
            "stage_elapsed_seconds": elapsed_values,
        }

    if name == "workspace_probe_media":
        path = workspace.resolve(str(arguments.get("path") or ""))
        if path.suffix.lower() not in MEDIA_EXTENSIONS:
            raise ValueError("media probe requires an image, audio, or video file")
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("media backend unavailable; install av") from exc
        with av.open(str(path)) as container:
            streams = []
            for stream in container.streams:
                item: dict[str, Any] = {
                    "index": stream.index,
                    "type": stream.type,
                    "codec": stream.codec_context.name,
                    "duration_seconds": (
                        float(stream.duration * stream.time_base)
                        if stream.duration is not None and stream.time_base is not None else None
                    ),
                    "metadata": _jsonable_metadata(dict(stream.metadata)),
                }
                if stream.type == "video":
                    item.update({
                        "width": stream.codec_context.width,
                        "height": stream.codec_context.height,
                        "frames": stream.frames or None,
                        "average_rate": str(stream.average_rate) if stream.average_rate else None,
                    })
                elif stream.type == "audio":
                    item.update({
                        "sample_rate": stream.codec_context.sample_rate,
                        "channels": stream.codec_context.channels,
                    })
                streams.append(item)
            duration = float(container.duration / av.time_base) if container.duration else None
            metadata = _jsonable_metadata(dict(container.metadata))
        return {"path": path.name, "duration_seconds": duration, "streams": streams, "metadata": metadata}

    if name == "workspace_extract_video_frames":
        path = workspace.resolve(str(arguments.get("path") or ""))
        if path.suffix.lower() not in {".mp4", ".mkv", ".mov", ".webm", ".avi", ".mpeg", ".mpg"}:
            raise ValueError("frame extraction requires a video file")
        if path.stat().st_size > 500 * 1024 * 1024:
            raise ValueError("video exceeds 500 MiB")
        interval = arguments.get("interval_seconds", 10.0)
        if isinstance(interval, bool) or not isinstance(interval, (int, float)):
            raise ValueError("interval_seconds must be numeric")
        interval = max(0.1, min(3600.0, float(interval)))
        max_frames = _int_arg(arguments, "max_frames", 12, 1, 60)
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError("FFmpeg backend unavailable; install imageio-ffmpeg") from exc
        prefix = re.sub(r"[^A-Za-z0-9_.-]", "_", path.stem)[:80]
        output_pattern = workspace.artifacts / f"{prefix}_frame_%03d.jpg"
        result = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(path),
             "-vf", f"fps=1/{interval}", "-frames:v", str(max_frames), "-q:v", "2", str(output_pattern)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=120, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"ffmpeg exit {result.returncode}")
        frames = sorted(workspace.artifacts.glob(f"{prefix}_frame_*.jpg"))
        return {
            "path": path.name,
            "interval_seconds": interval,
            "frames": [str(frame.relative_to(workspace.root)) for frame in frames],
        }

    if name == "workspace_transcribe_audio":
        path = workspace.resolve(str(arguments.get("path") or ""))
        if path.suffix.lower() not in MEDIA_EXTENSIONS - IMAGE_EXTENSIONS:
            raise ValueError("transcription requires an audio or video file")
        if path.stat().st_size > 500 * 1024 * 1024:
            raise ValueError("media file exceeds 500 MiB")
        model_name = str(arguments.get("model") or "tiny.en").strip()
        if model_name not in {"tiny", "tiny.en", "base", "base.en", "small", "small.en"}:
            raise ValueError("model must be tiny, tiny.en, base, base.en, small, or small.en")
        language = str(arguments.get("language") or "").strip() or None
        max_segments = _int_arg(arguments, "max_segments", 200, 20, 2000)
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("ASR backend unavailable; install faster-whisper") from exc
        device, compute_type = "cpu", "int8"
        key = (model_name, device, compute_type)
        model = _WHISPER_MODELS.get(key)
        if model is None:
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
            _WHISPER_MODELS[key] = model
        generated, info = model.transcribe(
            str(path), language=language, beam_size=5, vad_filter=True
        )
        segments = []
        for segment in generated:
            segments.append({
                "start": float(segment.start), "end": float(segment.end),
                "text": segment.text.strip(),
            })
            if len(segments) >= max_segments:
                break
        return {
            "path": path.name,
            "language": info.language,
            "language_probability": float(info.language_probability),
            "duration_seconds": float(info.duration),
            "text": " ".join(item["text"] for item in segments),
            "segments": segments,
        }

    if name == "python_calculate":
        code = str(arguments.get("code") or "")
        if not code.strip():
            raise ValueError("code is required")
        if len(code) > 12000:
            raise ValueError("code exceeds 12000 characters")
        tree = ast.parse(code, mode="exec")
        banned_names = {"open", "exec", "eval", "compile", "__import__", "input"}
        banned_modules = {"os", "sys", "subprocess", "socket", "pathlib", "shutil", "ctypes"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [alias.name.split(".")[0] for alias in node.names]
                if isinstance(node, ast.ImportFrom) and node.module:
                    modules.append(node.module.split(".")[0])
                if any(module in banned_modules for module in modules):
                    raise ValueError("restricted module in Python calculation")
            if isinstance(node, ast.Name) and node.id in banned_names:
                raise ValueError(f"restricted Python name: {node.id}")
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise ValueError("dunder attribute access is restricted")
        timeout = _int_arg(arguments, "timeout", 10, 1, 30)
        result = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=workspace.root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "exit_code": result.returncode,
            "stdout": _clip(result.stdout, 12000),
            "stderr": _clip(result.stderr, 4000),
        }

    if name == "web_search":
        import requests

        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        max_results = _int_arg(arguments, "max_results", 5, 1, 10)
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MAS-DAG-GAIA/1.0)"}
        provider = "duckduckgo"
        try:
            response = requests.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query}, headers=headers, timeout=12,
            )
            response.raise_for_status()
            parser: Any = _DuckDuckGoParser()
            parser.feed(response.text)
            if not parser.results:
                raise RuntimeError("DuckDuckGo returned no parsed results")
        except Exception:
            provider = "bing"
            response = requests.get(
                "https://www.bing.com/search",
                params={"q": query}, headers=headers, timeout=20,
            )
            response.raise_for_status()
            parser = _BingParser()
            parser.feed(response.text)
        results = []
        for item in parser.results:
            href = item["url"]
            parsed = urlparse(href)
            if parsed.netloc.endswith("duckduckgo.com"):
                href = unquote(parse_qs(parsed.query).get("uddg", [href])[0])
            results.append({"title": item["title"], "url": href})
            if len(results) >= max_results:
                break
        return {"query": query, "provider": provider, "results": results}

    if name == "web_open":
        import requests

        url = str(arguments.get("url") or "").strip()
        _safe_public_url(url)
        max_chars = _int_arg(arguments, "max_chars", 16000, 1, 50000)
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MAS-DAG-GAIA/1.0)"},
            timeout=25,
        )
        response.raise_for_status()
        _safe_public_url(response.url)
        content_type = response.headers.get("content-type", "")
        if "pdf" in content_type.lower():
            raise ValueError("web_open cannot parse remote PDF; download support is not enabled")
        parser = _ReadableTextParser()
        parser.feed(response.text)
        text = parser.text()
        return {"url": response.url, "status": response.status_code, "text": _clip(text, max_chars)}

    if name == "web_download":
        import requests

        url = str(arguments.get("url") or "").strip()
        _safe_public_url(url)
        max_bytes = _int_arg(arguments, "max_bytes", 10 * 1024 * 1024, 1, 25 * 1024 * 1024)
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MAS-DAG-GAIA/1.0)"},
            timeout=30,
            stream=True,
        )
        response.raise_for_status()
        _safe_public_url(response.url)
        filename = str(arguments.get("filename") or "").strip()
        if not filename:
            filename = Path(urlparse(response.url).path).name or "downloaded_file"
        filename = Path(filename).name
        if not filename:
            raise ValueError("invalid download filename")
        destination = workspace.artifacts / filename
        size = 0
        with destination.open("wb") as handle:
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > max_bytes:
                    destination.unlink(missing_ok=True)
                    raise ValueError(f"download exceeds {max_bytes} bytes")
                handle.write(chunk)
        return {
            "url": response.url,
            "path": str(destination.relative_to(workspace.root)),
            "size_bytes": size,
            "content_type": response.headers.get("content-type", ""),
        }

    if name == "web_find":
        text = str(arguments.get("text") or "")
        pattern = str(arguments.get("pattern") or "").strip()
        if not pattern:
            raise ValueError("pattern is required")
        max_matches = _int_arg(arguments, "max_matches", 20, 1, 100)
        matches = [line for line in text.splitlines() if pattern.casefold() in line.casefold()]
        return {"pattern": pattern, "matches": matches[:max_matches]}

    if name == "wikipedia_search":
        import requests

        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        max_results = _int_arg(arguments, "max_results", 5, 1, 10)
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search", "srsearch": query,
                "srlimit": max_results, "format": "json", "utf8": 1,
            },
            headers={"User-Agent": "MAS-DAG-GAIA/1.0"},
            timeout=20,
        )
        response.raise_for_status()
        results = [
            {
                "title": item["title"],
                "snippet": re.sub(r"<[^>]+>", "", item.get("snippet", "")),
                "url": "https://en.wikipedia.org/wiki/" + item["title"].replace(" ", "_"),
            }
            for item in response.json().get("query", {}).get("search", [])
        ]
        return {"query": query, "results": results}

    if name == "arxiv_search":
        import requests

        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        max_results = _int_arg(arguments, "max_results", 5, 1, 10)
        response = requests.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "start": 0, "max_results": max_results},
            headers={"User-Agent": "MAS-DAG-GAIA/1.0"},
            timeout=25,
        )
        response.raise_for_status()
        atom = "http://www.w3.org/2005/Atom"
        root = ET.fromstring(response.content)
        results = []
        for entry in root.findall(f"{{{atom}}}entry"):
            def value(tag: str) -> str:
                node = entry.find(f"{{{atom}}}{tag}")
                return re.sub(r"\s+", " ", node.text or "").strip() if node is not None else ""
            results.append({
                "title": value("title"), "summary": _clip(value("summary"), 1200),
                "published": value("published"), "url": value("id"),
            })
        return {"query": query, "results": results}

    raise ValueError(f"unknown tool: {name}")


def _tool_prompt(tools: Sequence[str], max_steps: int) -> str:
    available = "\n".join(
        f"- {name}: {TOOL_DESCRIPTIONS[name]}" for name in tools
    )
    return (
        "\n\nYou can use the following tools:\n"
        f"{available}\n"
        "To call one tool, output exactly one line beginning with TOOL_CALL: followed by "
        'a JSON object such as {"name":"web_search","arguments":{"query":"..."}}. '
        "After receiving TOOL_RESULT, reason again and call another tool if needed. "
        "When finished, output NODE_RESULT: followed by concise evidence for downstream agents. "
        f"You may make at most {max_steps} tool calls. Never claim a tool result you did not observe."
    )


def _parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    marker = re.search(r"TOOL_CALL\s*:\s*", text, re.IGNORECASE)
    if not marker:
        return None
    tail = text[marker.end():].lstrip()
    try:
        payload, _ = json.JSONDecoder().raw_decode(tail)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid TOOL_CALL JSON: {exc.msg}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
        raise ValueError("TOOL_CALL must contain a string name")
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("TOOL_CALL arguments must be an object")
    return payload["name"], arguments


def _final_text(text: str) -> str:
    match = re.search(r"NODE_RESULT\s*:\s*", text, re.IGNORECASE)
    return text[match.end():].strip() if match else text.strip()


def _prepare_messages(
    messages: Sequence[dict[str, str]], tools: Sequence[str], max_steps: int
) -> list[dict[str, str]]:
    unknown = [name for name in tools if name not in TOOL_DESCRIPTIONS]
    if unknown:
        raise ValueError(f"unknown node tools: {unknown}")
    prepared = [dict(message) for message in messages]
    prepared[0]["content"] += _tool_prompt(tools, max_steps)
    return prepared


def run_tool_loop_sync(
    backend: Any,
    messages: Sequence[dict[str, str]],
    tools: Sequence[str],
    workspace: EpisodeWorkspace,
    *,
    max_steps: int,
    max_new_tokens: int | None,
) -> ToolLoopResult:
    conversation = _prepare_messages(messages, tools, max_steps)
    trace: list[dict[str, Any]] = []
    total_input = total_output = 0
    total_latency = 0.0
    finish_reason = None
    repeated: dict[str, int] = {}
    for step in range(max_steps + 1):
        result = backend.generate(conversation, max_new_tokens=max_new_tokens)
        total_input += result.input_tokens
        total_output += result.output_tokens
        total_latency += result.latency_seconds
        finish_reason = result.finish_reason
        try:
            call = _parse_tool_call(result.text)
        except ValueError as exc:
            conversation.extend([
                {"role": "assistant", "content": result.text},
                {"role": "user", "content": f"TOOL_RESULT: {json.dumps({'status': 'error', 'error': str(exc)})}"},
            ])
            trace.append({"step": step + 1, "status": "parse_error", "error": str(exc)})
            continue
        if call is None:
            return ToolLoopResult(
                _final_text(result.text), total_input, total_output, total_latency,
                finish_reason, tuple(trace),
            )
        name, arguments = call
        entry: dict[str, Any] = {"step": step + 1, "name": name, "arguments": arguments}
        signature = json.dumps({"name": name, "arguments": arguments}, sort_keys=True)
        repeated[signature] = repeated.get(signature, 0) + 1
        if step >= max_steps:
            observation = {"status": "error", "error": "tool call limit reached"}
        elif name not in tools:
            observation = {"status": "error", "error": f"tool not allowed: {name}"}
        elif repeated[signature] > 2:
            observation = {"status": "error", "error": "identical tool call repeated too often"}
        else:
            started = time.perf_counter()
            try:
                observation = {"status": "ok", "result": execute_tool(name, arguments, workspace)}
            except Exception as exc:
                observation = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            entry["tool_latency_seconds"] = time.perf_counter() - started
        entry.update(observation)
        trace.append(entry)
        conversation.extend([
            {"role": "assistant", "content": result.text},
            {"role": "user", "content": "TOOL_RESULT: " + _clip(json.dumps(observation, ensure_ascii=False))},
        ])
    return ToolLoopResult(
        "Tool loop ended without a node result.", total_input, total_output,
        total_latency, finish_reason or "tool_limit", tuple(trace),
    )


async def run_tool_loop_async(
    backend: Any,
    messages: Sequence[dict[str, str]],
    tools: Sequence[str],
    workspace: EpisodeWorkspace,
    *,
    max_steps: int,
    max_new_tokens: int | None,
) -> ToolLoopResult:
    conversation = _prepare_messages(messages, tools, max_steps)
    trace: list[dict[str, Any]] = []
    total_input = total_output = 0
    total_latency = 0.0
    finish_reason = None
    repeated: dict[str, int] = {}
    for step in range(max_steps + 1):
        result = await backend.generate(conversation, max_new_tokens=max_new_tokens)
        total_input += result.input_tokens
        total_output += result.output_tokens
        total_latency += result.latency_seconds
        finish_reason = result.finish_reason
        try:
            call = _parse_tool_call(result.text)
        except ValueError as exc:
            conversation.extend([
                {"role": "assistant", "content": result.text},
                {"role": "user", "content": f"TOOL_RESULT: {json.dumps({'status': 'error', 'error': str(exc)})}"},
            ])
            trace.append({"step": step + 1, "status": "parse_error", "error": str(exc)})
            continue
        if call is None:
            return ToolLoopResult(
                _final_text(result.text), total_input, total_output, total_latency,
                finish_reason, tuple(trace),
            )
        name, arguments = call
        entry: dict[str, Any] = {"step": step + 1, "name": name, "arguments": arguments}
        signature = json.dumps({"name": name, "arguments": arguments}, sort_keys=True)
        repeated[signature] = repeated.get(signature, 0) + 1
        if step >= max_steps:
            observation = {"status": "error", "error": "tool call limit reached"}
        elif name not in tools:
            observation = {"status": "error", "error": f"tool not allowed: {name}"}
        elif repeated[signature] > 2:
            observation = {"status": "error", "error": "identical tool call repeated too often"}
        else:
            started = time.perf_counter()
            try:
                value = await asyncio.to_thread(execute_tool, name, arguments, workspace)
                observation = {"status": "ok", "result": value}
            except Exception as exc:
                observation = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            entry["tool_latency_seconds"] = time.perf_counter() - started
        entry.update(observation)
        trace.append(entry)
        conversation.extend([
            {"role": "assistant", "content": result.text},
            {"role": "user", "content": "TOOL_RESULT: " + _clip(json.dumps(observation, ensure_ascii=False))},
        ])
    return ToolLoopResult(
        "Tool loop ended without a node result.", total_input, total_output,
        total_latency, finish_reason or "tool_limit", tuple(trace),
    )
