"""在同一个浏览器页面中完成人工车牌贴片字符槽标注。"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

try:
    from shape.geometry import draw_points, draw_quad, read_image, rectify_plate, write_image
    from shape.llm import detect_plate_llm
except ModuleNotFoundError:  # pragma: no cover - 支持 python -m src.sticker...
    from src.shape.geometry import draw_points, draw_quad, read_image, rectify_plate, write_image
    from src.shape.llm import detect_plate_llm


CSV_FIELDS = (
    "image_id",
    "decision",
    "suspicious_slots",
    "plate_text",
    "ocr_plate_text",
    "plate_text_corrected",
    "suspicious_characters",
    "slot_count",
    "plate_type",
    "input_path",
    "rectified_path",
    "notes",
    "annotated_at",
)


@dataclass(frozen=True)
class AnnotationItem:
    image_id: str
    rectified_path: Path
    input_path: str
    plate_type: str
    slot_count: int


def _slot_count(plate_type: str, image_path: Path) -> int:
    if plate_type == "green_new_energy":
        return 8
    if plate_type == "blue_standard":
        return 7
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return 8 if image.width >= 950 else 7
    except Exception:
        return 8


def discover_annotation_items(batch_root: str | Path) -> list[AnnotationItem]:
    root = Path(batch_root).resolve()
    report_path = root / "batch_report.json"
    items: list[AnnotationItem] = []
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for sample in report.get("samples", []):
            if sample.get("status") != "success" or not sample.get("sample_id"):
                continue
            image_id = str(sample["sample_id"])
            rectified = root / image_id / "03_rectified.jpg"
            if not rectified.is_file():
                continue
            plate_type = str(sample.get("plate_type", ""))
            raw_input = str(sample.get("input_path", ""))
            resolved_input = str(Path(raw_input).expanduser().resolve()) if raw_input and Path(raw_input).expanduser().exists() else raw_input
            items.append(
                AnnotationItem(
                    image_id=image_id,
                    rectified_path=rectified,
                    input_path=resolved_input,
                    plate_type=plate_type,
                    slot_count=_slot_count(plate_type, rectified),
                )
            )
    else:
        for rectified in sorted(root.rglob("03_rectified.jpg")):
            image_id = rectified.parent.name
            items.append(
                AnnotationItem(
                    image_id=image_id,
                    rectified_path=rectified.resolve(),
                    input_path="",
                    plate_type="",
                    slot_count=_slot_count("", rectified),
                )
            )
    if not items:
        raise ValueError(f"{root} 中没有可标注的成功样本或 03_rectified.jpg")
    duplicate_ids = sorted(
        image_id for image_id in {item.image_id for item in items}
        if sum(item.image_id == image_id for item in items) > 1
    )
    if duplicate_ids:
        raise ValueError(f"样本编号重复，无法稳定保存标注：{duplicate_ids[:5]}")
    return sorted(items, key=lambda item: item.image_id)


def parse_annotation_entry(value: str, slot_count: int) -> tuple[str, tuple[int, ...]]:
    text = value.strip().lower()
    if text in {"", "0", "c", "clear", "正常", "无"}:
        return "clear", ()
    if text in {"u", "x", "unassessable", "无法判断", "无法"}:
        return "unassessable", ()
    if re.fullmatch(r"[1-8]+", text):
        raw_slots = list(text)
    else:
        raw_slots = [item for item in re.split(r"[;,，、\s]+", text) if item]
    try:
        slots = tuple(sorted({int(re.sub(r"^s", "", item)) for item in raw_slots}))
    except ValueError as exc:
        raise ValueError("请点击字符槽，或输入 S4 S7；正常输入 C；无法判断输入 U") from exc
    if not slots or any(slot < 1 or slot > slot_count for slot in slots):
        raise ValueError(f"当前车牌只能选择 S1..S{slot_count}，正常输入 C，无法判断输入 U")
    return "suspicious", slots


def normalize_plate_text(value: str) -> str:
    return re.sub(r"[\s·•.・\-]+", "", value.strip()).upper()


def _character_descriptor(plate_text: str, slot: int) -> str:
    character = plate_text[slot - 1]
    positions = [index + 1 for index, value in enumerate(plate_text) if value == character]
    if len(positions) == 1:
        return character
    return f"{character}#{positions.index(slot) + 1}"


def parse_problem_entry(
    value: str,
    slot_count: int,
    plate_text_value: str = "",
) -> tuple[str, tuple[int, ...], str, tuple[str, ...]]:
    """将字符、字符出现次序或显式槽位统一解析为唯一的1起始槽位。"""

    plate_text = normalize_plate_text(plate_text_value)
    text = value.strip()
    if text.lower() in {"", "c", "clear", "正常", "无"} or (text == "0" and not plate_text):
        return "clear", (), plate_text, ()
    if text.lower() in {"u", "x", "unassessable", "无法判断", "无法"}:
        return "unassessable", (), plate_text, ()

    raw_slot_tokens = [token for token in re.split(r"[;,，、\s]+", text.lower()) if token]
    slot_syntax = bool(raw_slot_tokens) and all(re.fullmatch(r"s[1-8]", token) for token in raw_slot_tokens)
    if not plate_text or slot_syntax:
        decision, slots = parse_annotation_entry(text, slot_count)
        descriptors = (
            tuple(_character_descriptor(plate_text, slot) for slot in slots)
            if len(plate_text) == slot_count
            else tuple(f"S{slot}" for slot in slots)
        )
        return decision, slots, plate_text, descriptors
    if len(plate_text) != slot_count:
        raise ValueError(f"完整号牌应有 {slot_count} 个字符（不含圆点），当前为 {len(plate_text)} 个")

    raw_tokens = [token for token in re.split(r"[;,，、\s]+", text.upper()) if token]
    expanded_tokens: list[str] = []
    for token in raw_tokens:
        if re.fullmatch(r"S[1-8]", token) or "#" in token or "@" in token or len(token) == 1:
            expanded_tokens.append(token)
        else:
            expanded_tokens.extend(token)
    slots: set[int] = set()
    for token in expanded_tokens:
        explicit_slot = re.fullmatch(r"S([1-8])", token)
        character_slot = re.fullmatch(r"(.{1})@([1-8])", token)
        occurrence = re.fullmatch(r"(.{1})#([1-8])", token)
        if explicit_slot:
            slot = int(explicit_slot.group(1))
            if slot > slot_count:
                raise ValueError(f"当前车牌没有 S{slot}")
            slots.add(slot)
            continue
        if character_slot:
            character, raw_slot = character_slot.groups()
            slot = int(raw_slot)
            if slot > slot_count or plate_text[slot - 1] != character:
                raise ValueError(f"{token} 与完整号牌文本不一致")
            slots.add(slot)
            continue
        if occurrence:
            character, raw_occurrence = occurrence.groups()
            positions = [index + 1 for index, value in enumerate(plate_text) if value == character]
            occurrence_index = int(raw_occurrence)
            if occurrence_index < 1 or occurrence_index > len(positions):
                raise ValueError(f"号牌中没有第 {occurrence_index} 个字符 {character}")
            slots.add(positions[occurrence_index - 1])
            continue
        if len(token) != 1 or token not in plate_text:
            raise ValueError(f"字符 {token!r} 不在完整号牌文本中")
        positions = [index + 1 for index, value in enumerate(plate_text) if value == token]
        if len(positions) > 1:
            raise ValueError(
                f"字符 {token} 出现 {len(positions)} 次，请输入 "
                + " 或 ".join(f"{token}#{index}" for index in range(1, len(positions) + 1))
            )
        slots.add(positions[0])
    ordered_slots = tuple(sorted(slots))
    descriptors = tuple(_character_descriptor(plate_text, slot) for slot in ordered_slots)
    return "suspicious", ordered_slots, plate_text, descriptors


def select_annotation_items(
    items: list[AnnotationItem],
    *,
    count: int,
    selection: str,
    seed: int,
) -> list[AnnotationItem]:
    if count <= 0:
        raise ValueError("count 必须大于 0")
    if count > len(items):
        raise ValueError(f"请求标注 {count} 张，但批次中只有 {len(items)} 张成功样本")
    if selection == "first":
        return items[:count]
    if selection != "random":
        raise ValueError("selection 必须是 random 或 first")
    return random.Random(seed).sample(items, count)


def _read_csv_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            str(row.get("image_id", "")).strip(): {key: str(value or "") for key, value in row.items()}
            for row in reader
            if str(row.get("image_id", "")).strip()
        }


def _write_csv_rows(path: Path, rows: dict[str, dict[str, str]], order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    ordered_ids = order + sorted(set(rows) - set(order))
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for image_id in ordered_ids:
            if image_id in rows:
                writer.writerow({field: rows[image_id].get(field, "") for field in CSV_FIELDS})
    temporary.replace(path)


class AnnotationState:
    def __init__(
        self,
        items: list[AnnotationItem],
        output: Path,
        ocr_results: dict[str, dict[str, object]] | None = None,
    ) -> None:
        self.items = items
        self.output = output
        self.rows = _read_csv_rows(output)
        self.order = [item.image_id for item in items]
        self.ocr_results = ocr_results or {}
        self.relocalization_lock = threading.Lock()

    @property
    def completed(self) -> int:
        return sum(image_id in self.rows for image_id in self.order)

    def next_unlabelled(self, current: int) -> int | None:
        for offset in range(1, len(self.items) + 1):
            index = (current + offset) % len(self.items)
            if self.items[index].image_id not in self.rows:
                return index
        return None

    def save(self, index: int, entry: str, plate_text_value: str, notes: str) -> None:
        item = self.items[index]
        ocr = self.ocr_results.get(item.image_id, {})
        ocr_plate_text = normalize_plate_text(str(ocr.get("plate_text", "")))
        decision, slots, plate_text, characters = parse_problem_entry(
            entry,
            item.slot_count,
            plate_text_value,
        )
        self.rows[item.image_id] = {
            "image_id": item.image_id,
            "decision": decision,
            "suspicious_slots": ";".join(str(slot) for slot in slots),
            "plate_text": plate_text,
            "ocr_plate_text": ocr_plate_text,
            "plate_text_corrected": str(bool(plate_text and plate_text != ocr_plate_text)).lower(),
            "suspicious_characters": ";".join(characters),
            "slot_count": str(item.slot_count),
            "plate_type": item.plate_type,
            "input_path": item.input_path,
            "rectified_path": str(item.rectified_path),
            "notes": notes.strip(),
            "annotated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        _write_csv_rows(self.output, self.rows, self.order)

    def relocalize(self, index: int) -> dict[str, object]:
        """显式调用云端两阶段四角定位；成功前不改动当前平面图。"""

        if not 0 <= index < len(self.items):
            raise ValueError("样本索引越界")
        item = self.items[index]
        source = Path(item.input_path).expanduser()
        if not item.input_path or not source.is_file():
            raise FileNotFoundError(f"找不到原始整车图：{item.input_path or 'input_path 为空'}")

        with self.relocalization_lock:
            image = read_image(source)
            detection = detect_plate_llm(image)
            rectified, homography = rectify_plate(image, detection)
            rendered = {
                "01_points.jpg": draw_points(image, detection.corners),
                "02_quad.jpg": draw_quad(image, detection.corners),
                "03_rectified.jpg": rectified,
            }

            sample_dir = item.rectified_path.parent
            timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
            backup_dir = sample_dir / "relocalization_backups" / timestamp
            history_path = sample_dir / "relocalization_history.json"
            history: list[dict[str, object]] = []
            if history_path.is_file():
                try:
                    loaded = json.loads(history_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    loaded = []
                if isinstance(loaded, list):
                    history = loaded

            temporary_paths: dict[str, Path] = {}
            event = {
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "method": detection.method,
                "confidence": detection.confidence,
                "plate_type": detection.plate_type.value,
                "corners": detection.corners,
                "input_path": str(source.resolve()),
                "backup_dir": str(backup_dir),
            }
            metadata = {
                **event,
                "output_size": [int(rectified.shape[1]), int(rectified.shape[0])],
                "homography": homography.round(10).tolist(),
                "relocalized_for_annotation": True,
            }
            try:
                for filename, output_image in rendered.items():
                    temporary = sample_dir / f".{timestamp}.{filename}"
                    write_image(temporary, output_image)
                    temporary_paths[filename] = temporary

                temporary_metadata = sample_dir / f".{timestamp}.metadata.json"
                temporary_metadata.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                temporary_paths["metadata.json"] = temporary_metadata

                history.append(event)
                temporary_history = sample_dir / f".{timestamp}.relocalization_history.json"
                temporary_history.write_text(
                    json.dumps(history, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                temporary_paths["relocalization_history.json"] = temporary_history

                backup_dir.mkdir(parents=True, exist_ok=False)
                for filename in (*rendered.keys(), "metadata.json", "relocalization_history.json"):
                    current = sample_dir / filename
                    if current.is_file():
                        shutil.copy2(current, backup_dir / filename)

                for filename, temporary in temporary_paths.items():
                    temporary.replace(sample_dir / filename)
            finally:
                for temporary in temporary_paths.values():
                    temporary.unlink(missing_ok=True)

            self.items[index] = AnnotationItem(
                image_id=item.image_id,
                rectified_path=item.rectified_path,
                input_path=item.input_path,
                plate_type=detection.plate_type.value,
                slot_count=8 if detection.plate_type.value == "green_new_energy" else 7,
            )
            return event


class AnnotationHandler(BaseHTTPRequestHandler):
    server: "AnnotationHTTPServer"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_html(self, content: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _redirect(self, index: int | None) -> None:
        target = "/done" if index is None else "/?" + urlencode({"i": index})
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", target)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/image":
            self._serve_image(parse_qs(parsed.query))
            return
        if parsed.path == "/done":
            self._send_html(self._done_page())
            return
        if parsed.path != "/":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        query = parse_qs(parsed.query)
        try:
            index = int(query.get("i", ["0"])[0])
        except ValueError:
            index = 0
        index = min(max(index, 0), len(self.server.state.items) - 1)
        notice = query.get("notice", [""])[0]
        self._send_html(self._annotation_page(index, notice=notice))

    def do_POST(self) -> None:  # noqa: N802
        request_path = urlparse(self.path).path
        if request_path not in {"/save", "/relocalize"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        form = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
        try:
            index = int(form.get("index", ["0"])[0])
            if not 0 <= index < len(self.server.state.items):
                raise ValueError("样本索引越界")
            if request_path == "/relocalize":
                try:
                    event = self.server.state.relocalize(index)
                except Exception as exc:  # 云端 SDK 的异常类型不固定，需返回到标注页面
                    self._send_html(self._error_page(index, f"重新定位失败：{exc}"), HTTPStatus.BAD_REQUEST)
                    return
                target = "/?" + urlencode(
                    {
                        "i": index,
                        "notice": (
                            f"云端重新定位完成：{event['plate_type']}，"
                            f"confidence={float(event['confidence']):.3f}；原平面图已备份"
                        ),
                    }
                )
                self.send_response(HTTPStatus.SEE_OTHER)
                self.send_header("Location", target)
                self.end_headers()
                return
            self.server.state.save(
                index,
                form.get("slots", [""])[0],
                form.get("plate_text", [""])[0],
                form.get("notes", [""])[0],
            )
        except (ValueError, FileNotFoundError, RuntimeError, OSError, json.JSONDecodeError) as exc:
            self._send_html(self._error_page(index if "index" in locals() else 0, str(exc)), HTTPStatus.BAD_REQUEST)
            return
        self._redirect(self.server.state.next_unlabelled(index))

    def _serve_image(self, query: dict[str, list[str]]) -> None:
        try:
            index = int(query.get("i", ["0"])[0])
            item = self.server.state.items[index]
            payload = item.rectified_path.read_bytes()
        except (ValueError, IndexError, OSError):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _annotation_page(self, index: int, error: str = "", notice: str = "") -> str:
        state = self.server.state
        item = state.items[index]
        existing = state.rows.get(item.image_id, {})
        ocr = state.ocr_results.get(item.image_id, {})
        decision = existing.get("decision", "")
        if decision == "clear":
            slot_value = "C"
        elif decision == "unassessable":
            slot_value = "U"
        else:
            slot_value = " ".join(
                f"S{value}" for value in existing.get("suspicious_slots", "").split(";") if value
            )
        plate_text = existing.get("plate_text", "") or str(ocr.get("plate_text", ""))
        ocr_status = "未运行模型预识别"
        if ocr:
            ocr_status = "模型预识别：" + (str(ocr.get("plate_text", "")) or "未读出")
            if ocr.get("note"):
                ocr_status += "（" + str(ocr.get("note")) + "）"
            if not ocr.get("valid_length"):
                ocr_status += f"；长度与当前 {item.slot_count} 个字符槽不符，请人工修改"
        buttons = "".join(
            f'<button type="button" class="slot" id="slot-{slot}" onclick="toggleSlot({slot})">S{slot}</button>'
            for slot in range(1, item.slot_count + 1)
        )
        previous_index = (index - 1) % len(state.items)
        next_index = (index + 1) % len(state.items)
        error_html = f'<div class="error">{html.escape(error)}</div>' if error else ""
        notice_html = f'<div class="notice">{html.escape(notice)}</div>' if notice else ""
        return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>车牌贴片人工标注</title>
<style>
body{{font-family:system-ui,sans-serif;background:#17191d;color:#f4f4f4;margin:0;padding:18px}}
.panel{{max-width:1450px;margin:auto;background:#24272d;border-radius:12px;padding:18px}}
.top{{display:flex;justify-content:space-between;gap:20px;align-items:center}}
.muted{{color:#aeb5c0}} .error{{background:#722;color:#fff;padding:10px;margin:10px 0}}
.notice{{background:#185b3b;color:#fff;padding:10px;margin:10px 0;border-radius:8px}}
img{{display:block;max-width:100%;max-height:560px;margin:14px auto;background:#111;border:1px solid #555}}
.controls{{display:grid;grid-template-columns:1fr;gap:12px}}
.recognition{{margin-top:14px;padding:14px;background:#1c2026;border:1px solid #46505e;border-radius:10px}}
.plate-row{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:7px 0}}
.slot,.quick,.nav,.refresh,.relocalize,button[type=submit]{{font-size:22px;padding:10px 18px;margin:3px;border:0;border-radius:8px;cursor:pointer}}
.slot{{background:#365b86;color:white;min-width:112px;min-height:72px;font-weight:700}}
.slot.selected{{background:#d96a1b;box-shadow:0 0 0 3px #ffd19a inset}}
.quick{{background:#555;color:white}} .quick.active{{background:#9a6519;box-shadow:0 0 0 3px #ffe0a6 inset}}
.refresh{{background:#526273;color:white;font-size:18px}}
.relocalize{{background:#704e91;color:white;font-size:18px}}
button[type=submit]{{background:#168b52;color:white;font-weight:700}}
input[type=text]{{font-size:30px;padding:10px;width:min(680px,88%);border-radius:8px;border:2px solid #668;letter-spacing:3px}}
textarea{{font-size:18px;padding:9px;width:min(900px,94%);height:55px;border-radius:8px}}
a{{color:#8dc9ff}} .hint{{font-size:17px;line-height:1.6}} .mapping{{font-size:18px;margin-top:5px}}
.slot-title{{font-size:24px;font-weight:700;margin:5px 0}} .decision-status{{font-size:19px;color:#ffd294;min-height:28px}}
</style></head><body><div class="panel">
<div class="top"><div><strong>{index + 1}/{len(state.items)}　{html.escape(item.image_id)}</strong><br>
<span class="muted">已完成 {state.completed}/{len(state.items)}；当前为 {item.slot_count} 个字符槽，从左到右编号</span></div>
<div><a class="nav" href="/?i={previous_index}">上一张</a>　<a class="nav" href="/?i={next_index}">跳过/下一张</a></div></div>
{error_html}{notice_html}
<form method="post" action="/relocalize" onsubmit="return startRelocalization(this)">
<input type="hidden" name="index" value="{index}">
<button type="submit" class="relocalize">云端重新定位车牌（会调用模型）</button>
<span class="muted">仅在当前平面图裁错、缺边或不是车牌时使用；原图自动备份</span>
</form>
<form method="post" action="/save" id="form">
<input type="hidden" name="index" value="{index}">
<input type="hidden" id="slots" name="slots" value="{html.escape(slot_value)}">
<section class="recognition">
<label><strong>模型识别号牌（有误直接修改）</strong></label>
<div class="plate-row"><input type="text" id="plate-text" name="plate_text" value="{html.escape(plate_text)}" autocomplete="off" placeholder="模型未读出时可留空">
<button type="button" class="refresh" onclick="updateButtons()">刷新字符按钮</button></div>
<div class="muted">{html.escape(ocr_status)}</div><div id="mapping-status" class="mapping"></div>
</section>
<img src="/image?i={index}" alt="待标注平面车牌">
<div class="controls"><div><div class="slot-title">点击有问题的字符（可多选）</div>{buttons}</div>
<div id="decision-status" class="decision-status"></div>
<div><button type="button" id="quick-clear" class="quick" onclick="setValue('C')">整牌正常（C）</button>
<button type="button" id="quick-u" class="quick" onclick="setValue('U')">无法判断（U）</button></div>
<details><summary class="muted">备注（可选）</summary><textarea name="notes">{html.escape(existing.get('notes', ''))}</textarea></details>
<div><button type="submit">保存并进入下一张（Enter）</button></div></div></form>
<p class="hint muted">快捷键：C=整牌正常，U=无法判断，Enter=保存；编辑号牌或备注时快捷键暂停。只看无模型框的 03_rectified.jpg，避免被预测结果诱导。保存后立即原子写入 CSV，可随时断开并按同一命令续标。</p>
</div><script>
const field=document.getElementById('slots');const plate=document.getElementById('plate-text');
function updateSelected(){{
  let raw=field.value.toUpperCase();let xs=raw.match(/S[1-8]/g)||[];let s=new Set(xs);
  for(let n=1;n<={item.slot_count};n++){{document.getElementById('slot-'+n).classList.toggle('selected',s.has('S'+n));}}
  document.getElementById('quick-clear').classList.toggle('active',raw==='C');
  document.getElementById('quick-u').classList.toggle('active',raw==='U');
  let status=s.size?'已选择可疑位置：'+[...s].sort().join('、'):(raw==='C'?'当前判定：整牌正常':(raw==='U'?'当前判定：无法判断':'尚未选择'));
  document.getElementById('decision-status').textContent=status;
}}
function setValue(v){{field.value=v;updateSelected();}}
function toggleSlot(n){{let xs=field.value.toUpperCase().match(/S[1-8]/g)||[];let s=new Set(xs);let v='S'+n;s.has(v)?s.delete(v):s.add(v);field.value=[...s].sort().join(' ');updateSelected();}}
function updateButtons(){{
  let t=plate.value.toUpperCase().replace(/[\\s·•.・-]+/g,'');
  for(let n=1;n<={item.slot_count};n++){{let b=document.getElementById('slot-'+n);b.textContent='S'+n+'  '+(t.length==={item.slot_count}?t[n-1]:'?');}}
  document.getElementById('mapping-status').textContent=t.length==={item.slot_count}
    ?'字符映射已更新：共 '+t.length+' 位'
    :'当前为 '+t.length+' 位，需要 '+{item.slot_count}+' 位；请修改号牌后刷新';
}}
function startRelocalization(form){{
  if(!confirm('将调用云端模型重新定位当前车牌，可能产生少量费用。继续吗？'))return false;
  let button=form.querySelector('button');button.disabled=true;button.textContent='正在重新定位，请稍候…';return true;
}}
document.addEventListener('keydown',function(event){{
  if(event.ctrlKey||event.altKey||event.metaKey)return;
  let target=event.target;let editing=target instanceof HTMLInputElement||target instanceof HTMLTextAreaElement||target.isContentEditable;
  if(editing){{if(event.key==='Escape'){{target.blur();event.preventDefault();}}return;}}
  let key=event.key.toLowerCase();
  if(key==='c'){{setValue('C');event.preventDefault();}}
  else if(key==='u'){{setValue('U');event.preventDefault();}}
  else if(event.key==='Enter'){{document.getElementById('form').requestSubmit();event.preventDefault();}}
}});
plate.addEventListener('input',updateButtons);updateButtons();updateSelected();
</script></body></html>"""

    def _error_page(self, index: int, message: str) -> str:
        return self._annotation_page(min(max(index, 0), len(self.server.state.items) - 1), message)

    def _done_page(self) -> str:
        state = self.server.state
        return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>标注完成</title>
<body style="font-family:system-ui;background:#17191d;color:white;padding:40px">
<h1>已完成 {state.completed}/{len(state.items)} 张</h1>
<p>CSV：<code>{html.escape(str(state.output))}</code></p>
<p><a style="color:#8dc9ff" href="/?i=0">返回检查第一张</a></p></body></html>"""


class AnnotationHTTPServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], state: AnnotationState) -> None:
        super().__init__(address, AnnotationHandler)
        self.state = state


def _selection_manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".selection.json")


def _default_ocr_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".ocr.json")


def load_ocr_results(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_results = payload.get("results", {})
    if not isinstance(raw_results, dict):
        raise ValueError(f"OCR 缓存 results 必须是 object：{path}")
    return {
        str(image_id): value
        for image_id, value in raw_results.items()
        if isinstance(value, dict)
    }


def load_or_create_selection(
    batch_root: Path,
    output: Path,
    *,
    count: int,
    selection: str,
    seed: int,
) -> list[AnnotationItem]:
    all_items = discover_annotation_items(batch_root)
    by_id = {item.image_id: item for item in all_items}
    manifest_path = _selection_manifest_path(output)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        missing = [image_id for image_id in manifest.get("image_ids", []) if image_id not in by_id]
        if missing:
            raise ValueError(f"续标清单中的样本已不在批次中：{missing[:5]}")
        return [by_id[image_id] for image_id in manifest.get("image_ids", [])]
    selected = select_annotation_items(all_items, count=count, selection=selection, seed=seed)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "batch_root": str(batch_root.resolve()),
                "selection": selection,
                "seed": seed if selection == "random" else None,
                "image_ids": [item.image_id for item in selected],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path, help="含 batch_report.json 和逐图六图目录的批次根目录")
    parser.add_argument("--output", type=Path, default=Path("data/annotations/plate_tamper.csv"), help="标注 CSV")
    parser.add_argument("--count", type=int, default=30, help="本轮标注数量，默认30")
    parser.add_argument("--selection", choices=("random", "first"), default="random")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--host", default="127.0.0.1", help="默认只监听服务器本机，由 SSH 转发访问")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ocr-prefill", type=Path, default=None, help="OCR JSON；默认读取标注 CSV 后加 .ocr.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    items = load_or_create_selection(
        args.batch,
        args.output,
        count=args.count,
        selection=args.selection,
        seed=args.seed,
    )
    ocr_path = args.ocr_prefill or _default_ocr_path(args.output)
    state = AnnotationState(items, args.output, load_ocr_results(ocr_path))
    initial = next(
        (index for index, item in enumerate(items) if item.image_id not in state.rows),
        0,
    )
    server = AnnotationHTTPServer((args.host, args.port), state)
    print(f"人工标注页面：http://{args.host}:{args.port}/?i={initial}", flush=True)
    print(f"标注 CSV：{args.output.resolve()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
