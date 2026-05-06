#!/usr/bin/env python3
"""Serve a lightweight frontend demo for the retrieval QA pipeline."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[1]
RETRIEVAL_DIR = ROOT_DIR / "scripts" / "retrieval"
STATIC_DIR = ROOT_DIR / "demo_ui"
INDEX_DIR = ROOT_DIR / "data" / "retrieval"
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
CHUNK_DIR = ROOT_DIR / "data" / "chunk"
EXTRACTION_DIR = ROOT_DIR / "scripts" / "extraction"

if str(RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_DIR))

from generate_answer import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_RETRIEVAL_METHOD,
    generate_answer,
    location,
    result_context_text,
    retrieve_results,
    select_context_results,
)


def slugify_course_id(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def available_courses(index_dir: Path) -> List[str]:
    course_ids: set[str] = set()
    for path in index_dir.glob("*_atomic.faiss"):
        course_ids.add(path.stem.removesuffix("_atomic"))
    for path in index_dir.glob("*_semantic.faiss"):
        course_ids.add(path.stem.removesuffix("_semantic"))
    return sorted(course_ids)


def load_course_name(course_id: str) -> str:
    candidate_paths = [
        PROCESSED_DIR / course_id / f"{course_id}_processed.json",
        PROCESSED_DIR / course_id / f"{course_id}_document.json",
        CHUNK_DIR / f"{course_id}_atomic_chunks.json",
        CHUNK_DIR / f"{course_id}_semantic_chunks.json",
    ]

    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        course_name = clean_text(payload.get("course_name"))
        if course_name:
            return course_name

    return course_id.upper()


def build_course_options(index_dir: Path) -> List[Dict[str, str]]:
    return [
        {
            "id": course_id,
            "name": load_course_name(course_id),
        }
        for course_id in available_courses(index_dir)
    ]


def default_course_id(courses: List[str]) -> str:
    if "eods" in courses:
        return "eods"
    return courses[0] if courses else ""


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def serve_file(handler: BaseHTTPRequestHandler, file_path: Path) -> None:
    if not file_path.exists() or not file_path.is_file():
        json_response(handler, HTTPStatus.NOT_FOUND, {"error": f"Not found: {file_path.name}"})
        return

    body = file_path.read_bytes()
    content_type, _ = mimetypes.guess_type(str(file_path))
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", (content_type or "application/octet-stream") + "; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def build_config(index_dir: Path) -> Dict[str, Any]:
    course_options = build_course_options(index_dir)
    courses = [course["id"] for course in course_options]
    return {
        "courses": courses,
        "courseOptions": course_options,
        "defaults": {
            "courseId": default_course_id(courses),
            "target": "both",
            "topK": 5,
            "contextK": 3,
            "candidateK": 4,
            "embeddingModel": DEFAULT_EMBEDDING_MODEL,
            "generationModel": DEFAULT_GENERATION_MODEL,
            "retrievalMethod": DEFAULT_RETRIEVAL_METHOD,
            "rrfK": 60,
            "faissWeight": 1.0,
            "bm25Weight": 1.0,
            "memoryWindow": 3,
            "densePoolMultiplier": 4,
            "denseRerankDenseWeight": 0.65,
            "denseRerankBm25Weight": 0.35,
        },
    }


def build_sources(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for index, result in enumerate(results, start=1):
        sources.append(
            {
                "label": f"S{index}",
                "location": location(result),
                "level": result.get("level"),
                "chunkType": result.get("chunk_type"),
                "preview": source_preview_text(result),
                "score": result.get("combined_score"),
            }
        )
    return sources


def source_preview_text(result: Dict[str, Any], max_length: int = 340) -> str:
    text = clean_text(result_context_text(result))
    if not text:
        return ""

    text = (
        text.replace("text_inline_math:", "")
        .replace("text:", "")
        .replace("formula:", "")
        .replace("figure:", "")
        .replace("auxiliary:", "")
    )
    text = re.sub(r"(Section:\s*[^.]+)\s+\1", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_length:
        return text

    clipped = text[: max_length - 1]
    last_space = clipped.rfind(" ")
    if last_space > max_length * 0.65:
        clipped = clipped[:last_space]
    return clipped.rstrip(" ,;:") + "..."


def normalize_turns(turns: Any) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    if not isinstance(turns, list):
        return normalized

    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = clean_text(turn.get("role")).lower()
        content = clean_text(turn.get("content"))
        if role in {"user", "assistant"} and content:
            normalized.append({"role": role, "content": content})
    return normalized


def is_generic_followup(query: str) -> bool:
    lowered = clean_text(query).lower()
    generic_phrases = {
        "example",
        "examples",
        "give example",
        "give examples",
        "give me example",
        "give me examples",
        "more examples",
        "why",
        "how",
        "explain more",
        "tell me more",
        "more",
        "what about that",
        "what about this",
    }
    return lowered in generic_phrases


def rewrite_generic_followup(query: str, current_topic: str) -> str:
    lowered = clean_text(query).lower()
    topic = clean_text(current_topic)
    if not topic:
        return query
    if "example" in lowered:
        return f"examples of {topic}"
    if lowered == "why":
        return f"why {topic}"
    if lowered == "how":
        return f"how {topic} works"
    if "explain" in lowered or "more" in lowered:
        return f"more explanation of {topic}"
    return f"{query} about {topic}"


def format_turns(turns: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for turn in turns:
        role = clean_text(turn.get("role")).title() or "Unknown"
        content = clean_text(turn.get("content"))
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def update_conversation_summary(
    *,
    client: OpenAI,
    model: str,
    existing_summary: str,
    turns_to_summarize: List[Dict[str, str]],
) -> str:
    if not turns_to_summarize:
        return clean_text(existing_summary)

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You maintain concise conversation memory for a course assistant. "
                    "Preserve the user's goals, what has been explained, open follow-ups, "
                    "and any topic references that still matter."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Update this running conversation summary.\n\n"
                    f"Existing summary:\n{existing_summary or 'None'}\n\n"
                    f"New turns to summarize:\n{format_turns(turns_to_summarize)}\n\n"
                    "Return only the updated summary in plain English, under 140 words."
                ),
            },
        ],
    )
    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise RuntimeError("OpenAI summary response did not include output_text")
    return clean_text(output_text)


def build_retrieval_query(
    *,
    client: OpenAI,
    model: str,
    current_query: str,
    conversation_summary: str,
    recent_turns: List[Dict[str, str]],
    current_topic: str,
) -> str:
    summary = clean_text(conversation_summary)
    topic = clean_text(current_topic)
    if is_generic_followup(current_query) and topic:
        return rewrite_generic_followup(current_query, topic)
    if not summary and not recent_turns:
        return current_query

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "Rewrite a follow-up student question into a standalone retrieval query "
                    "for searching lecture materials. Keep it concise and faithful. "
                    "When the current question is vague, resolve it to the most recent active topic, "
                    "giving strong priority to the latest user turn and latest assistant reply over older history."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation summary:\n{summary or 'None'}\n\n"
                    f"Recent turns:\n{format_turns(recent_turns) or 'None'}\n\n"
                    f"Current question:\n{current_query}\n\n"
                    "Rules:\n"
                    "- If the current question is generic, such as 'give examples', 'why', 'explain more', or 'what about that', bind it to the latest topic in the recent turns.\n"
                    "- Do not drift back to older topics unless the current question explicitly names them.\n"
                    f"- Current tracked topic: {topic or 'None'}.\n"
                    "- Return only the rewritten standalone retrieval query.\n"
                    "- Example: if the latest topic is statistical power and the user says 'give examples', return 'examples of statistical power in hypothesis testing'."
                ),
            },
        ],
    )
    output_text = getattr(response, "output_text", None)
    rewritten = clean_text(output_text)
    return rewritten or current_query


def infer_current_topic(
    *,
    client: OpenAI,
    model: str,
    previous_topic: str,
    user_query: str,
    assistant_answer: str,
) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "Extract the current topic of a tutoring conversation as a short noun phrase. "
                    "Prefer the most recent topic. Keep it under 12 words."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Previous topic: {previous_topic or 'None'}\n\n"
                    f"Latest user query: {user_query}\n\n"
                    f"Latest assistant answer: {assistant_answer}\n\n"
                    "Return only the current topic phrase."
                ),
            },
        ],
    )
    output_text = getattr(response, "output_text", None)
    topic = clean_text(output_text)
    return topic or clean_text(previous_topic)


def update_conversation_memory(
    *,
    client: OpenAI,
    model: str,
    existing_summary: str,
    recent_turns: List[Dict[str, str]],
    new_user_query: str,
    assistant_answer: str,
    memory_window: int,
) -> Dict[str, Any]:
    full_turns = recent_turns + [
        {"role": "user", "content": clean_text(new_user_query)},
        {"role": "assistant", "content": clean_text(assistant_answer)},
    ]
    max_recent_messages = max(2, memory_window * 2)

    if len(full_turns) <= max_recent_messages:
        return {
            "conversationSummary": clean_text(existing_summary),
            "recentTurns": full_turns,
        }

    turns_to_summarize = full_turns[:-max_recent_messages]
    updated_summary = update_conversation_summary(
        client=client,
        model=model,
        existing_summary=existing_summary,
        turns_to_summarize=turns_to_summarize,
    )
    return {
        "conversationSummary": updated_summary,
        "recentTurns": full_turns[-max_recent_messages:],
    }


def safe_filename(name: str) -> str:
    candidate = Path(name or "document.pdf").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate).strip("._")
    return cleaned or "document.pdf"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 10_000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create unique filename for {path.name}")


def decode_uploaded_files(items: Any) -> List[Dict[str, Any]]:
    decoded: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return decoded
    for item in items:
        if not isinstance(item, dict):
            continue
        name = safe_filename(str(item.get("name") or "document.pdf"))
        content_b64 = item.get("contentBase64")
        if not isinstance(content_b64, str) or not content_b64:
            continue
        try:
            content = base64.b64decode(content_b64)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Failed to decode file {name}: {exc}") from exc
        if not name.lower().endswith(".pdf"):
            raise ValueError(f"Only PDF uploads are supported. Invalid file: {name}")
        decoded.append({"name": name, "content": content})
    return decoded


def write_uploaded_files(course_id: str, files: List[Dict[str, Any]]) -> List[str]:
    course_dir = RAW_DIR / course_id
    course_dir.mkdir(parents=True, exist_ok=True)
    saved_files: List[str] = []
    for file_info in files:
        target_path = unique_path(course_dir / file_info["name"])
        target_path.write_bytes(file_info["content"])
        saved_files.append(target_path.name)
    return saved_files


def choose_post_process_command(course_id: str) -> List[str]:
    course_specific = EXTRACTION_DIR / f"post_process_document_{course_id}.py"
    if course_specific.exists():
        if course_id == "5703":
            return [sys.executable, str(course_specific), "--course-id", course_id]
        return [sys.executable, str(course_specific)]
    return [
        sys.executable,
        str(EXTRACTION_DIR / "post_process_document_generic.py"),
        "--course-id",
        course_id,
    ]


def run_command(command: List[str], log_lines: List[str], step_name: str) -> None:
    log_lines.append(f"$ {' '.join(command)}")
    process = subprocess.run(
        command,
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
    )
    if process.stdout:
        log_lines.extend(line for line in process.stdout.strip().splitlines() if line.strip())
    if process.stderr:
        log_lines.extend(line for line in process.stderr.strip().splitlines() if line.strip())
    if process.returncode != 0:
        raise RuntimeError(f"{step_name} failed with exit code {process.returncode}")


def execute_ingestion_pipeline(job: Dict[str, Any]) -> None:
    course_id = job["course_id"]
    course_name = job["course_name"]
    log_lines: List[str] = []

    commands = [
        (
            "extract",
            [
                sys.executable,
                str(EXTRACTION_DIR / "extract_course_documents.py"),
                "--course-id",
                course_id,
                "--course-name",
                course_name,
            ],
        ),
        (
            "post_process",
            choose_post_process_command(course_id),
        ),
        (
            "pre_chunk",
            [
                sys.executable,
                str(ROOT_DIR / "scripts" / "run_before_chunk.py"),
                "--course-id",
                course_id,
            ],
        ),
        (
            "atomic_chunk",
            [
                sys.executable,
                str(ROOT_DIR / "scripts" / "chunk" / "atomic_chunk.py"),
                "--course-id",
                course_id,
            ],
        ),
        (
            "atomic_embedding",
            [
                sys.executable,
                str(ROOT_DIR / "scripts" / "chunk" / "atomic_embedding.py"),
                "--course-id",
                course_id,
            ],
        ),
        (
            "semantic_chunk",
            [
                sys.executable,
                str(ROOT_DIR / "scripts" / "chunk" / "semantic_chunk.py"),
                "--course-id",
                course_id,
            ],
        ),
        (
            "semantic_embedding",
            [
                sys.executable,
                str(ROOT_DIR / "scripts" / "chunk" / "semantic_embedding.py"),
                "--course-id",
                course_id,
            ],
        ),
        (
            "build_faiss",
            [
                sys.executable,
                str(ROOT_DIR / "scripts" / "retrieval" / "build_faiss_index.py"),
                "--course-id",
                course_id,
                "--target",
                "both",
            ],
        ),
    ]

    for step_name, command in commands:
        job["current_step"] = step_name
        job["steps"].append(
            {
                "name": step_name,
                "status": "running",
                "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
        )
        run_command(command, log_lines, step_name)
        job["steps"][-1]["status"] = "completed"
        job["steps"][-1]["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        job["logs"] = log_lines[-120:]

    job["status"] = "completed"
    job["current_step"] = "done"
    job["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    job["logs"] = log_lines[-120:]


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "CourseAssistantDemo/0.2"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/config":
            json_response(self, HTTPStatus.OK, build_config(self.server.index_dir))
            return

        if path.startswith("/api/jobs/"):
            job_id = path.removeprefix("/api/jobs/").strip()
            if not job_id:
                json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Job id is required"})
                return
            job = self.server.get_job(job_id)
            if job is None:
                json_response(self, HTTPStatus.NOT_FOUND, {"error": "Job not found"})
                return
            json_response(self, HTTPStatus.OK, job)
            return

        if path == "/" or path == "/index.html":
            serve_file(self, self.server.static_dir / "index.html")
            return

        asset_path = (self.server.static_dir / path.lstrip("/")).resolve()
        try:
            asset_path.relative_to(self.server.static_dir.resolve())
        except ValueError:
            json_response(self, HTTPStatus.FORBIDDEN, {"error": "Invalid asset path"})
            return
        serve_file(self, asset_path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/answer":
            self.handle_answer()
            return
        if parsed.path == "/api/course-ingest":
            self.handle_course_ingest()
            return
        json_response(self, HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint"})

    def handle_answer(self) -> None:
        payload = self.read_json_body()
        if payload is None:
            return

        query = str(payload.get("query") or "").strip()
        if not query:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Query is required"})
            return

        try:
            client = OpenAI()
            course_id = str(payload.get("courseId") or default_course_id(available_courses(self.server.index_dir)))
            target = str(payload.get("target") or "both")
            top_k = int(payload.get("topK") or 8)
            context_k = int(payload.get("contextK") or 6)
            candidate_k = int(payload.get("candidateK") or 4)
            rrf_k = int(payload.get("rrfK") or 60)
            faiss_weight = float(payload.get("faissWeight") or 1.0)
            bm25_weight = float(payload.get("bm25Weight") or 1.0)
            embedding_model = str(payload.get("embeddingModel") or DEFAULT_EMBEDDING_MODEL)
            generation_model = str(payload.get("generationModel") or DEFAULT_GENERATION_MODEL)
            retrieval_method = str(payload.get("retrievalMethod") or DEFAULT_RETRIEVAL_METHOD)
            dense_pool_multiplier = int(payload.get("densePoolMultiplier") or 4)
            dense_rerank_dense_weight = float(payload.get("denseRerankDenseWeight") or 0.65)
            dense_rerank_bm25_weight = float(payload.get("denseRerankBm25Weight") or 0.35)
            memory_window = max(1, int(payload.get("memoryWindow") or 3))
            conversation_summary = clean_text(payload.get("conversationSummary"))
            recent_turns = normalize_turns(payload.get("recentTurns"))
            current_topic = clean_text(payload.get("currentTopic"))
            retrieval_query = build_retrieval_query(
                client=client,
                model=generation_model,
                current_query=query,
                conversation_summary=conversation_summary,
                recent_turns=recent_turns,
                current_topic=current_topic,
            )

            retrieved_results = retrieve_results(
                course_id=course_id,
                index_dir=self.server.index_dir,
                query=retrieval_query,
                target=target,
                top_k=top_k,
                candidate_k=candidate_k,
                embedding_model=embedding_model,
                method=retrieval_method,
                rrf_k=rrf_k,
                faiss_weight=faiss_weight,
                bm25_weight=bm25_weight,
                dense_pool_multiplier=dense_pool_multiplier,
                dense_rerank_dense_weight=dense_rerank_dense_weight,
                dense_rerank_bm25_weight=dense_rerank_bm25_weight,
            )
            context_results = select_context_results(retrieved_results, context_k)
            answer = generate_answer(
                client=client,
                model=generation_model,
                query=query,
                context_results=context_results,
                conversation_summary=conversation_summary,
                recent_turns=recent_turns,
                current_topic=current_topic,
            )
            updated_topic = infer_current_topic(
                client=client,
                model=generation_model,
                previous_topic=current_topic,
                user_query=query,
                assistant_answer=answer,
            )
            memory = update_conversation_memory(
                client=client,
                model=generation_model,
                existing_summary=conversation_summary,
                recent_turns=recent_turns,
                new_user_query=query,
                assistant_answer=answer,
                memory_window=memory_window,
            )
        except Exception as exc:  # noqa: BLE001
            json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return

        json_response(
            self,
            HTTPStatus.OK,
            {
                "answer": answer,
                "sources": build_sources(context_results),
                "retrievedResultsCount": len(retrieved_results),
                "usedSourcesCount": len(context_results),
                "retrievalQuery": retrieval_query,
                "conversationSummary": memory["conversationSummary"],
                "recentTurns": memory["recentTurns"],
                "currentTopic": updated_topic,
                "memoryWindow": memory_window,
            },
        )

    def handle_course_ingest(self) -> None:
        payload = self.read_json_body()
        if payload is None:
            return

        course_id = slugify_course_id(str(payload.get("courseId") or ""))
        course_name = clean_text(payload.get("courseName"))
        files = decode_uploaded_files(payload.get("files"))

        if not course_id:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Course id is required"})
            return
        if not course_name:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Course name is required"})
            return
        if not files:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "At least one PDF file is required"})
            return

        try:
            saved_files = write_uploaded_files(course_id, files)
            job = self.server.create_job(
                course_id=course_id,
                course_name=course_name,
                uploaded_files=saved_files,
            )
        except Exception as exc:  # noqa: BLE001
            json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return

        json_response(
            self,
            HTTPStatus.ACCEPTED,
            {
                "jobId": job["job_id"],
                "status": job["status"],
                "courseId": course_id,
                "courseName": course_name,
                "uploadedFiles": saved_files,
            },
        )

    def read_json_body(self) -> Dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        try:
            payload = json.loads(raw_body or b"{}")
        except json.JSONDecodeError:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Request body must be valid JSON"})
            return None
        if not isinstance(payload, dict):
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Request body must be a JSON object"})
            return None
        return payload

    def log_message(self, format: str, *args: Any) -> None:
        return


class DemoServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], static_dir: Path, index_dir: Path) -> None:
        super().__init__(server_address, DemoHandler)
        self.static_dir = static_dir
        self.index_dir = index_dir
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.jobs_lock = threading.Lock()

    def create_job(self, *, course_id: str, course_name: str, uploaded_files: List[str]) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "status": "queued",
            "course_id": course_id,
            "course_name": course_name,
            "uploaded_files": uploaded_files,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "steps": [],
            "logs": [],
            "current_step": "queued",
        }
        with self.jobs_lock:
            self.jobs[job_id] = job
        worker = threading.Thread(target=self.run_job, args=(job_id,), daemon=True)
        worker.start()
        return job

    def get_job(self, job_id: str) -> Dict[str, Any] | None:
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            return json.loads(json.dumps(job))

    def run_job(self, job_id: str) -> None:
        with self.jobs_lock:
            job = self.jobs[job_id]
            job["status"] = "running"
            job["started_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

        try:
            execute_ingestion_pipeline(job)
        except Exception as exc:  # noqa: BLE001
            with self.jobs_lock:
                failed_job = self.jobs[job_id]
                if failed_job.get("steps"):
                    failed_job["steps"][-1]["status"] = "failed"
                    failed_job["steps"][-1]["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
                failed_job["status"] = "failed"
                failed_job["error"] = str(exc)
                failed_job["current_step"] = "failed"
                failed_job["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--index-dir", default=str(INDEX_DIR))
    parser.add_argument("--static-dir", default=str(STATIC_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = DemoServer(
        (args.host, args.port),
        static_dir=Path(args.static_dir).resolve(),
        index_dir=Path(args.index_dir).resolve(),
    )
    print(f"Serving demo at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
