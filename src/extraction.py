"""PDF reading and BAML-based structured extraction for resumes/job postings."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pymupdf

from .config import PROJECT_ROOT


# Make ``baml_client`` importable when running as a Dash module from src/.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from baml_client.sync_client import b  # noqa: E402  (after path injection)
from baml_client.types import JobPosting, JobPostingDomain, Resume  # noqa: E402


KNOWN_FOLDER_DOMAINS: set[str] = {
    "ACCOUNTANT",
    "ADVOCATE",
    "AGRICULTURE",
    "APPAREL",
    "ARTS",
    "AUTOMOBILE",
    "AVIATION",
    "BANKING",
    "BPO",
    "BUSINESS-DEVELOPMENT",
    "CHEF",
    "CONSTRUCTION",
    "CONSULTANT",
    "DESIGNER",
    "DIGITAL-MEDIA",
    "ENGINEERING",
    "FINANCE",
    "FITNESS",
    "HEALTHCARE",
    "HR",
    "INFORMATION-TECHNOLOGY",
    "PUBLIC-RELATIONS",
    "SALES",
    "TEACHER",
}

# BAML enum values use underscores; on disk we use hyphenated folder names.
_ENUM_TO_FOLDER = {
    "BUSINESS_DEVELOPMENT": "BUSINESS-DEVELOPMENT",
    "DIGITAL_MEDIA": "DIGITAL-MEDIA",
    "INFORMATION_TECHNOLOGY": "INFORMATION-TECHNOLOGY",
    "PUBLIC_RELATIONS": "PUBLIC-RELATIONS",
}


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Return concatenated text content from an uploaded PDF."""

    doc = pymupdf.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    try:
        pages = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    return "\n\n".join(pages).strip()


def domain_enum_to_folder(domain: JobPostingDomain | str) -> str:
    """Normalise a domain value (BAML enum or string) to the folder format."""

    raw = (
        domain.value
        if isinstance(domain, JobPostingDomain)
        else str(domain)
    )
    raw = raw.strip().upper().replace("JOBPOSTINGDOMAIN.", "")
    return _ENUM_TO_FOLDER.get(raw, raw.replace("_", "-"))


def _model_to_dict(obj: Any) -> dict:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(obj)


def extract_resume(pdf_bytes: bytes) -> tuple[Resume, dict, str]:
    """Parse a CV PDF.

    Returns
    -------
    resume :
        Pydantic ``Resume`` model from BAML.
    payload :
        Plain ``dict`` representation suitable for embedding/serialisation.
    domain :
        Folder-format domain string (e.g. ``INFORMATION-TECHNOLOGY``).
    """

    text = extract_pdf_text(pdf_bytes)
    if not text:
        raise ValueError("PDF contains no extractable text")

    resume: Resume = b.ExtractResumeLlamaCppPC(text)
    payload = _model_to_dict(resume)

    summary = json.dumps(payload, ensure_ascii=False)
    inferred_domain = b.InferResumeDomain(summary)
    folder_domain = domain_enum_to_folder(inferred_domain)

    if folder_domain not in KNOWN_FOLDER_DOMAINS:
        # Fall back to the closest match by uppercasing the title.
        title_guess = (payload.get("title") or "").strip().upper().replace(" ", "-")
        if title_guess in KNOWN_FOLDER_DOMAINS:
            folder_domain = title_guess

    return resume, payload, folder_domain


def extract_job_posting(pdf_bytes: bytes) -> tuple[JobPosting, dict, str]:
    """Parse a job posting PDF and return the structured BAML output.

    Returns ``(model, payload, folder_domain)``.
    """

    text = extract_pdf_text(pdf_bytes)
    if not text:
        raise ValueError("PDF contains no extractable text")

    job: JobPosting = b.ExtractJobPostingLlamaCppPC(text)
    payload = _model_to_dict(job)
    folder_domain = domain_enum_to_folder(job.domain)
    payload["domain"] = folder_domain
    return job, payload, folder_domain


def payload_to_text(payload: dict) -> str:
    """Match the on-disk indexing serialisation used by the embedding notebook."""

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)


def load_processed_json(rel_or_abs_path: str) -> dict:
    """Load an indexed JSON document by ``source_path`` payload value.

    Accepts either a project-relative path (as stored in payloads) or an
    absolute path. Restricts reads to the project root for safety.
    """

    candidate = Path(rel_or_abs_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()

    project_root = PROJECT_ROOT.resolve()
    if project_root not in candidate.parents and candidate != project_root:
        raise ValueError(f"refusing to read path outside project: {candidate}")

    with candidate.open("r", encoding="utf-8") as f:
        return json.load(f)
