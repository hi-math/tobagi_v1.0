"""domain/ 폴더의 PDF·MD 파일을 도메인 지식(Resource)으로 로드한다.

교과서 등 교수 자료를 학습자 분석·교수자 의사결정·AI 학생 발화 프롬프트에
공통 참조 맥락(Resource)으로 주입하기 위한 모듈.

우선순위:
    1) 같은 스템(stem)의 .md 파일이 있으면 그것을 사용한다 (이미지 기반 PDF 대응).
    2) 그렇지 않으면 pypdf로 .pdf 텍스트를 추출한다.
    3) 추가로 .md 단독 파일도 로드한다.

반환 구조:
    {
        "documents": [
            {"id": str, "source": str, "title": str, "text": str, "char_count": int},
            ...
        ],
        "combined_text": str,      # 모든 문서를 하나로 합친 텍스트 (프롬프트 삽입용)
        "summary": str,            # 문서 목록 요약 (디버깅용)
    }
"""

from __future__ import annotations

from pathlib import Path


def _extract_pdf_text(pdf_path: Path) -> str:
    """pypdf로 PDF 텍스트 추출. 실패 시 빈 문자열."""
    try:
        import pypdf
    except ImportError:
        print(f"  ⚠️ pypdf 미설치 — {pdf_path.name} 텍스트 추출 건너뜀")
        return ""

    try:
        reader = pypdf.PdfReader(str(pdf_path))
        parts = []
        for i, page in enumerate(reader.pages):
            t = page.extract_text() or ""
            if t.strip():
                parts.append(f"[p.{i+1}]\n{t.strip()}")
        return "\n\n".join(parts)
    except Exception as e:
        print(f"  ⚠️ {pdf_path.name} 추출 실패: {e}")
        return ""


def _first_heading_or_stem(text: str, fallback: str) -> str:
    """MD 본문에서 최상위 제목을 찾아 title로 사용. 없으면 파일 stem."""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s.lstrip("# ").strip()
    return fallback


def load_domain_knowledge(base_path="team4", folder="domain", max_chars_per_doc=8000):
    """domain/ 폴더를 스캔하여 도메인 지식 딕셔너리를 반환.

    Args:
        base_path: 프로젝트 루트 (config_loader와 동일한 베이스)
        folder: 도메인 자료 서브폴더 이름 (기본 'domain')
        max_chars_per_doc: 각 문서당 최대 문자 수 (프롬프트 길이 제어). 0 이하면 제한 없음.

    Returns:
        dict: {"documents": [...], "combined_text": str, "summary": str}
    """
    base = Path(base_path)
    domain_dir = base / folder

    empty = {"documents": [], "combined_text": "", "summary": "(도메인 자료 없음)"}
    if not domain_dir.exists():
        return empty

    documents = []
    # .md 파일 먼저 수집 (stem -> path)
    md_by_stem = {p.stem: p for p in domain_dir.glob("*.md")}
    pdf_files = sorted(domain_dir.glob("*.pdf"))
    md_files = sorted(md_by_stem.values())
    processed_md_stems = set()

    # 1) PDF 파일 처리 — 같은 stem의 .md가 있으면 그쪽을 사용
    for pdf in pdf_files:
        # "1학년 1단원.pdf" 와 "1학년_1단원_*.md" 같은 변형도 고려
        matched_md = None
        for md_stem, md_path in md_by_stem.items():
            if pdf.stem.replace(" ", "_") in md_stem or md_stem.startswith(pdf.stem.replace(" ", "_")):
                matched_md = md_path
                break
        if matched_md is not None:
            text = matched_md.read_text(encoding="utf-8")
            title = _first_heading_or_stem(text, matched_md.stem)
            source_note = f"{pdf.name} (큐레이션: {matched_md.name})"
            processed_md_stems.add(matched_md.stem)
        else:
            text = _extract_pdf_text(pdf)
            title = pdf.stem
            source_note = pdf.name

        if max_chars_per_doc and len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + f"\n\n... (이하 생략, 원문 {len(text)}자)"

        documents.append({
            "id": pdf.stem,
            "source": source_note,
            "title": title,
            "text": text,
            "char_count": len(text),
        })

    # 2) 매칭되지 않은 독립 MD 파일
    for md in md_files:
        if md.stem in processed_md_stems:
            continue
        text = md.read_text(encoding="utf-8")
        title = _first_heading_or_stem(text, md.stem)
        if max_chars_per_doc and len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + f"\n\n... (이하 생략, 원문 {len(text)}자)"
        documents.append({
            "id": md.stem,
            "source": md.name,
            "title": title,
            "text": text,
            "char_count": len(text),
        })

    if not documents:
        return empty

    combined_parts = []
    for d in documents:
        combined_parts.append(
            f"=== [{d['title']}] (출처: {d['source']}) ===\n{d['text']}"
        )
    combined_text = "\n\n".join(combined_parts)

    summary = " / ".join(f"{d['title']}({d['char_count']}자)" for d in documents)

    return {
        "documents": documents,
        "combined_text": combined_text,
        "summary": summary,
    }
