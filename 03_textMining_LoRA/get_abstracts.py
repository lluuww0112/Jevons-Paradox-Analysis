import re
from pathlib import Path
from typing import List
import os

def extract_abstracts_from_folder(md_dir: str) -> List[str]:
    """
    특정 폴더 내 markdown 파일들에서 Abstract를 추출하여 list[str]로 반환한다.

    지원 시작 형식:
      - ## Abstract
      - # Abstract
      - Abstract
      - Abstract-This paper ...
      - Abstract -This paper ...
      - Abstract - This paper ...
      - Abstract: This paper ...

    종료 규칙:
      - abstract 내용 이후 첫 번째 빈 줄 (\\n\\n)
    """

    md_dir = Path(md_dir)
    abstracts: List[str] = []

    # Abstract 헤더 + inline 텍스트까지 허용
    start_pattern = re.compile(
        r"(?:^|\n)"                 # 문서 처음 or 줄 시작
        r"(?:#+\s*)?"               # optional markdown header
        r"abstract"                 # Abstract
        r"(?:\s*[-–—:]\s*|\s+)"     # -, —, :, 혹은 공백
        r"*",
        re.IGNORECASE,
    )

    for md_file in md_dir.glob("*.md"):
        text = md_file.read_text(encoding="utf-8", errors="ignore")

        match = start_pattern.search(text)
        if not match:
            continue

        start_idx = match.end()

        rest = text[start_idx:]

        # 첫 빈 줄 = 종료
        end_match = re.search(r"\n\s*\n", rest)
        if end_match:
            end_idx = start_idx + end_match.start()
        else:
            end_idx = len(text)

        abstract_text = text[start_idx:end_idx].strip()

        if abstract_text:
            abstracts.append(abstract_text)

    return abstracts



if __name__ == "__main__":

    abstracts = extract_abstracts_from_folder("./papers/IEEE/2022/01_markdown")
    print(len(abstracts))
    print(abstracts)