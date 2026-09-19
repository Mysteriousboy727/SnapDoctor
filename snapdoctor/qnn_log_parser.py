
import re
from dataclasses import dataclass
from typing import List

FALLBACK_PATTERN = re.compile(
    r"\[E:onnxruntime.*?QNN.*?\]\s*Node\s+\[(.*?)\].*?not supported.*?fallback", re.IGNORECASE
)
ERROR_CODE_PATTERN = re.compile(r"QNN_ERROR_(\d+)|error code[:\s]+(\d+)", re.IGNORECASE)


@dataclass
class FallbackEvent:
    node_name: str
    raw_line: str
    error_code: str = ""


def parse_log(log_path: str) -> List[FallbackEvent]:
    events = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = FALLBACK_PATTERN.search(line)
            if m:
                code_m = ERROR_CODE_PATTERN.search(line)
                code = ""
                if code_m:
                    code = code_m.group(1) or code_m.group(2)
                events.append(FallbackEvent(node_name=m.group(1), raw_line=line.strip(), error_code=code))
    return events


if __name__ == "__main__":
    import sys
    events = parse_log(sys.argv[1] if len(sys.argv) > 1 else "data/logs/sample.log")
    print(f"Found {len(events)} fallback events")
    for e in events:
        print(f"  {e.node_name} (code={e.error_code})")