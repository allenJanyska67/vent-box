#!/usr/bin/env bash

set -euo pipefail

# Remove references to grievance files that no longer exist.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
index_file="${repo_root}/grievances/index.json"

if [[ ! -f "${index_file}" ]]; then
  echo "Could not find ${index_file}" >&2
  exit 1
fi

python3 - "${index_file}" "${repo_root}" <<'PY'
import json
import pathlib
import sys

index_path = pathlib.Path(sys.argv[1])
repo_root = pathlib.Path(sys.argv[2])

try:
    data = json.loads(index_path.read_text())
except json.JSONDecodeError as exc:
    sys.exit(f"Failed to parse {index_path}: {exc}")

topics = data.get("topics", {})
removed_files = []
removed_topics = []

for topic_key in list(topics.keys()):
    topic = topics[topic_key]
    grievances = topic.get("grievances", [])
    filtered = []

    for entry in grievances:
        rel_path = entry.get("file")
        if not rel_path:
            filtered.append(entry)
            continue

        file_path = repo_root / rel_path
        if not file_path.exists():
            removed_files.append((topic_key, rel_path))
            continue

        filtered.append(entry)

    if not filtered:
        removed_topics.append(topic_key)
        topics.pop(topic_key, None)
        continue

    topic["grievances"] = filtered
    topic["occurrences"] = len(filtered)
    topic["highest_level"] = max((entry.get("level", 0) for entry in filtered), default=0)

if not removed_files and not removed_topics:
    print("No unused grievance references found.")
    sys.exit(0)

index_path.write_text(json.dumps(data, indent=2) + "\n")

if removed_files:
    print("Removed grievances pointing to missing files:")
    for topic_key, rel_path in removed_files:
        print(f"  - {topic_key}: {rel_path}")

if removed_topics:
    print("Removed topics with no remaining grievances:")
    for topic_key in removed_topics:
        print(f"  - {topic_key}")
PY
