"""CSV data serialization and Markdown table formatting."""

import csv
import json


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    def parse(key, value):
        if value == "":
            return None
        if key.endswith("_correct"):
            return value == "True"
        if key.endswith("_receiver_generated_tokens") or key in {
            "count",
            "sample_index",
        }:
            return int(value)
        if key == "ratio":
            return float(value)
        return value

    with path.open(encoding="utf-8", newline="") as stream:
        return [
            {key: parse(key, value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def display(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "&#124;")
        .replace("\n", "<br>")
    )


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *(
                "| " + " | ".join(display(value) for value in row) + " |"
                for row in rows
            ),
        ]
    )
