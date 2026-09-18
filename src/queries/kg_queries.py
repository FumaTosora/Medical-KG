import csv
import io
import json


# Formats a list of dicts as a CSV string.
def _format_as_csv(rows: list, fieldnames: list) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        r = {k: v for k, v in dict(row).items() if k in fieldnames}
        if "attributes" in r and isinstance(r["attributes"], dict):
            r["attributes"] = json.dumps(r["attributes"], ensure_ascii=False)
        writer.writerow(r)
    return output.getvalue()


def _normalize_row(row: dict, fieldnames: list[str]) -> dict:
    cleaned = {k: v for k, v in row.items() if k in fieldnames}
    for key in fieldnames:
        if cleaned.get(key) is None:
            cleaned[key] = ""
    return cleaned


def _parse_csv_block(block: str, fieldnames: list[str]) -> list[dict]:
    rows = []
    reader = csv.DictReader(io.StringIO(block.strip()))
    for row in reader:
        row = _normalize_row(row, fieldnames)
        if row.get("attributes"):
            try:
                row["attributes"] = json.loads(row["attributes"])
            except (json.JSONDecodeError, TypeError):
                row["attributes"] = {}
        rows.append(row)
    return rows


# Parses the model's CSV response into two lists (entities, relations).
def _parse_csv_response(text: str) -> dict:
    text = text.strip()
    # Strip code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()

    entity_header = ["id", "type", "name", "attributes"]
    relation_header = ["source", "target", "type", "attributes"]

    # Primary path: marker-based parsing (===ENTITIES=== / ===RELATIONS=== / ===END===).
    # Locate markers case-insensitively so minor LLM deviations are tolerated.
    lines = text.splitlines()
    marker_indices = {}
    for i, line in enumerate(lines):
        stripped = line.strip().upper()
        if stripped == "===ENTITIES===":
            marker_indices["entities"] = i
        elif stripped == "===RELATIONS===":
            marker_indices["relations"] = i
        elif stripped == "===END===":
            marker_indices["end"] = i

    if "entities" in marker_indices and "relations" in marker_indices:
        e_start = marker_indices["entities"] + 1
        e_end = marker_indices["relations"]
        r_start = marker_indices["relations"] + 1
        r_end = marker_indices.get("end", len(lines))

        entity_block = "\n".join(lines[e_start:e_end]).strip()
        relation_block = "\n".join(lines[r_start:r_end]).strip()

        entities = _parse_csv_block(entity_block, entity_header) if entity_block else []
        relations = _parse_csv_block(relation_block, relation_header) if relation_block else []
        return {"entities": entities, "relations": relations}

    # Fallback: blank-line block splitting for responses without markers.
    entities = []
    relations = []

    raw_blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

    # A block that starts with the entity header may contain the relations
    # header mid-way (no blank line between the two CSV tables). Split it.
    blocks = []
    for block in raw_blocks:
        blines = block.strip().split("\n")
        split_idx = next(
            (i for i, l in enumerate(blines) if i > 0 and l.strip().startswith("source,")),
            None,
        )
        if split_idx is not None:
            top = "\n".join(blines[:split_idx]).strip()
            bottom = "\n".join(blines[split_idx:]).strip()
            if top:
                blocks.append(top)
            if bottom:
                blocks.append(bottom)
        else:
            blocks.append(block)

    for block in blocks:
        blines = block.strip().split("\n")
        first_line = blines[0].strip()

        if first_line in (",".join(entity_header), ",".join(relation_header)):
            if len(blines) == 1:
                continue

        if first_line.lower() in ("entities", "relations"):
            blines = blines[1:]
            if not blines:
                continue
            first_line = blines[0].strip()
            block = "\n".join(blines)

        if first_line.startswith("id,"):
            entities.extend(_parse_csv_block(block, entity_header))
        elif first_line.startswith("source,"):
            relations.extend(_parse_csv_block(block, relation_header))

    return {"entities": entities, "relations": relations}

