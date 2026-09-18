import base64
import mimetypes
import os
from src.connection import get_chat_completion_text


def _collect_files(folder_path: str, extensions: list[str] | None = None) -> list[str]:
    files_found: list[str] = []
    for root, dirs, files in os.walk(folder_path):
        for filename in sorted(files):
            if filename.startswith("."):
                continue
            if extensions and not any(filename.lower().endswith(ext.lower()) for ext in extensions):
                continue
            filepath = os.path.join(root, filename)
            files_found.append(filepath)
    return files_found


def _build_user_content_with_files(prompt: str, file_paths: list[str]) -> list[dict]:
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"{prompt}\n\n"
                "Input format instructions:\n"
                "- Non-image files are attached as [FILE_ATTACHMENT] blocks with base64 payload.\n"
                "- Use the provided mime_type and decode the base64 data when needed.\n"
                "- Keep source file path provenance in extracted attributes."
            ),
        }
    ]
    for filepath in file_paths:
        try:
            with open(filepath, "rb") as f:
                raw = f.read()
        except PermissionError as e:
            print(f"Skipping {filepath}: {e}")
            continue

        mime_type, _ = mimetypes.guess_type(filepath)
        mime_type = mime_type or "application/octet-stream"
        b64 = base64.b64encode(raw).decode("ascii")

        if mime_type.startswith("image/"):
            content.append(
                {
                    "type": "text",
                    "text": f"Attached image file: {filepath}",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                }
            )
            continue

        # For non-image files, keep bytes as base64 so we pass the file itself instead of raw CSV/text payload.
        content.append(
            {
                "type": "text",
                "text": (
                    "[FILE_ATTACHMENT]\n"
                    f"path: {filepath}\n"
                    f"mime_type: {mime_type}\n"
                    "encoding: base64\n"
                    f"data: {b64}\n"
                    "[/FILE_ATTACHMENT]"
                ),
            }
        )
    return content


# Reads a folder and sends its content together with a prompt to the model.
# client: OpenAI Client
# folder_path: Path to the folder containing the files
# prompt: Instruction for what to do with the data
# extensions: Optional filter for file extensions
def send_folder_to_model(client, folder_path: str, prompt: str, extensions: list[str] | None = None):
    file_paths = _collect_files(folder_path, extensions)
    print(f"{len(file_paths)} files prepared from {folder_path}.")

    user_content = _build_user_content_with_files(prompt, file_paths)
    return get_chat_completion_text(client, user_content)
