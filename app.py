import csv
import io
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, flash, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from detector import FaceDuplicateDetector

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
THUMB_DIR = BASE_DIR / "static" / "thumbnails"
DB_PATH = BASE_DIR / "instance" / "face_duplicates.sqlite3"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "change-this-for-local-development")
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB per request
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = FaceDuplicateDetector(
            model_name=os.environ.get("INSIGHTFACE_MODEL", "buffalo_l"),
            providers=os.environ.get("ONNX_PROVIDERS", "CPUExecutionProvider").split(","),
        )
    return _detector


def db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                original_name TEXT NOT NULL,
                thumbnail TEXT,
                face_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'processed',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_a TEXT NOT NULL,
                image_b TEXT NOT NULL,
                similarity REAL NOT NULL,
                threshold REAL NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(image_a, image_b)
            )
        """)


init_db()


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def make_thumbnail(image_path, thumbnail_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    h, w = image.shape[:2]
    scale = min(420 / max(w, 1), 420 / max(h, 1), 1.0)
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
    return bool(cv2.imwrite(str(thumbnail_path), resized))


@app.route("/")
def index():
    with db() as conn:
        images = conn.execute("SELECT * FROM images ORDER BY created_at DESC").fetchall()
        matches = conn.execute("""
            SELECT m.*, a.original_name AS name_a, a.thumbnail AS thumb_a,
                   b.original_name AS name_b, b.thumbnail AS thumb_b
            FROM matches m
            JOIN images a ON a.id = m.image_a
            JOIN images b ON b.id = m.image_b
            ORDER BY m.similarity DESC
            LIMIT 30
        """).fetchall()
    return render_template("index.html", images=images, matches=matches)


@app.route("/upload", methods=["POST"])
def upload():
    uploaded = request.files.getlist("images")
    threshold_raw = request.form.get("threshold", "0.48")
    try:
        threshold = float(threshold_raw)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError
    except ValueError:
        flash("Similarity threshold must be between 0 and 1.", "error")
        return redirect(url_for("index"))

    candidates = [f for f in uploaded if f and f.filename]
    if not candidates:
        flash("Choose one or more image files first.", "error")
        return redirect(url_for("index"))

    valid = [f for f in candidates if allowed_file(f.filename)]
    if not valid:
        flash("No supported images found. Use JPG, JPEG, PNG, WEBP, or BMP.", "error")
        return redirect(url_for("index"))

    # Load model before saving anything, so a missing model dependency is surfaced clearly.
    try:
        detector = get_detector()
    except Exception as exc:
        flash(f"Could not initialize InsightFace. Check setup instructions. Details: {exc}", "error")
        return redirect(url_for("index"))

    new_records = []
    skipped = 0
    for file in valid:
        image_id = uuid.uuid4().hex
        safe_name = secure_filename(file.filename) or "image"
        stored_name = f"{image_id}_{safe_name}"
        image_path = UPLOAD_DIR / stored_name
        thumb_name = f"{image_id}.jpg"
        thumb_path = THUMB_DIR / thumb_name
        file.save(image_path)

        image = cv2.imread(str(image_path))
        if image is None:
            image_path.unlink(missing_ok=True)
            skipped += 1
            continue

        try:
            faces = detector.detect_faces(image)
            face_count = len(faces)
            new_records.append({
                "id": image_id,
                "filename": stored_name,
                "original_name": safe_name,
                "thumbnail": f"thumbnails/{thumb_name}",
                "face_count": face_count,
                "image_path": image_path,
                "embeddings": [face["embedding"] for face in faces],
            })
            make_thumbnail(image_path, thumb_path)
        except Exception:
            image_path.unlink(missing_ok=True)
            skipped += 1

    if not new_records:
        flash("No readable images could be processed.", "error")
        return redirect(url_for("index"))

    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    with db() as conn:
        for record in new_records:
            conn.execute("""
                INSERT INTO images (id, filename, original_name, thumbnail, face_count, status, created_at)
                VALUES (?, ?, ?, ?, ?, 'processed', ?)
            """, (record["id"], record["filename"], record["original_name"],
                  record["thumbnail"], record["face_count"], now))

    # Compare each new image to every stored image. Each image can contain multiple faces;
    # a pair matches if the most similar face pair meets the selected threshold.
    with db() as conn:
        existing = conn.execute("SELECT id, filename FROM images").fetchall()

    image_embeddings = {}
    for record in new_records:
        image_embeddings[record["id"]] = record["embeddings"]

    # Compute embeddings for older images too, so results persist across upload sessions.
    for row in existing:
        if row["id"] in image_embeddings:
            continue
        path = UPLOAD_DIR / row["filename"]
        old_img = cv2.imread(str(path))
        if old_img is None:
            image_embeddings[row["id"]] = []
            continue
        try:
            image_embeddings[row["id"]] = [
                face["embedding"] for face in detector.detect_faces(old_img)
            ]
        except Exception:
            image_embeddings[row["id"]] = []

    inserted_matches = 0
    with db() as conn:
        for i, row_a in enumerate(existing):
            for row_b in existing[i + 1:]:
                # Avoid recalculating old-old pairs; only evaluate pairs involving a new upload.
                if row_a["id"] not in {r["id"] for r in new_records} and row_b["id"] not in {r["id"] for r in new_records}:
                    continue
                best = detector.best_similarity(
                    image_embeddings.get(row_a["id"], []),
                    image_embeddings.get(row_b["id"], [])
                )
                if best is not None and best >= threshold:
                    a, b = sorted((row_a["id"], row_b["id"]))
                    conn.execute("""
                        INSERT INTO matches (image_a, image_b, similarity, threshold, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(image_a, image_b) DO UPDATE SET
                            similarity=excluded.similarity,
                            threshold=excluded.threshold,
                            created_at=excluded.created_at
                    """, (a, b, best, threshold, now))
                    inserted_matches += 1

    message = f"Processed {len(new_records)} image(s). Found/updated {inserted_matches} matching image pair(s)."
    if skipped:
        message += f" Skipped {skipped} unreadable image(s)."
    flash(message, "success")
    return redirect(url_for("index"))


@app.route("/export.csv")
def export_csv():
    with db() as conn:
        rows = conn.execute("""
            SELECT a.original_name AS image_a, b.original_name AS image_b,
                   m.similarity, m.threshold, m.created_at
            FROM matches m
            JOIN images a ON a.id = m.image_a
            JOIN images b ON b.id = m.image_b
            ORDER BY m.similarity DESC
        """).fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["image_a", "image_b", "cosine_similarity", "threshold", "created_at"])
    for row in rows:
        writer.writerow([row["image_a"], row["image_b"], f'{row["similarity"]:.4f}',
                         f'{row["threshold"]:.4f}', row["created_at"]])
    memory = io.BytesIO(output.getvalue().encode("utf-8-sig"))
    memory.seek(0)
    return send_file(memory, mimetype="text/csv", as_attachment=True,
                     download_name="face_duplicate_matches.csv")


@app.route("/clear", methods=["POST"])
def clear():
    # This deletes only local app data. Ask for explicit confirmation in the UI.
    with db() as conn:
        conn.execute("DELETE FROM matches")
        rows = conn.execute("SELECT filename, thumbnail FROM images").fetchall()
        conn.execute("DELETE FROM images")
    for row in rows:
        (UPLOAD_DIR / row["filename"]).unlink(missing_ok=True)
        if row["thumbnail"]:
            (BASE_DIR / "static" / row["thumbnail"]).unlink(missing_ok=True)
    flash("Local image library and match history cleared.", "success")
    return redirect(url_for("index"))


@app.errorhandler(413)
def too_large(_error):
    flash("Upload is too large. Keep the total request under 12 MB.", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    # Local development only. Do not expose this development server publicly.
    app.run(debug=True, host="127.0.0.1", port=5000)
