# main.py
# Streamlit app: Daily manual screenshot uploads + optional OCR + export/download (ZIP + CSV)
#
# Run:
#   pip install streamlit pandas pillow
# Optional OCR (recommended):
#   pip install pytesseract
#   # AND install the Tesseract engine on your machine (Windows/Linux/Mac) then ensure it's in PATH.
#
# Start:
#   streamlit run main.py

import os
import io
import re
import csv
import json
import time
import sqlite3
import zipfile
from pathlib import Path
from datetime import datetime, date

import streamlit as st
import pandas as pd
from PIL import Image

# ----------------------------
# App config
# ----------------------------
APP_TITLE = "NSE Dashboard — Daily Uploads + OCR + Downloads"
DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "screenshots"
DB_PATH = DATA_DIR / "app.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Upload screenshots daily, optionally run OCR, and download everything anytime (ZIP + CSV).")

# ----------------------------
# DB helpers
# ----------------------------
def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_date TEXT NOT NULL,
            label TEXT,
            notes TEXT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            mimetype TEXT,
            uploaded_at TEXT NOT NULL,
            ocr_text TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_uploads_date ON uploads(upload_date)
    """)
    conn.commit()
    return conn

CONN = db_conn()

def insert_upload(upload_date: str, label: str | None, notes: str | None,
                  filename: str, filepath: str, mimetype: str | None,
                  uploaded_at: str, ocr_text: str | None):
    CONN.execute(
        """INSERT INTO uploads
           (upload_date, label, notes, filename, filepath, mimetype, uploaded_at, ocr_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (upload_date, label, notes, filename, filepath, mimetype, uploaded_at, ocr_text),
    )
    CONN.commit()

def fetch_uploads(upload_date: str | None = None, limit: int = 500):
    cur = CONN.cursor()
    if upload_date:
        cur.execute(
            """SELECT id, upload_date, label, notes, filename, filepath, mimetype, uploaded_at, ocr_text
               FROM uploads
               WHERE upload_date = ?
               ORDER BY id DESC
               LIMIT ?""",
            (upload_date, limit),
        )
    else:
        cur.execute(
            """SELECT id, upload_date, label, notes, filename, filepath, mimetype, uploaded_at, ocr_text
               FROM uploads
               ORDER BY id DESC
               LIMIT ?""",
            (limit,),
        )
    rows = cur.fetchall()
    cols = ["id", "upload_date", "label", "notes", "filename", "filepath", "mimetype", "uploaded_at", "ocr_text"]
    return pd.DataFrame(rows, columns=cols)

def get_upload_by_id(row_id: int):
    cur = CONN.cursor()
    cur.execute(
        """SELECT id, upload_date, label, notes, filename, filepath, mimetype, uploaded_at, ocr_text
           FROM uploads WHERE id = ?""",
        (row_id,),
    )
    r = cur.fetchone()
    if not r:
        return None
    cols = ["id", "upload_date", "label", "notes", "filename", "filepath", "mimetype", "uploaded_at", "ocr_text"]
    return dict(zip(cols, r))

def update_ocr(row_id: int, ocr_text: str):
    CONN.execute("UPDATE uploads SET ocr_text = ? WHERE id = ?", (ocr_text, row_id))
    CONN.commit()

def delete_upload(row_id: int):
    item = get_upload_by_id(row_id)
    if not item:
        return False
    fp = item["filepath"]
    try:
        if fp and os.path.exists(fp):
            os.remove(fp)
    except Exception:
        pass
    CONN.execute("DELETE FROM uploads WHERE id = ?", (row_id,))
    CONN.commit()
    return True

# ----------------------------
# OCR helpers (optional)
# ----------------------------
def ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401
        return True
    except Exception:
        return False

def run_ocr_on_image(image_path: str) -> str:
    import pytesseract
    img = Image.open(image_path).convert("RGB")
    # Light pre-processing: increase contrast by converting to grayscale sometimes helps.
    # Keep it simple/robust:
    return pytesseract.image_to_string(img)

# ----------------------------
# Export helpers
# ----------------------------
def build_metadata_csv_bytes(df: pd.DataFrame) -> bytes:
    out = io.StringIO()
    df.to_csv(out, index=False)
    return out.getvalue().encode("utf-8")

def build_zip_bytes(df: pd.DataFrame, include_images: bool = True, include_ocr_txt: bool = True) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        # metadata
        z.writestr("metadata.csv", build_metadata_csv_bytes(df))
        # also dump JSON for easy programmatic use
        z.writestr("metadata.json", json.dumps(df.fillna("").to_dict(orient="records"), indent=2))

        # images + OCR text files
        if include_images or include_ocr_txt:
            for _, row in df.iterrows():
                upload_date = row["upload_date"]
                label = (row["label"] or "").strip() if isinstance(row["label"], str) else ""
                safe_label = re.sub(r"[^a-zA-Z0-9_\-]+", "_", label)[:40] if label else "unlabeled"
                base_folder = f"uploads/{upload_date}/{safe_label}/"

                if include_images:
                    fp = row["filepath"]
                    if isinstance(fp, str) and fp and os.path.exists(fp):
                        # keep original filename inside zip
                        z.write(fp, arcname=base_folder + row["filename"])

                if include_ocr_txt:
                    text = row["ocr_text"]
                    if isinstance(text, str) and text.strip():
                        txt_name = row["filename"] + ".txt"
                        z.writestr(base_folder + txt_name, text)
    bio.seek(0)
    return bio.read()

# ----------------------------
# UI
# ----------------------------
tab_upload, tab_gallery, tab_downloads = st.tabs(["📤 Upload", "🖼️ Gallery", "⬇️ Downloads"])

with tab_upload:
    st.subheader("Upload screenshots (daily)")
    left, right = st.columns([1, 1])

    with left:
        upload_date = st.date_input("Date", value=date.today())
        label = st.text_input("Label (optional)", placeholder="e.g., Daily Price List / Portfolio / Watchlist")
    with right:
        notes = st.text_area("Notes (optional)", placeholder="Market notes, setups, why you’re watching a counter, etc.")

    files = st.file_uploader(
        "Choose screenshot(s) (PNG/JPG/WEBP)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )

    ocr_toggle = st.checkbox("Run OCR on upload (requires pytesseract + Tesseract installed)", value=False)

    if ocr_toggle and not ocr_available():
        st.warning("OCR library not found. Install with: pip install pytesseract (and install Tesseract engine).")

    if st.button("Save Uploads", type="primary", disabled=not files):
        saved = 0
        for f in files:
            ts = datetime.now().strftime("%H%M%S")
            safe_name = f.name.replace("/", "_").replace("\\", "_").replace("..", "_")
            out_name = f"{upload_date.isoformat()}__{ts}__{safe_name}"
            out_path = UPLOAD_DIR / out_name

            with open(out_path, "wb") as out:
                out.write(f.getbuffer())

            ocr_text = None
            if ocr_toggle and ocr_available():
                try:
                    ocr_text = run_ocr_on_image(str(out_path))
                except Exception as e:
                    ocr_text = f"[OCR FAILED] {e}"

            insert_upload(
                upload_date=upload_date.isoformat(),
                label=(label.strip() if label else None),
                notes=(notes.strip() if notes else None),
                filename=out_name,
                filepath=str(out_path),
                mimetype=getattr(f, "type", None),
                uploaded_at=datetime.now().isoformat(timespec="seconds"),
                ocr_text=ocr_text
            )
            saved += 1

        st.success(f"Saved {saved} screenshot(s) for {upload_date.isoformat()} ✅")
        st.rerun()

    st.divider()
    st.markdown("#### Today’s quick preview")
    df_today = fetch_uploads(upload_date=date.today().isoformat(), limit=24)
    if df_today.empty:
        st.info("No uploads for today yet.")
    else:
        cols = st.columns(4)
        for i, row in df_today.iterrows():
            with cols[i % 4]:
                st.image(row["filepath"], use_container_width=True)
                st.caption(f"**{row['label'] or 'Screenshot'}** · {row['uploaded_at']}")
                if isinstance(row["notes"], str) and row["notes"].strip():
                    st.write(row["notes"][:180] + ("…" if len(row["notes"]) > 180 else ""))

with tab_gallery:
    st.subheader("Gallery / History")
    top = st.columns([1, 1, 1, 1])
    with top[0]:
        show_all = st.checkbox("Show all dates", value=True)
    with top[1]:
        filter_date = st.date_input("Filter date", value=date.today())
    with top[2]:
        max_rows = st.number_input("Max rows", min_value=50, max_value=2000, value=500, step=50)
    with top[3]:
        st.write("")  # spacer

    query_date = None if show_all else filter_date.isoformat()
    df = fetch_uploads(upload_date=query_date, limit=int(max_rows))

    if df.empty:
        st.info("No screenshots found.")
    else:
        # Group by date
        for d in sorted(df["upload_date"].unique(), reverse=True):
            df_d = df[df["upload_date"] == d].copy()
            with st.expander(f"{d} — {len(df_d)} upload(s)", expanded=(d == date.today().isoformat())):
                grid = st.columns(4)
                for i, row in df_d.iterrows():
                    with grid[i % 4]:
                        st.image(row["filepath"], use_container_width=True)
                        st.caption(f"**{row['label'] or 'Screenshot'}** · {row['uploaded_at']}")
                        if isinstance(row["notes"], str) and row["notes"].strip():
                            st.write(row["notes"])
                        # OCR box
                        if isinstance(row["ocr_text"], str) and row["ocr_text"].strip():
                            with st.popover("View OCR text"):
                                st.code(row["ocr_text"][:8000])
                        else:
                            # Allow OCR later on demand
                            if st.button("Run OCR now", key=f"ocr_{int(row['id'])}", disabled=not ocr_available()):
                                try:
                                    txt = run_ocr_on_image(row["filepath"])
                                    update_ocr(int(row["id"]), txt)
                                    st.success("OCR saved.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"OCR failed: {e}")

                        # Delete
                        if st.button("Delete", key=f"del_{int(row['id'])}"):
                            ok = delete_upload(int(row["id"]))
                            if ok:
                                st.warning("Deleted.")
                                st.rerun()

    st.divider()
    st.markdown("#### Uploads table (for quick searching)")
    st.dataframe(
        df[["id", "upload_date", "label", "notes", "filename", "uploaded_at"]].head(300),
        use_container_width=True
    )

with tab_downloads:
    st.subheader("Download your stored data")
    st.caption("Export metadata (CSV) and/or everything (ZIP with images + OCR text).")

    df_all = fetch_uploads(upload_date=None, limit=100000)  # safe for most personal apps
    st.write(f"Total uploads stored: **{len(df_all)}**")

    colA, colB = st.columns([1, 1])
    with colA:
        export_date_mode = st.selectbox("Export scope", ["All dates", "Single date"])
        export_date = st.date_input("Export date", value=date.today(), disabled=(export_date_mode == "All dates"))

    with colB:
        include_images = st.checkbox("Include images in ZIP", value=True)
        include_ocr_txt = st.checkbox("Include OCR text files in ZIP", value=True)

    if export_date_mode == "Single date":
        df_export = df_all[df_all["upload_date"] == export_date.isoformat()].copy()
    else:
        df_export = df_all.copy()

    st.write(f"Items to export: **{len(df_export)}**")

    # CSV download (metadata)
    csv_bytes = build_metadata_csv_bytes(df_export)
    st.download_button(
        "Download metadata.csv",
        data=csv_bytes,
        file_name="metadata.csv",
        mime="text/csv",
        disabled=df_export.empty
    )

    # ZIP download (everything)
    # Build ZIP only when user clicks to avoid slow UI
    if st.button("Build ZIP for download", type="primary", disabled=df_export.empty):
        with st.spinner("Building ZIP..."):
            zip_bytes = build_zip_bytes(df_export, include_images=include_images, include_ocr_txt=include_ocr_txt)
        st.download_button(
            "Download uploads_export.zip",
            data=zip_bytes,
            file_name="uploads_export.zip",
            mime="application/zip"
        )

    st.divider()
    st.markdown("#### Where your data is stored on disk")
    st.code(f"""
{DATA_DIR.resolve()}/
  app.db
  screenshots/
    <your uploaded images...>
""".strip())