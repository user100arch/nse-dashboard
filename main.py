# main.py
# NSE Screenshot → OCR → Dataset + Dashboard (Streamlit Cloud friendly)
#
# Requirements (requirements.txt):
#   streamlit
#   pandas
#   pillow
#   numpy
#   pytesseract
#   opencv-python-headless
#
# System packages (packages.txt):
#   tesseract-ocr
#   tesseract-ocr-eng

import os
import io
import re
import json
import sqlite3
import zipfile
import shutil
from pathlib import Path
from datetime import datetime, date

import streamlit as st
import pandas as pd
from PIL import Image

import cv2
import numpy as np

# ----------------------------
# OCR availability
# ----------------------------
try:
    import pytesseract
    PYTESSERACT_OK = True
except Exception:
    PYTESSERACT_OK = False

TESS_BIN = shutil.which("tesseract")
TESSERACT_OK = bool(TESS_BIN) and PYTESSERACT_OK

# ----------------------------
# App config
# ----------------------------
APP_TITLE = "NSE Screenshot → Dataset + Dashboard"
DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "screenshots"
DB_PATH = DATA_DIR / "app.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Upload NSE market screenshots, extract table data, store a dataset, and view dashboards.")

with st.sidebar:
    st.header("System status")
    if PYTESSERACT_OK:
        st.success("pytesseract: OK")
    else:
        st.error("pytesseract: NOT installed (check requirements.txt)")
    if TESS_BIN:
        st.success(f"Tesseract found: {TESS_BIN}")
    else:
        st.error("Tesseract NOT found (check packages.txt)")
    if TESSERACT_OK:
        st.success("OCR: READY ✅")
    else:
        st.warning("OCR: Not ready (install both pytesseract + tesseract-ocr).")

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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_uploads_date ON uploads(upload_date)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_quotes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER NOT NULL,
            quote_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            last_price REAL,
            volume INTEGER,
            direction TEXT,
            raw_line TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(upload_id) REFERENCES uploads(id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_quotes_date ON market_quotes(quote_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_quotes_symbol ON market_quotes(symbol)")

    conn.commit()
    return conn

CONN = db_conn()

def insert_upload(upload_date: str, label: str | None, notes: str | None,
                  filename: str, filepath: str, mimetype: str | None,
                  uploaded_at: str, ocr_text: str | None) -> int:
    cur = CONN.cursor()
    cur.execute(
        """INSERT INTO uploads
           (upload_date, label, notes, filename, filepath, mimetype, uploaded_at, ocr_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (upload_date, label, notes, filename, filepath, mimetype, uploaded_at, ocr_text),
    )
    CONN.commit()
    return int(cur.lastrowid)

def fetch_uploads(upload_date: str | None = None, limit: int = 500) -> pd.DataFrame:
    cur = CONN.cursor()
    if upload_date:
        cur.execute(
            """SELECT id, upload_date, label, notes, filename, filepath, mimetype, uploaded_at, ocr_text
               FROM uploads WHERE upload_date = ?
               ORDER BY id DESC LIMIT ?""",
            (upload_date, limit),
        )
    else:
        cur.execute(
            """SELECT id, upload_date, label, notes, filename, filepath, mimetype, uploaded_at, ocr_text
               FROM uploads ORDER BY id DESC LIMIT ?""",
            (limit,),
        )
    rows = cur.fetchall()
    cols = ["id", "upload_date", "label", "notes", "filename", "filepath", "mimetype", "uploaded_at", "ocr_text"]
    return pd.DataFrame(rows, columns=cols)

def fetch_quotes(date_filter: str | None = None, symbol: str | None = None, limit: int = 50000) -> pd.DataFrame:
    cur = CONN.cursor()
    q = """SELECT id, upload_id, quote_date, symbol, last_price, volume, direction, raw_line, created_at
           FROM market_quotes WHERE 1=1"""
    params = []
    if date_filter:
        q += " AND quote_date = ?"
        params.append(date_filter)
    if symbol:
        q += " AND symbol = ?"
        params.append(symbol.upper().strip())
    q += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    cur.execute(q, tuple(params))
    rows = cur.fetchall()
    cols = ["id","upload_id","quote_date","symbol","last_price","volume","direction","raw_line","created_at"]
    return pd.DataFrame(rows, columns=cols)

def insert_quotes(upload_id: int, quote_date: str, quotes: list[dict]):
    now = datetime.now().isoformat(timespec="seconds")
    cur = CONN.cursor()
    for r in quotes:
        cur.execute(
            """INSERT INTO market_quotes
               (upload_id, quote_date, symbol, last_price, volume, direction, raw_line, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                upload_id,
                quote_date,
                r.get("symbol"),
                r.get("last_price"),
                r.get("volume"),
                r.get("direction"),
                r.get("raw_line"),
                now,
            )
        )
    CONN.commit()

# ----------------------------
# OCR + parsing
# ----------------------------
def preprocess_for_ocr(pil_img: Image.Image,
                       crop_right_pct: float = 0.30,
                       crop_top_pct: float = 0.00,
                       crop_bottom_pct: float = 0.00,
                       upscale: float = 2.5) -> np.ndarray:
    """
    Preprocess screenshot:
    - Crop right-side action buttons (B/S/trash)
    - Upscale
    - Grayscale + denoise
    - Otsu threshold + auto-invert
    """
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    right_crop = int(w * (1.0 - crop_right_pct))
    top_crop = int(h * crop_top_pct)
    bottom_crop = int(h * (1.0 - crop_bottom_pct))

    img = img[top_crop:bottom_crop, :right_crop]

    if upscale and upscale != 1.0:
        img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_ratio = (th > 200).mean()
    if white_ratio < 0.5:
        th = cv2.bitwise_not(th)

    return th

def ocr_image(cv_img: np.ndarray) -> str:
    if not TESSERACT_OK:
        raise RuntimeError("OCR not ready. Ensure packages.txt (tesseract) + requirements.txt (pytesseract).")

    # Better spacing preservation for table-like lines
    config = r"--oem 3 --psm 6 -c preserve_interword_spaces=1"
    return pytesseract.image_to_string(cv_img, config=config)

# Accept comma thousands, optional decimals, and tolerate OCR noise around spaces
ROW_RE = re.compile(
    r"""
    ^\s*
    (?P<symbol>[A-Z0-9\-]{2,12})
    \s+
    (?P<price>\d{1,3}(?:,\d{3})*(?:\.\d+)?)
    \s+
    (?P<volume>\d{1,3}(?:,\d{3})*)
    """,
    re.VERBOSE
)

def parse_ocr_text_to_quotes(ocr_text: str) -> list[dict]:
    quotes = []

    for line in ocr_text.splitlines():
        raw = line.strip()
        if not raw:
            continue

        # Basic cleanup
        raw = raw.replace("\t", " ")
        raw = re.sub(r"\s{2,}", " ", raw)

        # A few common OCR misreads: 'I'/'l' as '1' in numbers can be handled by regex later; keep minimal here.

        m = ROW_RE.match(raw)
        if not m:
            continue

        symbol = m.group("symbol").upper().strip()
        price_str = m.group("price").replace(",", "")
        vol_str = m.group("volume").replace(",", "")

        try:
            last_price = float(price_str)
        except Exception:
            last_price = None

        try:
            volume = int(vol_str)
        except Exception:
            volume = None

        quotes.append({
            "symbol": symbol,
            "last_price": last_price,
            "volume": volume,
            "direction": "UNKNOWN",
            "raw_line": raw
        })

    # de-duplicate by symbol (keep last)
    dedup = {}
    for q in quotes:
        dedup[q["symbol"]] = q
    return list(dedup.values())

# ----------------------------
# Export helpers
# ----------------------------
def build_zip_bytes(quotes_df: pd.DataFrame, uploads_df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("quotes.csv", quotes_df.to_csv(index=False))
        z.writestr("uploads.csv", uploads_df.to_csv(index=False))
        z.writestr("quotes.json", json.dumps(quotes_df.fillna("").to_dict(orient="records"), indent=2))
        z.writestr("uploads.json", json.dumps(uploads_df.fillna("").to_dict(orient="records"), indent=2))

        for _, r in uploads_df.iterrows():
            fp = r.get("filepath")
            if isinstance(fp, str) and fp and os.path.exists(fp):
                z.write(fp, arcname=f"screenshots/{r['filename']}")

    bio.seek(0)
    return bio.read()

# ----------------------------
# UI tabs
# ----------------------------
tab_upload, tab_dashboard, tab_downloads = st.tabs(
    ["📤 Upload + Extract", "📊 Dashboard", "⬇️ Download Dataset"]
)

# ----------------------------
# Upload + Extract
# ----------------------------
with tab_upload:
    st.subheader("1) Upload screenshot  2) Extract table  3) Save dataset")

    col1, col2 = st.columns([1, 1])
    with col1:
        quote_date = st.date_input("Quote date", value=date.today())
        label = st.text_input("Label (optional)", value="NSE Market Watch Screenshot")
    with col2:
        notes = st.text_area("Notes (optional)", placeholder="Any notes about the market today...")

    file = st.file_uploader("Upload screenshot (PNG/JPG/WEBP)", type=["png", "jpg", "jpeg", "webp"])

    st.markdown("### OCR tuning (if extraction misses rows)")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        crop_right_pct = st.slider("Crop right side (%)", 0.0, 0.5, 0.30, 0.01)
    with t2:
        crop_top_pct = st.slider("Crop top (%)", 0.0, 0.3, 0.00, 0.01)
    with t3:
        crop_bottom_pct = st.slider("Crop bottom (%)", 0.0, 0.3, 0.00, 0.01)
    with t4:
        upscale = st.slider("Upscale", 1.0, 3.0, 2.5, 0.1)

    if file:
        pil_img = Image.open(file)
        st.image(pil_img, caption="Uploaded screenshot", use_container_width=True)

        processed = preprocess_for_ocr(
            pil_img,
            crop_right_pct=crop_right_pct,
            crop_top_pct=crop_top_pct,
            crop_bottom_pct=crop_bottom_pct,
            upscale=upscale
        )
        st.image(processed, caption="Preprocessed for OCR", use_container_width=True)

        if not TESSERACT_OK:
            st.warning("OCR is not ready on this deployment. Check sidebar status and your packages.txt / requirements.txt.")

        if st.button("Extract & Save Dataset", type="primary", disabled=not TESSERACT_OK):
            # Save image to disk
            ts = datetime.now().strftime("%H%M%S")
            safe_name = file.name.replace("/", "_").replace("\\", "_").replace("..", "_")
            out_name = f"{quote_date.isoformat()}__{ts}__{safe_name}"
            out_path = UPLOAD_DIR / out_name
            pil_img.save(out_path)

            # OCR + parse
            ocr_text = ocr_image(processed)
            quotes = parse_ocr_text_to_quotes(ocr_text)

            # store upload
            upload_id = insert_upload(
                upload_date=quote_date.isoformat(),
                label=label.strip() if label else None,
                notes=notes.strip() if notes else None,
                filename=out_name,
                filepath=str(out_path),
                mimetype=getattr(file, "type", None),
                uploaded_at=datetime.now().isoformat(timespec="seconds"),
                ocr_text=ocr_text
            )

            # store quotes
            if quotes:
                insert_quotes(upload_id=upload_id, quote_date=quote_date.isoformat(), quotes=quotes)

            st.success(f"Saved upload + extracted **{len(quotes)}** rows into dataset ✅")

            st.markdown("#### Extracted rows preview")
            if len(quotes) == 0:
                st.warning("No rows were extracted. Try increasing crop-right (0.30–0.38) and upscale (2.5–3.0).")
                with st.expander("Show OCR raw text (debug)"):
                    st.code(ocr_text[:12000])
            else:
                df_preview = pd.DataFrame(quotes)
                df_preview["volume"] = pd.to_numeric(df_preview.get("volume"), errors="coerce")
                df_preview["last_price"] = pd.to_numeric(df_preview.get("last_price"), errors="coerce")
                df_preview = df_preview.sort_values(["volume", "symbol"], ascending=[False, True])
                st.dataframe(df_preview, use_container_width=True)

                with st.expander("Show OCR raw text"):
                    st.code(ocr_text[:12000])

# ----------------------------
# Dashboard
# ----------------------------
with tab_dashboard:
    st.subheader("Market Dashboard (from your screenshot dataset)")

    dates_df = pd.read_sql_query("SELECT DISTINCT quote_date FROM market_quotes ORDER BY quote_date DESC", CONN)
    available_dates = dates_df["quote_date"].tolist() if not dates_df.empty else []

    if not available_dates:
        st.info("No dataset yet. Go to **Upload + Extract** and upload your first screenshot.")
    else:
        c0, c1, c2 = st.columns([1, 1, 1])
        with c0:
            mode = st.selectbox("View", ["Latest date", "Pick a date"])
        with c1:
            pick_date = st.selectbox("Date", available_dates, disabled=(mode == "Latest date"))
        with c2:
            symbol_filter = st.text_input("Filter symbol (optional)", placeholder="e.g., SCBK")

        date_to_use = available_dates[0] if mode == "Latest date" else pick_date

        quotes_df = fetch_quotes(date_filter=date_to_use, symbol=symbol_filter if symbol_filter else None)
        if quotes_df.empty:
            st.warning("No rows match your filters.")
        else:
            quotes_df = quotes_df.copy()
            quotes_df["volume"] = pd.to_numeric(quotes_df["volume"], errors="coerce")
            quotes_df["last_price"] = pd.to_numeric(quotes_df["last_price"], errors="coerce")
            quotes_df["notional"] = quotes_df["volume"].fillna(0) * quotes_df["last_price"].fillna(0)

            st.markdown(f"### Dataset date: **{date_to_use}**  | Rows: **{len(quotes_df)}**")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Volume", f"{int(quotes_df['volume'].fillna(0).sum()):,}")
            m2.metric("Total Notional (approx)", f"{float(quotes_df['notional'].fillna(0).sum()):,.2f}")
            m3.metric("Unique Symbols", f"{quotes_df['symbol'].nunique():,}")
            m4.metric("Uploads used", f"{quotes_df['upload_id'].nunique():,}")

            st.markdown("### Full table")
            st.dataframe(
                quotes_df[["symbol", "last_price", "volume", "notional", "direction", "quote_date", "upload_id"]]
                .sort_values(["notional", "volume"], ascending=False),
                use_container_width=True
            )

            st.markdown("### Top by Volume")
            st.dataframe(
                quotes_df[["symbol", "last_price", "volume"]]
                .sort_values("volume", ascending=False)
                .head(25),
                use_container_width=True
            )

            st.markdown("### Top by Notional (Volume × Price)")
            st.dataframe(
                quotes_df[["symbol", "last_price", "volume", "notional"]]
                .sort_values("notional", ascending=False)
                .head(25),
                use_container_width=True
            )

# ----------------------------
# Downloads
# ----------------------------
with tab_downloads:
    st.subheader("Download your dataset")

    uploads_df = fetch_uploads(upload_date=None, limit=100000)
    quotes_all = fetch_quotes(date_filter=None, symbol=None, limit=200000)

    st.write(f"Uploads stored: **{len(uploads_df)}**")
    st.write(f"Quote rows stored: **{len(quotes_all)}**")

    if not quotes_all.empty:
        st.download_button(
            "Download quotes.csv",
            data=quotes_all.to_csv(index=False).encode("utf-8"),
            file_name="quotes.csv",
            mime="text/csv"
        )

    if not uploads_df.empty:
        st.download_button(
            "Download uploads.csv",
            data=uploads_df.to_csv(index=False).encode("utf-8"),
            file_name="uploads.csv",
            mime="text/csv"
        )

    if (not uploads_df.empty) and (not quotes_all.empty):
        zip_bytes = build_zip_bytes(quotes_all, uploads_df)
        st.download_button(
            "Download FULL export (ZIP: images + dataset)",
            data=zip_bytes,
            file_name="nse_screenshot_dataset_export.zip",
            mime="application/zip"
        )

    st.divider()
    st.markdown("#### Storage note (Streamlit Cloud)")
    st.info(
        "Streamlit Cloud storage can reset on rebuild/restart. "
        "Use the downloads above to keep a permanent copy of your dataset."
    )
