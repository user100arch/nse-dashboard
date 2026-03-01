# main.py
# Faida / NSE screenshot -> OCR -> dataset (correct columns) -> dashboard
#
# requirements.txt:
# streamlit
# pandas
# pillow
# numpy
# pytesseract
# opencv-python-headless
#
# packages.txt:
# tesseract-ocr
# tesseract-ocr-eng

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
OCR_READY = bool(TESS_BIN) and PYTESSERACT_OK

# ----------------------------
# App config
# ----------------------------
APP_TITLE = "NSE Screenshot → Dataset + Dashboard (Faida Watchlist)"
DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "screenshots"
DB_PATH = DATA_DIR / "app.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Upload Faida/NSE watchlist screenshots → extract Symbol/Last Price/Last Qty/Bid Qty/Bid Price → dataset + dashboard.")

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
    st.write("---")
    st.write("If extraction is poor, increase crop sliders and upscale.")

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

    # Correct schema for the Faida watchlist table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist_rows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER NOT NULL,
            quote_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            last_price REAL,
            last_qty INTEGER,
            bid_qty INTEGER,
            bid_price REAL,
            raw_line TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(upload_id) REFERENCES uploads(id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_date ON watchlist_rows(quote_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_symbol ON watchlist_rows(symbol)")

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

def fetch_uploads(limit: int = 1000) -> pd.DataFrame:
    cur = CONN.cursor()
    cur.execute(
        """SELECT id, upload_date, label, notes, filename, filepath, mimetype, uploaded_at, ocr_text
           FROM uploads ORDER BY id DESC LIMIT ?""",
        (limit,),
    )
    rows = cur.fetchall()
    cols = ["id", "upload_date", "label", "notes", "filename", "filepath", "mimetype", "uploaded_at", "ocr_text"]
    return pd.DataFrame(rows, columns=cols)

def insert_watchlist_rows(upload_id: int, quote_date: str, rows: list[dict]):
    now = datetime.now().isoformat(timespec="seconds")
    cur = CONN.cursor()
    for r in rows:
        cur.execute(
            """INSERT INTO watchlist_rows
               (upload_id, quote_date, symbol, last_price, last_qty, bid_qty, bid_price, raw_line, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                upload_id,
                quote_date,
                r.get("symbol"),
                r.get("last_price"),
                r.get("last_qty"),
                r.get("bid_qty"),
                r.get("bid_price"),
                r.get("raw_line"),
                now,
            )
        )
    CONN.commit()

def fetch_watchlist(date_filter: str | None = None, symbol: str | None = None, limit: int = 100000) -> pd.DataFrame:
    cur = CONN.cursor()
    q = """SELECT id, upload_id, quote_date, symbol, last_price, last_qty, bid_qty, bid_price, raw_line, created_at
           FROM watchlist_rows WHERE 1=1"""
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
    cols = ["id","upload_id","quote_date","symbol","last_price","last_qty","bid_qty","bid_price","raw_line","created_at"]
    return pd.DataFrame(rows, columns=cols)

# ----------------------------
# OCR + preprocessing
# ----------------------------
def preprocess_for_ocr(
    pil_img: Image.Image,
    crop_left_pct: float = 0.20,
    crop_right_pct: float = 0.35,
    crop_top_pct: float = 0.10,
    crop_bottom_pct: float = 0.10,
    upscale: float = 2.5,
) -> np.ndarray:
    """
    For full-screen Faida screenshots, we need to crop away:
      - left sidebar
      - right panels/buttons
      - top nav
      - bottom panels
    Then enhance for OCR.
    """
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    x1 = int(w * crop_left_pct)
    x2 = int(w * (1.0 - crop_right_pct))
    y1 = int(h * crop_top_pct)
    y2 = int(h * (1.0 - crop_bottom_pct))

    # guard against bad crops
    x1 = max(0, min(x1, w - 2))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 2))
    y2 = max(y1 + 1, min(y2, h))

    cropped = img[y1:y2, x1:x2]

    if upscale and upscale != 1.0:
        cropped = cv2.resize(cropped, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    # Otsu threshold
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # auto-invert if needed
    if (th > 200).mean() < 0.5:
        th = cv2.bitwise_not(th)

    return th

def ocr_image(cv_img: np.ndarray) -> str:
    if not OCR_READY:
        raise RuntimeError("OCR not ready. Ensure packages.txt (tesseract) + requirements.txt (pytesseract).")

    # Better for table rows + spacing
    config = r"--oem 3 --psm 6 -c preserve_interword_spaces=1"
    return pytesseract.image_to_string(cv_img, config=config)

# ----------------------------
# Parsing (TOKEN-BASED) to avoid wrong columns
# ----------------------------
NUM_RE = re.compile(r"^\d{1,3}(?:,\d{3})*(?:\.\d+)?$")

def parse_float(tok: str):
    try:
        return float(tok.replace(",", ""))
    except Exception:
        return None

def parse_int(tok: str):
    try:
        return int(tok.replace(",", ""))
    except Exception:
        return None

def clean_line(line: str) -> str:
    # remove weird OCR chars and normalize spaces
    line = line.replace("\t", " ")
    line = re.sub(r"\s{2,}", " ", line)
    return line.strip()

def parse_watchlist_rows(ocr_text: str) -> list[dict]:
    """
    Expected logical row (from Faida watchlist):
      SYMBOL  LastPrice  LastQty  BidQty  BidPrice
    We do token-based parsing:
      - first token that looks like a symbol
      - then next 4 numeric tokens in order
    """
    rows = []
    for raw in ocr_text.splitlines():
        raw = clean_line(raw)
        if not raw:
            continue

        tokens = raw.split(" ")
        if len(tokens) < 2:
            continue

        # symbol candidate is usually first token
        symbol = tokens[0].upper()

        # quick filter: symbol must look like ABSA, KPLC-P, SCBK
        if not re.match(r"^[A-Z0-9\-]{2,12}$", symbol):
            continue

        # collect numeric tokens from the rest
        nums = [t for t in tokens[1:] if NUM_RE.match(t)]
        if len(nums) < 2:
            # not enough to be useful
            continue

        # map in expected order
        last_price = parse_float(nums[0]) if len(nums) >= 1 else None
        last_qty   = parse_int(nums[1])   if len(nums) >= 2 else None
        bid_qty    = parse_int(nums[2])   if len(nums) >= 3 else None
        bid_price  = parse_float(nums[3]) if len(nums) >= 4 else None

        rows.append({
            "symbol": symbol,
            "last_price": last_price,
            "last_qty": last_qty,
            "bid_qty": bid_qty,
            "bid_price": bid_price,
            "raw_line": raw
        })

    # Deduplicate by symbol (keep last)
    dedup = {}
    for r in rows:
        dedup[r["symbol"]] = r
    return list(dedup.values())

# ----------------------------
# Export helpers
# ----------------------------
def build_zip_bytes(watch_df: pd.DataFrame, uploads_df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("watchlist_rows.csv", watch_df.to_csv(index=False))
        z.writestr("uploads.csv", uploads_df.to_csv(index=False))
        z.writestr("watchlist_rows.json", json.dumps(watch_df.fillna("").to_dict(orient="records"), indent=2))
        z.writestr("uploads.json", json.dumps(uploads_df.fillna("").to_dict(orient="records"), indent=2))

        for _, r in uploads_df.iterrows():
            fp = r.get("filepath")
            if isinstance(fp, str) and fp and os.path.exists(fp):
                z.write(fp, arcname=f"screenshots/{r['filename']}")

    bio.seek(0)
    return bio.read()

# ----------------------------
# UI
# ----------------------------
tab_upload, tab_dashboard, tab_downloads = st.tabs(
    ["📤 Upload + Extract", "📊 Dashboard", "⬇️ Download Dataset"]
)

# ----------------------------
# Upload + Extract
# ----------------------------
with tab_upload:
    st.subheader("Upload screenshot → Extract watchlist table → Save dataset")

    col1, col2 = st.columns([1, 1])
    with col1:
        quote_date = st.date_input("Quote date", value=date.today())
        label = st.text_input("Label", value="Faida Watchlist Screenshot")
    with col2:
        notes = st.text_area("Notes (optional)", placeholder="Any notes about today's watchlist...")

    file = st.file_uploader("Upload screenshot (PNG/JPG/WEBP)", type=["png", "jpg", "jpeg", "webp"])

    st.markdown("### Crop & OCR tuning (critical for full-screen Faida screenshots)")
    t1, t2, t3, t4, t5 = st.columns(5)
    with t1:
        crop_left_pct = st.slider("Crop LEFT (%)", 0.0, 0.6, 0.20, 0.01)
    with t2:
        crop_right_pct = st.slider("Crop RIGHT (%)", 0.0, 0.6, 0.35, 0.01)
    with t3:
        crop_top_pct = st.slider("Crop TOP (%)", 0.0, 0.4, 0.10, 0.01)
    with t4:
        crop_bottom_pct = st.slider("Crop BOTTOM (%)", 0.0, 0.4, 0.10, 0.01)
    with t5:
        upscale = st.slider("Upscale", 1.0, 3.0, 2.5, 0.1)

    if file:
        pil_img = Image.open(file)
        st.image(pil_img, caption="Uploaded screenshot", use_container_width=True)

        processed = preprocess_for_ocr(
            pil_img,
            crop_left_pct=crop_left_pct,
            crop_right_pct=crop_right_pct,
            crop_top_pct=crop_top_pct,
            crop_bottom_pct=crop_bottom_pct,
            upscale=upscale
        )
        st.image(processed, caption="Preprocessed (cropped) for OCR", use_container_width=True)

        if not OCR_READY:
            st.warning("OCR is not ready on this deployment. Check packages.txt + requirements.txt.")

        if st.button("Extract & Save Dataset", type="primary", disabled=not OCR_READY):
            ts = datetime.now().strftime("%H%M%S")
            safe_name = file.name.replace("/", "_").replace("\\", "_").replace("..", "_")
            out_name = f"{quote_date.isoformat()}__{ts}__{safe_name}"
            out_path = UPLOAD_DIR / out_name
            pil_img.save(out_path)

            ocr_text = ocr_image(processed)
            rows = parse_watchlist_rows(ocr_text)

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

            if rows:
                insert_watchlist_rows(upload_id=upload_id, quote_date=quote_date.isoformat(), rows=rows)

            st.success(f"Saved upload + extracted **{len(rows)}** rows ✅")

            st.markdown("### Extracted rows preview (correct columns)")
            if not rows:
                st.warning("No rows extracted. Increase crop (left/right/top/bottom) until ONLY the watchlist table remains.")
                with st.expander("Show OCR raw text (debug)"):
                    st.code(ocr_text[:12000])
            else:
                df_preview = pd.DataFrame(rows)
                df_preview["last_price"] = pd.to_numeric(df_preview["last_price"], errors="coerce")
                df_preview["last_qty"] = pd.to_numeric(df_preview["last_qty"], errors="coerce")
                df_preview["bid_qty"] = pd.to_numeric(df_preview["bid_qty"], errors="coerce")
                df_preview["bid_price"] = pd.to_numeric(df_preview["bid_price"], errors="coerce")

                df_preview = df_preview.sort_values(["bid_qty", "last_qty", "symbol"], ascending=[False, False, True])
                st.dataframe(df_preview[["symbol","last_price","last_qty","bid_qty","bid_price"]], use_container_width=True)

                with st.expander("Show OCR raw text"):
                    st.code(ocr_text[:12000])

# ----------------------------
# Dashboard
# ----------------------------
with tab_dashboard:
    st.subheader("Dashboard (from extracted dataset)")

    dates_df = pd.read_sql_query("SELECT DISTINCT quote_date FROM watchlist_rows ORDER BY quote_date DESC", CONN)
    available_dates = dates_df["quote_date"].tolist() if not dates_df.empty else []

    if not available_dates:
        st.info("No dataset yet. Upload a screenshot in **Upload + Extract**.")
    else:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            mode = st.selectbox("View", ["Latest date", "Pick a date"])
        with c2:
            pick_date = st.selectbox("Date", available_dates, disabled=(mode == "Latest date"))
        with c3:
            symbol_filter = st.text_input("Filter symbol (optional)", placeholder="e.g., ABSA")

        date_to_use = available_dates[0] if mode == "Latest date" else pick_date

        df = fetch_watchlist(date_filter=date_to_use, symbol=symbol_filter if symbol_filter else None)
        if df.empty:
            st.warning("No rows match your filters.")
        else:
            df = df.copy()
            df["last_price"] = pd.to_numeric(df["last_price"], errors="coerce")
            df["last_qty"] = pd.to_numeric(df["last_qty"], errors="coerce")
            df["bid_qty"] = pd.to_numeric(df["bid_qty"], errors="coerce")
            df["bid_price"] = pd.to_numeric(df["bid_price"], errors="coerce")

            df["notional_last"] = df["last_price"].fillna(0) * df["last_qty"].fillna(0)
            df["notional_bid"] = df["bid_price"].fillna(0) * df["bid_qty"].fillna(0)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Symbols", f"{df['symbol'].nunique():,}")
            m2.metric("Total Last Qty", f"{int(df['last_qty'].fillna(0).sum()):,}")
            m3.metric("Total Bid Qty", f"{int(df['bid_qty'].fillna(0).sum()):,}")
            m4.metric("Uploads used", f"{df['upload_id'].nunique():,}")

            st.markdown("### Full Watchlist Table")
            st.dataframe(
                df[["symbol","last_price","last_qty","bid_qty","bid_price","notional_last","notional_bid"]]
                .sort_values(["notional_bid","bid_qty"], ascending=False),
                use_container_width=True
            )

            st.markdown("### Top Bid Qty")
            st.dataframe(
                df[["symbol","bid_qty","bid_price","notional_bid"]]
                .sort_values("bid_qty", ascending=False)
                .head(25),
                use_container_width=True
            )

            st.markdown("### Top Last Qty (Most traded in your watchlist)")
            st.dataframe(
                df[["symbol","last_qty","last_price","notional_last"]]
                .sort_values("last_qty", ascending=False)
                .head(25),
                use_container_width=True
            )

# ----------------------------
# Downloads
# ----------------------------
with tab_downloads:
    st.subheader("Download dataset (correct column names)")

    uploads_df = fetch_uploads(limit=100000)
    watch_df = fetch_watchlist(date_filter=None, symbol=None, limit=200000)

    st.write(f"Uploads stored: **{len(uploads_df)}**")
    st.write(f"Extracted watchlist rows stored: **{len(watch_df)}**")

    if not watch_df.empty:
        st.download_button(
            "Download watchlist_rows.csv",
            data=watch_df.to_csv(index=False).encode("utf-8"),
            file_name="watchlist_rows.csv",
            mime="text/csv"
        )

    if not uploads_df.empty:
        st.download_button(
            "Download uploads.csv",
            data=uploads_df.to_csv(index=False).encode("utf-8"),
            file_name="uploads.csv",
            mime="text/csv"
        )

    if (not uploads_df.empty) and (not watch_df.empty):
        zip_bytes = build_zip_bytes(watch_df, uploads_df)
        st.download_button(
            "Download FULL export (ZIP: images + dataset)",
            data=zip_bytes,
            file_name="faida_watchlist_export.zip",
            mime="application/zip"
        )

    st.divider()
    st.info(
        "Streamlit Cloud storage can reset on rebuild/restart. "
        "Download your dataset regularly (CSV/ZIP) to keep a permanent copy."
    )
