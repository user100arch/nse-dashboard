# main.py — NSE/Faida Watchlist Screenshot Extractor (Multi-upload + Clean Dataset)
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
st.caption("Upload multiple Faida/NSE watchlist screenshots → extract rows → build dataset → dashboard + clean CSV exports.")

with st.sidebar:
    st.header("System status")
    if PYTESSERACT_OK:
        st.success("pytesseract: OK")
    else:
        st.error("pytesseract: NOT installed (requirements.txt)")
    if TESS_BIN:
        st.success(f"Tesseract: {TESS_BIN}")
    else:
        st.error("Tesseract NOT found (packages.txt)")
    st.write("---")
    st.caption("Tip: Crop until ONLY the watchlist table remains.")

# ----------------------------
# DB helpers
# ----------------------------
def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quote_date TEXT NOT NULL,
            label TEXT,
            notes TEXT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            mimetype TEXT,
            uploaded_at TEXT NOT NULL,
            ocr_text TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_uploads_quote_date ON uploads(quote_date)")

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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_upload ON watchlist_rows(upload_id)")

    conn.commit()
    return conn

CONN = db_conn()

def insert_upload(quote_date: str, label: str | None, notes: str | None,
                  filename: str, filepath: str, mimetype: str | None,
                  uploaded_at: str, ocr_text: str | None) -> int:
    cur = CONN.cursor()
    cur.execute(
        """INSERT INTO uploads
           (quote_date, label, notes, filename, filepath, mimetype, uploaded_at, ocr_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (quote_date, label, notes, filename, filepath, mimetype, uploaded_at, ocr_text),
    )
    CONN.commit()
    return int(cur.lastrowid)

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

def fetch_dates() -> list[str]:
    df = pd.read_sql_query("SELECT DISTINCT quote_date FROM watchlist_rows ORDER BY quote_date DESC", CONN)
    return df["quote_date"].tolist() if not df.empty else []

def fetch_uploads(limit: int = 2000) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM uploads ORDER BY id DESC LIMIT ?",
        CONN,
        params=(limit,)
    )

def fetch_rows(date_filter: str | None = None, upload_id: int | None = None, limit: int = 200000) -> pd.DataFrame:
    q = "SELECT * FROM watchlist_rows WHERE 1=1"
    params = []
    if date_filter:
        q += " AND quote_date = ?"
        params.append(date_filter)
    if upload_id:
        q += " AND upload_id = ?"
        params.append(upload_id)
    q += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    return pd.read_sql_query(q, CONN, params=tuple(params))

# ----------------------------
# Preprocess + OCR
# ----------------------------
def preprocess_for_ocr(
    pil_img: Image.Image,
    crop_left_pct: float,
    crop_right_pct: float,
    crop_top_pct: float,
    crop_bottom_pct: float,
    upscale: float,
) -> np.ndarray:
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    x1 = int(w * crop_left_pct)
    x2 = int(w * (1.0 - crop_right_pct))
    y1 = int(h * crop_top_pct)
    y2 = int(h * (1.0 - crop_bottom_pct))

    # safe guards
    x1 = max(0, min(x1, w - 2))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 2))
    y2 = max(y1 + 1, min(y2, h))

    cropped = img[y1:y2, x1:x2]

    if upscale and upscale != 1.0:
        cropped = cv2.resize(cropped, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # auto-invert if needed
    if (th > 200).mean() < 0.5:
        th = cv2.bitwise_not(th)

    return th

def ocr_image(cv_img: np.ndarray) -> str:
    if not OCR_READY:
        raise RuntimeError("OCR not ready. Ensure packages.txt + requirements.txt are correct.")
    config = r"--oem 3 --psm 6 -c preserve_interword_spaces=1"
    return pytesseract.image_to_string(cv_img, config=config)

# ----------------------------
# Parsing (robust)
# ----------------------------
NUM_RE = re.compile(r"^\d{1,3}(?:,\d{3})*(?:\.\d+)?$")

def to_float(tok: str):
    try:
        return float(tok.replace(",", ""))
    except Exception:
        return None

def to_int(tok: str):
    try:
        return int(tok.replace(",", ""))
    except Exception:
        return None

def clean_line(s: str) -> str:
    s = s.replace("\t", " ")
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def parse_watchlist_rows(ocr_text: str) -> list[dict]:
    """
    Attempts to extract rows from Faida watchlist table:
      SYMBOL  LastPrice  LastQty  BidQty  BidPrice
    OCR may include arrows/icons or extra tokens.
    Strategy:
      - Find first token that looks like SYMBOL
      - Then take the first 4 numeric tokens AFTER the symbol (anywhere in that line)
    """
    rows = []
    for line in ocr_text.splitlines():
        raw = clean_line(line)
        if not raw:
            continue
        toks = raw.split(" ")

        # find a symbol token in the line (not always index 0 with OCR noise)
        sym_idx = None
        symbol = None
        for i, t in enumerate(toks[:4]):  # symbol usually near the start
            tt = t.upper()
            if re.match(r"^[A-Z0-9\-]{2,12}$", tt) and not NUM_RE.match(tt):
                sym_idx = i
                symbol = tt
                break
        if symbol is None:
            continue

        after = toks[sym_idx + 1:]
        nums = [t for t in after if NUM_RE.match(t)]
        if len(nums) < 2:
            continue  # not enough data to be a row

        last_price = to_float(nums[0]) if len(nums) >= 1 else None
        last_qty   = to_int(nums[1])   if len(nums) >= 2 else None
        bid_qty    = to_int(nums[2])   if len(nums) >= 3 else None
        bid_price  = to_float(nums[3]) if len(nums) >= 4 else None

        rows.append({
            "symbol": symbol,
            "last_price": last_price,
            "last_qty": last_qty,
            "bid_qty": bid_qty,
            "bid_price": bid_price,
            "raw_line": raw
        })

    # keep last occurrence per symbol for a "snapshot"
    dedup = {}
    for r in rows:
        dedup[r["symbol"]] = r
    return list(dedup.values())

# ----------------------------
# Exports (clean)
# ----------------------------
def clean_rows_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    # enforce types
    for c in ["last_price", "bid_price"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ["last_qty", "bid_qty"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    # keep clean column order
    cols = ["symbol", "last_price", "last_qty", "bid_qty", "bid_price", "quote_date", "upload_id"]
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[cols]

    # sort best-first
    out["bid_qty_sort"] = out["bid_qty"].fillna(0).astype("float")
    out["last_qty_sort"] = out["last_qty"].fillna(0).astype("float")
    out = out.sort_values(["bid_qty_sort", "last_qty_sort", "symbol"], ascending=[False, False, True]).drop(columns=["bid_qty_sort","last_qty_sort"])
    return out

def make_zip(dataset_df: pd.DataFrame, uploads_df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("watchlist_rows.csv", dataset_df.to_csv(index=False))
        z.writestr("uploads.csv", uploads_df.to_csv(index=False))
        z.writestr("watchlist_rows.json", json.dumps(dataset_df.fillna("").to_dict(orient="records"), indent=2))
        z.writestr("uploads.json", json.dumps(uploads_df.fillna("").to_dict(orient="records"), indent=2))

        # include images if present
        for _, r in uploads_df.iterrows():
            fp = r.get("filepath")
            if isinstance(fp, str) and fp and os.path.exists(fp):
                z.write(fp, arcname=f"screenshots/{r['filename']}")
    bio.seek(0)
    return bio.read()

# ----------------------------
# UI
# ----------------------------
tab_upload, tab_dash, tab_download = st.tabs(["📤 Upload (multi)", "📊 Dashboard", "⬇️ Downloads"])

with tab_upload:
    st.subheader("Upload multiple screenshots and extract rows")

    c1, c2 = st.columns([1, 1])
    with c1:
        quote_date = st.date_input("Quote date", value=date.today())
        label = st.text_input("Label", value="Faida Watchlist Screenshot")
    with c2:
        notes = st.text_area("Notes (optional)", placeholder="Anything about today...")

    # ✅ multi upload enabled
    files = st.file_uploader(
        "Upload screenshot(s)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )

    st.markdown("### Crop & OCR tuning (make the preview look like ONLY the watchlist table)")
    t1, t2, t3, t4, t5 = st.columns(5)
    with t1:
        crop_left_pct = st.slider("Crop LEFT (%)", 0.0, 0.7, 0.22, 0.01)
    with t2:
        crop_right_pct = st.slider("Crop RIGHT (%)", 0.0, 0.7, 0.42, 0.01)
    with t3:
        crop_top_pct = st.slider("Crop TOP (%)", 0.0, 0.5, 0.12, 0.01)
    with t4:
        crop_bottom_pct = st.slider("Crop BOTTOM (%)", 0.0, 0.5, 0.18, 0.01)
    with t5:
        upscale = st.slider("Upscale", 1.0, 3.0, 2.7, 0.1)

    if not OCR_READY:
        st.warning("OCR not ready. Fix packages.txt/requirements.txt first (see sidebar).")

    if files:
        st.info(f"Selected files: {len(files)}")

        # Show preview of the first file's crop result (so you can tune crop once)
        first = Image.open(files[0])
        processed_preview = preprocess_for_ocr(
            first, crop_left_pct, crop_right_pct, crop_top_pct, crop_bottom_pct, upscale
        )
        st.image(first, caption="Original (first file)", use_container_width=True)
        st.image(processed_preview, caption="OCR crop preview (first file)", use_container_width=True)

        if st.button("Extract & Save ALL", type="primary", disabled=not OCR_READY):
            total_rows = 0
            per_file_results = []

            for f in files:
                pil_img = Image.open(f)

                processed = preprocess_for_ocr(
                    pil_img, crop_left_pct, crop_right_pct, crop_top_pct, crop_bottom_pct, upscale
                )
                ocr_text = ocr_image(processed)
                rows = parse_watchlist_rows(ocr_text)

                # save image
                ts = datetime.now().strftime("%H%M%S")
                safe_name = f.name.replace("/", "_").replace("\\", "_").replace("..", "_")
                out_name = f"{quote_date.isoformat()}__{ts}__{safe_name}"
                out_path = UPLOAD_DIR / out_name
                pil_img.save(out_path)

                upload_id = insert_upload(
                    quote_date=quote_date.isoformat(),
                    label=label.strip() if label else None,
                    notes=notes.strip() if notes else None,
                    filename=out_name,
                    filepath=str(out_path),
                    mimetype=getattr(f, "type", None),
                    uploaded_at=datetime.now().isoformat(timespec="seconds"),
                    ocr_text=ocr_text
                )

                if rows:
                    insert_watchlist_rows(upload_id, quote_date.isoformat(), rows)

                total_rows += len(rows)
                per_file_results.append((f.name, upload_id, len(rows), ocr_text, rows))

            st.success(f"Done ✅ Files: {len(files)} | Total extracted rows: {total_rows}")

            # Show per-file extraction summaries + preview
            for fname, upload_id, nrows, ocr_text, rows in per_file_results:
                with st.expander(f"{fname} → upload_id={upload_id} → rows={nrows}", expanded=(nrows == 0)):
                    if nrows == 0:
                        st.warning("0 rows extracted from this screenshot. Your crop still includes other panels OR OCR isn't reading the table.")
                        st.caption("Fix by increasing crop left/right/top/bottom until only table remains.")
                        st.code(ocr_text[:6000])
                    else:
                        dfp = pd.DataFrame(rows)
                        st.dataframe(
                            dfp[["symbol","last_price","last_qty","bid_qty","bid_price"]]
                            .sort_values(["bid_qty","last_qty","symbol"], ascending=[False, False, True]),
                            use_container_width=True
                        )

with tab_dash:
    st.subheader("Dashboard")

    dates = fetch_dates()
    if not dates:
        st.info("No extracted data yet. Go to Upload tab and extract at least one screenshot.")
    else:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            view_mode = st.selectbox("View mode", ["Latest snapshot (recommended)", "Pick date", "Pick upload"])
        with c2:
            chosen_date = st.selectbox("Date", dates, disabled=(view_mode == "Pick upload"))
        with c3:
            symbol_filter = st.text_input("Symbol filter", placeholder="e.g., ABSA (optional)")

        if view_mode == "Pick upload":
            uploads = fetch_uploads(limit=2000)
            if uploads.empty:
                st.warning("No uploads found.")
            else:
                # show latest 50 uploads for selection
                pick = uploads.head(50)
                pick["label_show"] = pick["id"].astype(str) + " | " + pick["quote_date"] + " | " + pick["filename"]
                chosen = st.selectbox("Upload", pick["label_show"].tolist())
                chosen_id = int(chosen.split("|")[0].strip())
                df = fetch_rows(upload_id=chosen_id)
        elif view_mode == "Pick date":
            df = fetch_rows(date_filter=chosen_date)
        else:
            # Latest snapshot: take latest upload_id per symbol on that date
            base = fetch_rows(date_filter=chosen_date)
            if base.empty:
                df = base
            else:
                # latest row per symbol by upload_id
                base = base.sort_values(["symbol", "upload_id"], ascending=[True, False])
                df = base.drop_duplicates(subset=["symbol"], keep="first")

        if symbol_filter:
            df = df[df["symbol"].str.upper() == symbol_filter.strip().upper()] if not df.empty else df

        if df.empty:
            st.warning("Dashboard is empty because extraction produced 0 usable rows. Go back and adjust cropping until only the watchlist table remains.")
        else:
            clean = clean_rows_df(df)
            st.dataframe(clean, use_container_width=True)

            # simple KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Symbols", f"{clean['symbol'].nunique():,}")
            k2.metric("Total Last Qty", f"{int(pd.to_numeric(clean['last_qty'], errors='coerce').fillna(0).sum()):,}")
            k3.metric("Total Bid Qty", f"{int(pd.to_numeric(clean['bid_qty'], errors='coerce').fillna(0).sum()):,}")
            k4.metric("Date", chosen_date)

with tab_download:
    st.subheader("Downloads (clean CSV)")

    uploads_df = fetch_uploads(limit=200000)
    all_rows = fetch_rows(limit=500000)

    st.write(f"Uploads stored: **{len(uploads_df)}**")
    st.write(f"Extracted rows stored: **{len(all_rows)}**")

    if all_rows.empty:
        st.warning("No data to download yet (extraction returned 0 rows).")
    else:
        # Clean full dataset export
        clean_all = clean_rows_df(all_rows)

        st.download_button(
            "Download FULL dataset (watchlist_rows.csv)",
            data=clean_all.to_csv(index=False).encode("utf-8"),
            file_name="watchlist_rows.csv",
            mime="text/csv"
        )

        # Latest snapshot export (most useful)
        dates = fetch_dates()
        latest_date = dates[0] if dates else None
        if latest_date:
            base = fetch_rows(date_filter=latest_date)
            if not base.empty:
                base = base.sort_values(["symbol", "upload_id"], ascending=[True, False]).drop_duplicates("symbol", keep="first")
                snap = clean_rows_df(base)

                st.download_button(
                    f"Download latest snapshot ({latest_date})",
                    data=snap.to_csv(index=False).encode("utf-8"),
                    file_name=f"watchlist_snapshot_{latest_date}.csv",
                    mime="text/csv"
                )

        # ZIP export
        zip_bytes = make_zip(clean_all, uploads_df)
        st.download_button(
            "Download FULL export ZIP (dataset + uploads)",
            data=zip_bytes,
            file_name="faida_watchlist_export.zip",
            mime="application/zip"
        )

    st.info(
        "If your CSV looks wrong/empty, it means OCR extracted wrong/zero rows. "
        "Fix by cropping until the OCR preview shows ONLY the watchlist table."
    )
