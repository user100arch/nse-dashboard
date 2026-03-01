# main.py
# ------------------------------------------------------------
# Faida/NSE Watchlist Screenshot → OCR → Dataset + Dashboard
# Streamlit Cloud SAFE: no matplotlib, robust DB migration, multi-upload
# ------------------------------------------------------------
# requirements.txt:
#   streamlit
#   pandas
#   pillow
#   numpy
#   pytesseract
#   opencv-python-headless
#
# packages.txt:
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
st.caption("Upload Faida/NSE watchlist screenshots → extract Symbol/Last Price/Last Qty/Bid Qty/Bid Price → dataset + dashboard + clean CSV.")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("System status")
    st.write(f"pytesseract: {'✅' if PYTESSERACT_OK else '❌'}")
    st.write(f"tesseract: {'✅ ' + str(TESS_BIN) if TESS_BIN else '❌'}")
    st.write(f"OCR ready: {'✅' if OCR_READY else '❌'}")

    st.divider()
    st.caption("Admin")
    if st.button("⚠️ Reset database"):
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        st.success("Database deleted. Refresh the page.")
        st.stop()

# ----------------------------
# DB (with migration safety)
# ----------------------------
def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    # Create uploads table (new schema)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quote_date TEXT,
            upload_date TEXT,
            label TEXT,
            notes TEXT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            mimetype TEXT,
            uploaded_at TEXT NOT NULL,
            ocr_text TEXT
        )
    """)

    # If an old DB exists without quote_date filled, backfill from upload_date
    cols = [r[1] for r in conn.execute("PRAGMA table_info(uploads)").fetchall()]
    if "quote_date" not in cols:
        conn.execute("ALTER TABLE uploads ADD COLUMN quote_date TEXT")
        conn.commit()

    # Backfill quote_date from upload_date if needed
    try:
        conn.execute("UPDATE uploads SET quote_date = upload_date WHERE (quote_date IS NULL OR quote_date = '') AND upload_date IS NOT NULL")
        conn.commit()
    except Exception:
        pass

    # Create index safely (only if column exists)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(uploads)").fetchall()]
    if "quote_date" in cols:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_uploads_quote_date ON uploads(quote_date)")

    # Rows table
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rows_date ON watchlist_rows(quote_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rows_symbol ON watchlist_rows(symbol)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rows_upload ON watchlist_rows(upload_id)")

    conn.commit()
    return conn

CONN = db_conn()

def insert_upload(quote_date: str, label: str | None, notes: str | None,
                  filename: str, filepath: str, mimetype: str | None,
                  uploaded_at: str, ocr_text: str | None) -> int:
    cur = CONN.cursor()
    cur.execute(
        """INSERT INTO uploads
           (quote_date, upload_date, label, notes, filename, filepath, mimetype, uploaded_at, ocr_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (quote_date, quote_date, label, notes, filename, filepath, mimetype, uploaded_at, ocr_text),
    )
    CONN.commit()
    return int(cur.lastrowid)

def insert_rows(upload_id: int, quote_date: str, rows: list[dict]):
    now = datetime.now().isoformat(timespec="seconds")
    cur = CONN.cursor()
    for r in rows:
        cur.execute(
            """INSERT INTO watchlist_rows
               (upload_id, quote_date, symbol, last_price, last_qty, bid_qty, bid_price, raw_line, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                upload_id, quote_date,
                r.get("symbol"),
                r.get("last_price"),
                r.get("last_qty"),
                r.get("bid_qty"),
                r.get("bid_price"),
                r.get("raw_line"),
                now
            )
        )
    CONN.commit()

def fetch_dates() -> list[str]:
    df = pd.read_sql_query("SELECT DISTINCT quote_date FROM watchlist_rows ORDER BY quote_date DESC", CONN)
    return df["quote_date"].tolist() if not df.empty else []

def fetch_uploads(limit: int = 2000) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT id, quote_date, label, notes, filename, filepath, mimetype, uploaded_at FROM uploads ORDER BY id DESC LIMIT ?",
        CONN, params=(limit,)
    )

def fetch_rows(date_filter: str | None = None, upload_id: int | None = None, limit: int = 500000) -> pd.DataFrame:
    q = """SELECT id, upload_id, quote_date, symbol, last_price, last_qty, bid_qty, bid_price, raw_line, created_at
           FROM watchlist_rows WHERE 1=1"""
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
# OCR preprocess + OCR
# ----------------------------
def preprocess_for_ocr(pil_img: Image.Image,
                       crop_left_pct: float, crop_right_pct: float,
                       crop_top_pct: float, crop_bottom_pct: float,
                       upscale: float) -> np.ndarray:
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    x1 = int(w * crop_left_pct)
    x2 = int(w * (1.0 - crop_right_pct))
    y1 = int(h * crop_top_pct)
    y2 = int(h * (1.0 - crop_bottom_pct))

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
    if (th > 200).mean() < 0.5:
        th = cv2.bitwise_not(th)

    return th

def ocr_image(cv_img: np.ndarray) -> str:
    if not OCR_READY:
        raise RuntimeError("OCR not ready. Check requirements.txt and packages.txt.")
    config = r"--oem 3 --psm 6 -c preserve_interword_spaces=1"
    return pytesseract.image_to_string(cv_img, config=config)

# ----------------------------
# Parse rows (Faida watchlist)
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
    Expected columns:
      Symbol | Last Price | Last Qty | Bid Qty | Bid Price

    OCR is noisy; strategy:
      - Find a SYMBOL token near the start (first 4 tokens)
      - Take first 4 numeric tokens after symbol (if present)
    """
    rows = []
    for line in ocr_text.splitlines():
        raw = clean_line(line)
        if not raw:
            continue

        toks = raw.split(" ")
        if len(toks) < 2:
            continue

        sym_idx, symbol = None, None
        for i, t in enumerate(toks[:4]):
            tt = t.upper()
            if re.match(r"^[A-Z0-9\-]{2,12}$", tt) and not NUM_RE.match(tt):
                sym_idx, symbol = i, tt
                break
        if symbol is None:
            continue

        after = toks[sym_idx + 1:]
        nums = [t for t in after if NUM_RE.match(t)]
        if len(nums) < 2:
            continue

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

    # last row per symbol within one screenshot
    dedup = {}
    for r in rows:
        dedup[r["symbol"]] = r
    return list(dedup.values())

# ----------------------------
# Cleaning + exports
# ----------------------------
def clean_rows_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    for c in ["last_price", "bid_price"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ["last_qty", "bid_qty"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["notional_last"] = out["last_price"].fillna(0) * out["last_qty"].fillna(0)
    out["notional_bid"] = out["bid_price"].fillna(0) * out["bid_qty"].fillna(0)

    cols = ["symbol","last_price","last_qty","bid_qty","bid_price",
            "notional_last","notional_bid","quote_date","upload_id"]
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA

    out = out[cols].sort_values(["notional_last","bid_qty","symbol"], ascending=[False, False, True])
    return out

def make_zip(dataset_df: pd.DataFrame, uploads_df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("watchlist_rows.csv", dataset_df.to_csv(index=False))
        z.writestr("uploads.csv", uploads_df.to_csv(index=False))
        z.writestr("watchlist_rows.json", json.dumps(dataset_df.fillna("").to_dict(orient="records"), indent=2))
        z.writestr("uploads.json", json.dumps(uploads_df.fillna("").to_dict(orient="records"), indent=2))

        # include images
        for _, r in uploads_df.iterrows():
            fp = r.get("filepath")
            fn = r.get("filename")
            if isinstance(fp, str) and fp and os.path.exists(fp) and isinstance(fn, str) and fn:
                z.write(fp, arcname=f"screenshots/{fn}")

    bio.seek(0)
    return bio.read()

# ----------------------------
# Tabs
# ----------------------------
tab_upload, tab_dash, tab_download = st.tabs(["📤 Upload (multi)", "📊 Dashboard", "⬇️ Downloads"])

# ----------------------------
# Upload tab
# ----------------------------
with tab_upload:
    st.subheader("Upload multiple screenshots → extract rows → save dataset")

    a, b = st.columns([1, 1])
    with a:
        qdate = st.date_input("Quote date", value=date.today())
        label = st.text_input("Label", value="Faida Watchlist Screenshot")
    with b:
        notes = st.text_area("Notes (optional)", placeholder="Any notes about today's watchlist...")

    files = st.file_uploader(
        "Upload screenshot(s)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )

    st.markdown("### Crop tuning (make preview look like ONLY the watchlist table)")
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        crop_left_pct = st.slider("Crop LEFT (%)", 0.0, 0.7, 0.22, 0.01)
    with s2:
        crop_right_pct = st.slider("Crop RIGHT (%)", 0.0, 0.7, 0.42, 0.01)
    with s3:
        crop_top_pct = st.slider("Crop TOP (%)", 0.0, 0.5, 0.12, 0.01)
    with s4:
        crop_bottom_pct = st.slider("Crop BOTTOM (%)", 0.0, 0.5, 0.18, 0.01)
    with s5:
        upscale = st.slider("Upscale", 1.0, 3.0, 2.7, 0.1)

    if not OCR_READY:
        st.warning("OCR not ready. Fix packages.txt/requirements.txt first (see sidebar).")

    if files:
        st.info(f"Selected files: {len(files)}")

        # preview first image
        first_img = Image.open(files[0])
        preview = preprocess_for_ocr(first_img, crop_left_pct, crop_right_pct, crop_top_pct, crop_bottom_pct, upscale)
        st.image(first_img, caption="Original (first screenshot)", use_container_width=True)
        st.image(preview, caption="OCR Preview (cropped table)", use_container_width=True)

        if st.button("Extract & Save ALL", type="primary", disabled=not OCR_READY):
            results = []
            total_rows = 0

            for f in files:
                pil_img = Image.open(f)
                processed = preprocess_for_ocr(pil_img, crop_left_pct, crop_right_pct, crop_top_pct, crop_bottom_pct, upscale)
                ocr_text = ocr_image(processed)
                rows = parse_watchlist_rows(ocr_text)

                ts = datetime.now().strftime("%H%M%S")
                safe_name = f.name.replace("/", "_").replace("\\", "_").replace("..", "_")
                out_name = f"{qdate.isoformat()}__{ts}__{safe_name}"
                out_path = UPLOAD_DIR / out_name
                pil_img.save(out_path)

                upload_id = insert_upload(
                    quote_date=qdate.isoformat(),
                    label=(label.strip() if label else None),
                    notes=(notes.strip() if notes else None),
                    filename=out_name,
                    filepath=str(out_path),
                    mimetype=getattr(f, "type", None),
                    uploaded_at=datetime.now().isoformat(timespec="seconds"),
                    ocr_text=ocr_text
                )

                if rows:
                    insert_rows(upload_id, qdate.isoformat(), rows)

                total_rows += len(rows)
                results.append((f.name, upload_id, len(rows), rows, ocr_text))

            st.success(f"Done ✅ Uploaded {len(files)} file(s). Extracted total rows: {total_rows}")

            st.markdown("### Per-file extraction results")
            for fname, upload_id, n, rows, ocr_text in results:
                with st.expander(f"{fname} → upload_id={upload_id} → rows={n}", expanded=(n == 0)):
                    if n == 0:
                        st.warning("0 rows extracted. Increase cropping until ONLY the watchlist table is visible in preview.")
                        st.code(ocr_text[:6000])
                    else:
                        dfp = pd.DataFrame(rows)
                        st.dataframe(
                            dfp[["symbol","last_price","last_qty","bid_qty","bid_price"]]
                            .sort_values(["bid_qty","last_qty","symbol"], ascending=[False, False, True]),
                            use_container_width=True
                        )
    else:
        st.info("Upload one or more screenshots to begin.")

# ----------------------------
# Dashboard tab (BI-style using built-in Streamlit charts)
# ----------------------------
with tab_dash:
    st.subheader("Dashboard")

    dates = fetch_dates()
    if not dates:
        st.info("No extracted data yet. Upload screenshots first.")
    else:
        r1, r2, r3, r4 = st.columns([1, 1, 1, 1])
        with r1:
            view_mode = st.selectbox("View", ["Latest snapshot (recommended)", "Pick date", "Pick upload"])
        with r2:
            chosen_date = st.selectbox("Date", dates, disabled=(view_mode == "Pick upload"))
        with r3:
            symbol_query = st.text_input("Symbol search", "")
        with r4:
            top_n = st.slider("Top N", 5, 50, 15)

        if view_mode == "Pick upload":
            up = fetch_uploads(limit=2000)
            if up.empty:
                st.warning("No uploads available.")
                st.stop()
            pick = up.head(75).copy()
            pick["label_show"] = pick["id"].astype(str) + " | " + pick["quote_date"] + " | " + pick["filename"]
            chosen = st.selectbox("Upload", pick["label_show"].tolist())
            chosen_id = int(chosen.split("|")[0].strip())
            raw = fetch_rows(upload_id=chosen_id)
        elif view_mode == "Pick date":
            raw = fetch_rows(date_filter=chosen_date)
        else:
            base = fetch_rows(date_filter=chosen_date)
            if base.empty:
                raw = base
            else:
                base = base.sort_values(["symbol", "upload_id"], ascending=[True, False])
                raw = base.drop_duplicates(subset=["symbol"], keep="first")

        if raw.empty:
            st.warning("Dashboard empty. That means OCR extracted 0 usable rows. Go to Upload tab and adjust cropping.")
            st.stop()

        df = clean_rows_df(raw)

        if symbol_query.strip():
            q = symbol_query.strip().upper()
            df = df[df["symbol"].astype(str).str.contains(q, na=False)]

        if df.empty:
            st.warning("No rows match your filters.")
            st.stop()

        # KPI cards
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Symbols", f"{df['symbol'].nunique():,}")
        k2.metric("Total Last Qty", f"{int(pd.to_numeric(df['last_qty'], errors='coerce').fillna(0).sum()):,}")
        k3.metric("Total Bid Qty", f"{int(pd.to_numeric(df['bid_qty'], errors='coerce').fillna(0).sum()):,}")
        k4.metric("Notional (Last)", f"{float(pd.to_numeric(df['notional_last'], errors='coerce').fillna(0).sum()):,.2f}")
        k5.metric("Notional (Bid)", f"{float(pd.to_numeric(df['notional_bid'], errors='coerce').fillna(0).sum()):,.2f}")

        st.markdown("### Table")
        st.dataframe(df, use_container_width=True)

        # Charts using Streamlit built-ins (no matplotlib needed)
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### Top by Last Qty")
            tmp = df[["symbol","last_qty"]].copy()
            tmp["last_qty"] = pd.to_numeric(tmp["last_qty"], errors="coerce").fillna(0)
            tmp = tmp.sort_values("last_qty", ascending=False).head(top_n).set_index("symbol")
            st.bar_chart(tmp)

        with c2:
            st.markdown("#### Top by Bid Qty")
            tmp = df[["symbol","bid_qty"]].copy()
            tmp["bid_qty"] = pd.to_numeric(tmp["bid_qty"], errors="coerce").fillna(0)
            tmp = tmp.sort_values("bid_qty", ascending=False).head(top_n).set_index("symbol")
            st.bar_chart(tmp)

        with c3:
            st.markdown("#### Top by Notional (Last)")
            tmp = df[["symbol","notional_last"]].copy()
            tmp["notional_last"] = pd.to_numeric(tmp["notional_last"], errors="coerce").fillna(0)
            tmp = tmp.sort_values("notional_last", ascending=False).head(top_n).set_index("symbol")
            st.bar_chart(tmp)

# ----------------------------
# Downloads tab
# ----------------------------
with tab_download:
    st.subheader("Downloads")

    uploads_df = fetch_uploads(limit=200000)
    all_rows = fetch_rows(limit=500000)

    st.write(f"Uploads stored: **{len(uploads_df)}**")
    st.write(f"Extracted rows stored: **{len(all_rows)}**")

    if all_rows.empty:
        st.warning("No data to download yet.")
        st.stop()

    clean_all = clean_rows_df(all_rows)

    st.download_button(
        "Download FULL dataset (watchlist_rows.csv)",
        data=clean_all.to_csv(index=False).encode("utf-8"),
        file_name="watchlist_rows.csv",
        mime="text/csv"
    )

    # Latest snapshot for latest date
    dates = fetch_dates()
    if dates:
        latest_date = dates[0]
        base = fetch_rows(date_filter=latest_date)
        if not base.empty:
            base = base.sort_values(["symbol","upload_id"], ascending=[True, False]).drop_duplicates("symbol", keep="first")
            snap = clean_rows_df(base)
            st.download_button(
                f"Download latest snapshot ({latest_date})",
                data=snap.to_csv(index=False).encode("utf-8"),
                file_name=f"watchlist_snapshot_{latest_date}.csv",
                mime="text/csv"
            )

    zip_bytes = make_zip(clean_all, uploads_df)
    st.download_button(
        "Download FULL export ZIP (dataset + uploads + screenshots)",
        data=zip_bytes,
        file_name="faida_watchlist_export.zip",
        mime="application/zip"
    )

    st.info("If CSV looks empty, it means OCR extracted 0 rows. Fix by adjusting crop sliders on Upload tab.")
