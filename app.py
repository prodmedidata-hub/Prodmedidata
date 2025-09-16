import os
import io
import json
import time
import pandas as pd
import streamlit as st
from rapidfuzz import process

# ------------------ Paths ------------------
MEMORY_FOLDER = "memory"
MEMORY_FILE = os.path.join(MEMORY_FOLDER, "memory_filled.xlsx")
os.makedirs(MEMORY_FOLDER, exist_ok=True)

# ------------------ Keyword Rules ------------------
keyword_rules = {
    "sterile": {"SCI_UNSPSC Code": "42132207", "SCI_UNSPSC Description": "Non-latex surgical gloves"},
    "non-sterile": {"SCI_UNSPSC Code": "42141504", "SCI_UNSPSC Description": "Swabstick"},
    "catheter": {"SCI_UNSPSC Code": "42142744", "SCI_UNSPSC Description": "Foley catheter"},
    "syringe": {"SCI_UNSPSC Code": "42142523", "SCI_UNSPSC Description": "Bone marrow aspiration systems"},
    "scalpel": {"SCI_UNSPSC Code": "42291613", "SCI_UNSPSC Description": "Surgical scalpels or knives"}
}

def apply_keyword_rules(desc, sci_cols):
    desc = str(desc).lower()
    filled = {}
    for kw, mapping in keyword_rules.items():
        if kw in desc:
            for col, val in mapping.items():
                if col in sci_cols:
                    filled[col] = val
    return filled

# ------------------ Memory Functions ------------------
def load_memory():
    if os.path.exists(MEMORY_FILE):
        return pd.read_excel(MEMORY_FILE)
    else:
        return pd.DataFrame(columns=["Product - Description"])

def save_to_memory(new_rows):
    if not new_rows.empty:
        if os.path.exists(MEMORY_FILE):
            old = pd.read_excel(MEMORY_FILE)
            combined = pd.concat([old, new_rows], ignore_index=True)
        else:
            combined = new_rows
        combined.drop_duplicates(subset=["Product - Description"], keep="last", inplace=True)
        combined.to_excel(MEMORY_FILE, index=False)

# ------------------ QC Check Function ------------------
def qc_check(df, sci_cols):
    """
    Simple QC check: 
    - Count blanks 
    - Detect duplicates 
    - Consistency check on SCI codes
    """
    qc_report = {}
    qc_report["Total Rows"] = len(df)
    qc_report["Missing Fields"] = sum(df[sci_cols].isna().any(axis=1))
    qc_report["Duplicate Descriptions"] = df["Product - Description"].duplicated().sum()

    # Consistency check: one description → multiple UNSPSC?
    inconsistent = 0
    grouped = df.groupby("Product - Description")
    for name, group in grouped:
        if len(group[sci_cols].dropna().drop_duplicates()) > 1:
            inconsistent += 1
    qc_report["Inconsistent Entries"] = inconsistent
    return qc_report

# ------------------ AI Batch Helper ------------------
def call_ai_batch(rows, sci_cols, batch_size=50, max_retries=3):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return [{} for _ in rows]

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    results = []

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        prompt = f"""
        You are an expert in SCI medical catalog data enrichment.
        Fill ONLY these columns: {sci_cols}

        Here is a batch of product rows:
        {json.dumps(batch, indent=2)}

        Return a JSON list, one item per row, same order.
        """

        retries = 0
        batch_filled = False
        while retries < max_retries:
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a precise assistant for filling SCI attributes. Return ONLY valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                content = resp.choices[0].message.content.strip()
                if content.startswith("```json"):
                    content = content[7:-3]
                batch_result = json.loads(content)
                results.extend(batch_result)
                batch_filled = True
                break
            except Exception as e:
                retries += 1
                wait_time = 10 * retries
                st.warning(f"⚠️ Rate limit hit! Waiting {wait_time}s before retry {retries}/{max_retries}...")
                time.sleep(wait_time)

        if not batch_filled:
            st.error("❌ Max retries reached. Filling rows with 'Unfilled'.")
            results.extend([{col: "Unfilled" for col in sci_cols} for _ in batch])

    return results

# ------------------ Streamlit App ------------------
st.set_page_config(page_title="SCI Auto-Fill Bot (Batch AI)", layout="wide")

st.markdown("## 🤖 SCI Auto-Fill Bot (Batch AI + Rules + Memory + QC + KPI)")
st.caption("Rules + Memory + QC Check + AI (batch mode, retry on rate-limit, safe fallback).")

target_file = st.file_uploader("📂 Upload Target File", type=["xlsx"])

if target_file:
    target_df = pd.read_excel(target_file, sheet_name=0)
    sci_cols = [c for c in target_df.columns if str(c).startswith("SCI_")]

    for col in sci_cols:
        target_df[col] = target_df[col].astype("object")

    df_out = target_df.copy()
    df_out["Fill_Source"] = ""
    df_out["Confidence"] = ""

    mem_df = load_memory()
    new_memory_rows = []
    counts = {"RuleDict":0, "Memory":0, "AI":0, "Unfilled":0}

    rows_for_ai = []
    row_indices = []

    # Step 1: Rules + Memory
    for idx, row in df_out.iterrows():
        desc = row.get("Product - Description", "")
        filled = apply_keyword_rules(desc, sci_cols)
        if filled:
            for col, val in filled.items():
                df_out.at[idx, col] = val
            df_out.at[idx, "Fill_Source"] = "RuleDict"
            df_out.at[idx, "Confidence"] = 1.0
            counts["RuleDict"] += 1
            continue

        mem_row = mem_df[mem_df["Product - Description"].str.lower() == str(desc).lower()]
        if not mem_row.empty:
            for col in sci_cols:
                if col in mem_row.columns:
                    df_out.at[idx, col] = mem_row.iloc[0][col]
            df_out.at[idx, "Fill_Source"] = "Memory"
            df_out.at[idx, "Confidence"] = 0.9
            counts["Memory"] += 1
            continue

        row_dict = {
            "Description": desc,
            "Vendor": row.get("Vendor - Name",""),
            "Manufacturer": row.get("Product - Manufacturer Name (Complete)",""),
            "Catalog": row.get("Product - Manufacturer Catalog Number",""),
            "Country": row.get("Country","")
        }
        rows_for_ai.append(row_dict)
        row_indices.append(idx)

    # Step 2: AI Fallback
    if rows_for_ai:
        st.info(f"🤖 Sending {len(rows_for_ai)} rows to AI in batches...")
        ai_results = call_ai_batch(rows_for_ai, sci_cols)
        for idx, ai_dict in zip(row_indices, ai_results):
            if isinstance(ai_dict, dict):
                for col in sci_cols:
                    val = ai_dict.get(col, None)
                    if val:
                        df_out.at[idx, col] = val
                if all(str(df_out.at[idx, col]).strip() != "" for col in sci_cols):
                    df_out.at[idx, "Fill_Source"] = "AI"
                    df_out.at[idx, "Confidence"] = 0.7
                    counts["AI"] += 1
                else:
                    df_out.at[idx, "Fill_Source"] = "Unfilled"
                    df_out.at[idx, "Confidence"] = 0.0
                    counts["Unfilled"] += 1

    # Save new memory
    for idx, row in df_out.iterrows():
        if df_out.at[idx, "Fill_Source"] in ["RuleDict", "AI"]:
            mem_entry = {col: df_out.at[idx, col] for col in sci_cols}
            mem_entry["Product - Description"] = row.get("Product - Description", "")
            new_memory_rows.append(mem_entry)
    if new_memory_rows:
        save_to_memory(pd.DataFrame(new_memory_rows))

    # ------------------ KPI Metrics ------------------
    total_rows = len(df_out)
    st.subheader("📊 KPI Dashboard")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("RuleDict", f"{counts['RuleDict']} ({counts['RuleDict']/total_rows:.0%})")
    c2.metric("Memory", f"{counts['Memory']} ({counts['Memory']/total_rows:.0%})")
    c3.metric("AI Fallback", f"{counts['AI']} ({counts['AI']/total_rows:.0%})")
    c4.metric("Unfilled", f"{counts['Unfilled']} ({counts['Unfilled']/total_rows:.0%})")
    c5.metric("New Memory Entries", len(new_memory_rows))

    # ------------------ QC Report ------------------
    qc_report = qc_check(df_out, sci_cols)
    st.subheader("🛠 QC Report")
    st.json(qc_report)

    # ------------------ Data Preview ------------------
    st.subheader("📋 Preview Completed Data")
    st.dataframe(df_out.head(20))

    # Save Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="Completed")
        target_df.to_excel(writer, index=False, sheet_name="Original_Input")
    buffer.seek(0)

    st.download_button(
        "📥 Download Completed Excel",
        data=buffer,
        file_name="SCI_Filled_BatchAI.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
