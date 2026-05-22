from flask import Flask, render_template, request, redirect, url_for
import gradio as gr  
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import csv
import random

flask_app = Flask(__name__)

# =========================================================
# CONFIG
# =========================================================
MAIN_FILE = "katheer_annotated_subset_2.5PRO.csv"
RARE_FILE = "most_frequent.csv"

XML_COLUMN = "extracted_text_full"

SAMPLES_PER_TAG = 10
RANDOM_SEED = 42

# Always rebuild sampled datasets on startup so you do NOT
# accidentally keep old huge files from previous runs.
FORCE_REBUILD = True

PHASE_FILES = {
    "high": "eval_high.csv",
    "medium": "eval_medium.csv",
    "low": "eval_low.csv",
    "rare": "eval_rare.csv",
}

RESULT_FILES = {
    "high": "annotation_results_high.csv",
    "medium": "annotation_results_medium.csv",
    "low": "annotation_results_low.csv",
    "rare": "annotation_results_rare.csv",
}

PHASE_LABELS = {
    "high": "الوسوم الأعلى تكرارًا / High-Frequency Tags",
    "medium": "الوسوم متوسطة التكرار / Medium-Frequency Tags",
    "low": "الوسوم الأقل تكرارًا / Low-Frequency Tags",
    "rare": "مجموعة الوسوم النادرة / Rare-Tag Challenge Set",
}

SKIP_TAGS = {
    "tafsir_section",
    "tafsir_section_block",
    "tafsir_chunk",
    "argument",
}

TAG_LABELS_AR = {
    "quran_verse": "آية قرآنية",
    "hadith": "حديث",
    "isnad": "سند",
    "source": "مصدر",
    "command_and_prohibition": "أمر ونهي",
    "chain_evaluation": "الحكم على السند",
    "narrator_criticism": "نقد الرواة",
    "asbab_al_nuzul": "أسباب النزول",
    "hadith_support": "استشهاد بحديث",
    "opinions_of_scholars": "أقوال العلماء",
    "linguistic_analysis": "تحليل لغوي",
    "qiraat": "قراءات",
    "cross_references": "إحالات",
    "fiqh_implications": "أحكام فقهية",
    "theological_points": "قضايا عقدية",
    "historical_context": "سياق تاريخي",
    "balagha_analysis": "تحليل بلاغي",
    "nasikh_wa_mansukh": "الناسخ والمنسوخ",
    "grammatical_parsing": "إعراب / تحليل نحوي",
    "etymology": "اشتقاق",
    "parabolic_meaning": "معنى تمثيلي",
    "sectarian_perspective": "منظور مذهبي",
    "spiritual_lessons": "فوائد إيمانية",
    "biographies_of_narrators": "تراجم الرواة",
    "ijma_status": "حالة الإجماع",
    "reason_for_revelation_variants": "اختلافات سبب النزول",
    "legal_maxims": "قواعد فقهية",
    "maqasid_al_sharia": "مقاصد الشريعة",
    "variant_interpretations": "اختلافات تفسيرية",
    "minority_opinions": "قول الأقلية",
    "majority_opinions": "قول الجمهور",
    "literal_interpretation": "تفسير حرفي",
    "metaphorical_interpretation": "تفسير مجازي",
    "contextual_scope": "النطاق السياقي",
    "general_vs_specific": "عام وخاص",
    "absolute_vs_restricted": "مطلق ومقيد",
    "rhetorical_devices": "أساليب بلاغية",
    "semantic_fields": "حقول دلالية",
    "synonym_analysis_homonym_antonym_analysis": "ترادف / تضاد / اشتراك لفظي",
    "chronological_order": "ترتيب زمني",
    "meccan_medinan": "مكي / مدني",
    "intertextual_links": "روابط نصية",
    "israiliyyat": "إسرائيليات",
    "philosophical_reflections": "تأملات فلسفية",
    "ethical_implications": "دلالات أخلاقية",
    "creedal_implications": "دلالات عقدية",
    "aqidah_classification": "تصنيف عقدي",
    "scientific_allusions": "إشارات علمية",
    "cosmological_notes": "ملاحظات كونية",
    "social_norms": "أعراف اجتماعية",
    "political_implications": "دلالات سياسية",
    "pedagogical_lessons": "دروس تربوية",
    "daawah_applications": "تطبيقات دعوية",
    "practical_guidance": "توجيه عملي",
    "ritual_implications": "دلالات تعبدية",
    "disputed_terms": "مصطلحات خلافية",
    "technical_definitions": "تعريفات اصطلاحية",
    "textual_variants": "اختلافات نصية",
    "manuscript_evidence": "أدلة مخطوطية",
    "comparative_tafsir": "تفسير مقارن",
    "methodological_notes": "ملاحظات منهجية",
    "summary": "ملخص",
    "conclusion": "خلاصة",
}

bad_rows = []


# =========================================================
# HELPERS
# =========================================================
def safe_text(text):
    if text is None:
        return ""
    return " ".join(str(text).split()).strip()


def tag_display(tag):
    return f"{TAG_LABELS_AR.get(tag, tag)} / {tag}"


def get_section_id(row, fallback_index):
    for col in ["id", "section_id", "verse_id"]:
        if col in row.index and pd.notna(row[col]):
            try:
                return int(row[col])
            except Exception:
                return safe_text(row[col])
    return fallback_index + 1


def highlight_with_window(context, span, window=300):
    context = safe_text(context)
    span = safe_text(span)

    if not context or not span or span not in context:
        return context

    start = context.index(span)
    end = start + len(span)

    left = max(0, start - window)
    right = min(len(context), end + window)

    snippet = context[left:right]

    if left > 0:
        snippet = "… " + snippet
    if right < len(context):
        snippet = snippet + " …"

    return snippet.replace(span, f"<mark>{span}</mark>", 1)


# =========================================================
# CSV LOADING (compatible with old pandas)
# =========================================================
def _read_csv_old_compatible(path_str, encoding, sep):
    try:
        # newer pandas
        return pd.read_csv(
            path_str,
            encoding=encoding,
            sep=sep,
            engine="python",
            quoting=csv.QUOTE_MINIMAL,
            on_bad_lines="skip"
        )
    except TypeError:
        # older pandas
        return pd.read_csv(
            path_str,
            encoding=encoding,
            sep=sep,
            engine="python",
            quoting=csv.QUOTE_MINIMAL,
            error_bad_lines=False,
            warn_bad_lines=False
        )


def read_csv_flexible(path_str):
    attempts = [
        ("utf-8-sig", ","),
        ("utf-8", ","),
        ("utf-16", ","),
        ("latin1", ","),
        ("utf-8-sig", ";"),
        ("utf-8", ";"),
        ("utf-16", ";"),
        ("latin1", ";"),
        ("utf-8-sig", "\t"),
        ("utf-8", "\t"),
        ("utf-16", "\t"),
        ("latin1", "\t"),
    ]

    last_error = None

    for encoding, sep in attempts:
        try:
            df = _read_csv_old_compatible(path_str, encoding, sep)
            if df is not None and df.shape[1] > 0:
                print(f"Loaded {path_str} with encoding={encoding}, sep={repr(sep)}, shape={df.shape}")
                return df
        except Exception as e:
            last_error = e

    raise last_error if last_error else ValueError(f"Could not read CSV: {path_str}")


def safe_read_existing_csv(path_str):
    path = Path(path_str)

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(path_str, encoding="utf-8-sig", engine="python")
    except Exception:
        return pd.DataFrame()


# =========================================================
# XML EXTRACTION
# =========================================================
def extract_annotations_from_xml(xml_text, row_id, source_name="main", filled_cols="", frequency_rank=""):
    rows = []

    if pd.isna(xml_text):
        return rows

    xml_text = str(xml_text).strip()
    if not xml_text:
        return rows

    try:
        root = ET.fromstring(xml_text)
    except Exception as e:
        bad_rows.append((row_id, str(e)))
        return rows

    section_context = safe_text("".join(root.itertext()))

    block_num = 0
    chunk_global_num = 0

    for block in root.findall(".//tafsir_section_block"):
        block_num += 1
        chunk_num = 0

        for chunk in block.findall("./tafsir_chunk"):
            chunk_num += 1
            chunk_global_num += 1

            chunk_context = safe_text("".join(chunk.itertext()))
            seen_in_chunk = set()

            for elem in chunk.iter():
                if not isinstance(elem.tag, str):
                    continue
                if elem.tag in SKIP_TAGS:
                    continue

                span = safe_text("".join(elem.itertext()))
                if not span:
                    continue

                key = (elem.tag, span)
                if key in seen_in_chunk:
                    continue
                seen_in_chunk.add(key)

                rows.append({
                    "dataset_source": source_name,
                    "csv_row": row_id,
                    "block_id": block_num,
                    "chunk_id": chunk_num,
                    "chunk_global_id": chunk_global_num,
                    "predicted_tag": elem.tag,
                    "predicted_tag_display": tag_display(elem.tag),
                    "predicted_span": span,
                    "chunk_context": chunk_context,
                    "section_context": section_context,
                    "highlighted_context": highlight_with_window(section_context, span, window=300),
                    "filled_cols": filled_cols,
                    "frequency_rank": frequency_rank,
                })

    return rows


# =========================================================
# MAIN DATASET
# =========================================================
def load_main_annotations():
    frame = read_csv_flexible(MAIN_FILE)

    if XML_COLUMN not in frame.columns:
        raise ValueError(f"Column '{XML_COLUMN}' not found in {MAIN_FILE}. Found columns: {frame.columns.tolist()}")

    all_rows = []

    for i, row in frame.iterrows():
        row_id = get_section_id(row, i)
        xml_text = row.get(XML_COLUMN, "")
        all_rows.extend(
            extract_annotations_from_xml(
                xml_text=xml_text,
                row_id=row_id,
                source_name="main"
            )
        )

    return all_rows


# =========================================================
# RARE DATASET
# =========================================================
def detect_xml_column(frame):
    preferred = [
        "extracted_text_full",
        "xml",
        "output_xml",
        "annotated_xml",
        "model_output",
        "response",
    ]

    for col in preferred:
        if col in frame.columns:
            return col

    for col in frame.columns:
        if frame[col].dtype == object:
            sample = frame[col].dropna().astype(str).head(10).tolist()
            if any("<tafsir_section" in x or "<tafsir_section_block" in x for x in sample):
                return col

    return None


def resolve_optional_column(frame, candidates):
    lower_map = {c.lower(): c for c in frame.columns}
    for c in candidates:
        if c in frame.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def load_rare_annotations():
    rare_path = Path(RARE_FILE)
    if not rare_path.exists():
        print(f"Rare file not found: {RARE_FILE}")
        return []

    rare_df = read_csv_flexible(RARE_FILE)
    print("Rare file columns:", rare_df.columns.tolist())

    xml_col = detect_xml_column(rare_df)
    if not xml_col:
        raise ValueError(f"Could not detect XML column in {RARE_FILE}. Columns: {rare_df.columns.tolist()}")

    filled_col = resolve_optional_column(rare_df, ["filled_cols", "filled_columns", "num_filled_cols"])
    rank_col = resolve_optional_column(rare_df, ["frequency_rank", "rank", "coverage_rank"])

    rows = []

    for i, row in rare_df.iterrows():
        row_id = get_section_id(row, i)
        filled_cols = row[filled_col] if filled_col else ""
        frequency_rank = row[rank_col] if rank_col else ""

        rows.extend(
            extract_annotations_from_xml(
                xml_text=row.get(xml_col, ""),
                row_id=row_id,
                source_name="rare",
                filled_cols=filled_cols,
                frequency_rank=frequency_rank,
            )
        )

    # Sort richer / earlier-ranked examples first
    def sort_key(x):
        try:
            rank = float(x["frequency_rank"]) if x["frequency_rank"] != "" else 10**9
        except Exception:
            rank = 10**9

        try:
            filled = -float(x["filled_cols"]) if x["filled_cols"] != "" else 0
        except Exception:
            filled = 0

        return (rank, filled)

    rows.sort(key=sort_key)
    return rows


# =========================================================
# FREQUENCY SPLIT + SAMPLE 10 PER TAG
# =========================================================
def compute_tag_frequencies(all_annotations):
    return Counter(a["predicted_tag"] for a in all_annotations)


def split_tags_by_frequency(counter):
    tags_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    tags = [tag for tag, _ in tags_sorted]
    n = len(tags)

    if n == 0:
        return [], [], []

    first_cut = n // 3
    second_cut = (2 * n) // 3

    high_tags = tags[:first_cut]
    medium_tags = tags[first_cut:second_cut]
    low_tags = tags[second_cut:]

    return high_tags, medium_tags, low_tags


def sample_annotations_per_tag(annotations, tags, n=10, seed=42):
    rng = random.Random(seed)
    sampled = []

    for tag in tags:
        tag_items = [a for a in annotations if a["predicted_tag"] == tag]

        if len(tag_items) <= n:
            chosen = tag_items
        else:
            chosen = rng.sample(tag_items, n)

        sampled.extend(chosen)

    return sampled


# =========================================================
# PHASE DATASETS
# =========================================================
def save_phase_dataset(rows, filename):
    pd.DataFrame(rows).to_csv(filename, index=False, encoding="utf-8-sig")


def load_phase_dataset(filename):
    path = Path(filename)
    if not path.exists() or path.stat().st_size == 0:
        return []

    try:
        return pd.read_csv(filename, encoding="utf-8-sig", engine="python").fillna("").to_dict(orient="records")
    except Exception:
        return []


def phase_files_exist():
    for phase in ["high", "medium", "low", "rare"]:
        path = Path(PHASE_FILES[phase])
        if not path.exists() or path.stat().st_size == 0:
            return False
    return True


def build_phase_datasets():
    if FORCE_REBUILD:
        for path in PHASE_FILES.values():
            p = Path(path)
            if p.exists():
                p.unlink()

    if phase_files_exist():
        print("Phase datasets already exist. Reusing them.")
        return

    # ---------- main ----------
    all_main_annotations = load_main_annotations()
    print(f"Loaded {len(all_main_annotations)} main annotations before sampling.")

    counter = compute_tag_frequencies(all_main_annotations)

    print("\nMain tag frequencies:")
    for tag, count in counter.most_common():
        print(f"{tag}: {count}")

    high_tags, medium_tags, low_tags = split_tags_by_frequency(counter)

    high_candidates = [a for a in all_main_annotations if a["predicted_tag"] in high_tags]
    medium_candidates = [a for a in all_main_annotations if a["predicted_tag"] in medium_tags]
    low_candidates = [a for a in all_main_annotations if a["predicted_tag"] in low_tags]

    high_rows = sample_annotations_per_tag(high_candidates, high_tags, n=SAMPLES_PER_TAG, seed=RANDOM_SEED)
    medium_rows = sample_annotations_per_tag(medium_candidates, medium_tags, n=SAMPLES_PER_TAG, seed=RANDOM_SEED)
    low_rows = sample_annotations_per_tag(low_candidates, low_tags, n=SAMPLES_PER_TAG, seed=RANDOM_SEED)

    # ---------- rare ----------
    all_rare_annotations = load_rare_annotations()
    print(f"Loaded {len(all_rare_annotations)} rare annotations before sampling.")

    rare_counter = compute_tag_frequencies(all_rare_annotations)

    print("\nRare tag frequencies:")
    for tag, count in rare_counter.most_common():
        print(f"{tag}: {count}")

    rare_tags = list(rare_counter.keys())
    rare_rows = sample_annotations_per_tag(all_rare_annotations, rare_tags, n=SAMPLES_PER_TAG, seed=RANDOM_SEED)

    # ---------- save ----------
    save_phase_dataset(high_rows, PHASE_FILES["high"])
    save_phase_dataset(medium_rows, PHASE_FILES["medium"])
    save_phase_dataset(low_rows, PHASE_FILES["low"])
    save_phase_dataset(rare_rows, PHASE_FILES["rare"])

    print(f"\nSaved {len(high_rows)} rows to {PHASE_FILES['high']}")
    print(f"Saved {len(medium_rows)} rows to {PHASE_FILES['medium']}")
    print(f"Saved {len(low_rows)} rows to {PHASE_FILES['low']}")
    print(f"Saved {len(rare_rows)} rows to {PHASE_FILES['rare']}")


# =========================================================
# RESULT FILES
# =========================================================
def load_existing_results(results_file):
    return safe_read_existing_csv(results_file)


def get_completed_keys_for_annotator(annotator_name, results_file):
    existing = load_existing_results(results_file)

    if existing.empty or not annotator_name:
        return set()

    subset = existing[existing["annotator"] == annotator_name]
    if subset.empty:
        return set()

    return set(
        zip(
            subset["dataset_source"],
            subset["csv_row"],
            subset["block_id"],
            subset["chunk_global_id"],
            subset["predicted_tag"],
            subset["predicted_span"],
        )
    )


def save_result(record, results_file):
    file_exists = Path(results_file).exists()

    fieldnames = [
        "annotator",
        "phase",
        "dataset_source",
        "csv_row",
        "block_id",
        "chunk_id",
        "chunk_global_id",
        "predicted_tag",
        "predicted_span",
        "decision",
        "correct_tag",
        "correct_tag_ar",
        "correct_span",
        "notes",
        "annotation_time_seconds",
        "filled_cols",
        "frequency_rank",
    ]

    with open(results_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)

        if not file_exists:
            writer.writeheader()

        writer.writerow(record)


# =========================================================
# BUILD ON STARTUP
# =========================================================
build_phase_datasets()

if bad_rows:
    print("\nRows with XML parse issues:")
    for row_num, err in bad_rows[:20]:
        print(f"Row {row_num}: {err}")
    if len(bad_rows) > 20:
        print(f"... and {len(bad_rows) - 20} more")


# =========================================================
# ROUTE
# =========================================================
@flask_app.route("/", methods=["GET", "POST"])
def index():
    phase = request.args.get("phase", "high").strip().lower()
    if phase not in {"high", "medium", "low", "rare"}:
        phase = "high"

    phase_file = PHASE_FILES[phase]
    results_file = RESULT_FILES[phase]

    annotations = load_phase_dataset(phase_file)
    tag_options = sorted(set(a["predicted_tag"] for a in annotations)) if annotations else []
    annotator = request.args.get("annotator", "").strip()

    if request.method == "POST":
        phase = request.form.get("phase", "high").strip().lower()
        results_file = RESULT_FILES[phase]

        annotator = request.form.get("annotator", "").strip()
        current_pos = int(request.form.get("current_pos", "0"))

        if not annotator:
            return "Please enter annotator name."

        current_annotations = load_phase_dataset(PHASE_FILES[phase])
        if current_pos >= len(current_annotations):
            return redirect(url_for("index", annotator=annotator, phase=phase))

        current_item = current_annotations[current_pos]
        correct_tag = request.form.get("correct_tag", "").strip()

        record = {
            "annotator": annotator,
            "phase": phase,
            "dataset_source": current_item.get("dataset_source", ""),
            "csv_row": current_item["csv_row"],
            "block_id": current_item["block_id"],
            "chunk_id": current_item["chunk_id"],
            "chunk_global_id": current_item["chunk_global_id"],
            "predicted_tag": current_item["predicted_tag"],
            "predicted_span": current_item["predicted_span"],
            "decision": request.form.get("decision", "").strip(),
            "correct_tag": correct_tag,
            "correct_tag_ar": TAG_LABELS_AR.get(correct_tag, "") if correct_tag else "",
            "correct_span": request.form.get("correct_span", "").strip(),
            "notes": request.form.get("notes", "").strip(),
            "annotation_time_seconds": request.form.get("annotation_time_seconds", "").strip(),
            "filled_cols": current_item.get("filled_cols", ""),
            "frequency_rank": current_item.get("frequency_rank", ""),
        }

        save_result(record, results_file)
        return redirect(url_for("index", annotator=annotator, phase=phase))

    if not annotator:
        return render_template(
            "index.html",
            annotator="",
            item=None,
            current_pos=None,
            total=len(annotations),
            remaining=len(annotations),
            tag_options=tag_options,
            tag_labels_ar=TAG_LABELS_AR,
            phase=phase,
            phase_label=PHASE_LABELS[phase],
            phase_labels=PHASE_LABELS,
        )

    completed_keys = get_completed_keys_for_annotator(annotator, results_file)

    remaining_items = []
    for idx, item in enumerate(annotations):
        key = (
            item.get("dataset_source", ""),
            item["csv_row"],
            item["block_id"],
            item["chunk_global_id"],
            item["predicted_tag"],
            item["predicted_span"],
        )
        if key not in completed_keys:
            remaining_items.append((idx, item))

    if not remaining_items:
        return render_template(
            "index.html",
            annotator=annotator,
            item=None,
            current_pos=None,
            total=len(annotations),
            remaining=0,
            tag_options=tag_options,
            tag_labels_ar=TAG_LABELS_AR,
            phase=phase,
            phase_label=PHASE_LABELS[phase],
            phase_labels=PHASE_LABELS,
            finished=True,
        )

    current_pos, item = remaining_items[0]

    return render_template(
        "index.html",
        annotator=annotator,
        item=item,
        current_pos=current_pos,
        total=len(annotations),
        remaining=len(remaining_items),
        tag_options=tag_options,
        tag_labels_ar=TAG_LABELS_AR,
        phase=phase,
        phase_label=PHASE_LABELS[phase],
        phase_labels=PHASE_LABELS,
    )
def greet():
    print("A tool to rate Gemini's semantic annotation of Classical Arabic.")

gr_interface = gr.Interface(fn=greet, inputs="text", outputs="text")

# Step 3: Mount the Gradio app to the Flask app
gr.mount_gradio_app(flask_app, gr_interface, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(flask_app, host="0.0.0.0", port=7860)
