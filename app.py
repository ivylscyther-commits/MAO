from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as PDFImage, LongTable, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

from mao_engine import (
    ALL_FEATURES, ALL_TARGETS, CORE_FEATURES, CURRENT_FEATURES,
    FAMILIES, MODE_LABELS, REMARK_COLUMN, ROUGHNESS_TARGETS,
    THICKNESS_TARGETS, ModelSuite, combine_training_data, ensure_database,
    list_trials, read_dataset, save_trial, trials_to_frame, update_trial,
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DEFAULT = BASE_DIR / "MAO DATASET.xlsx"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "trial_history.sqlite3"
ARTIFACT_PATH = BASE_DIR / "artifacts" / "mao_model_bundle.joblib"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ensure_database(DB_PATH)

DISPLAY_NAMES = {
    "Voltage (V) +ve": "Positive voltage (V)",
    "Voltage (V) -ve": "Negative voltage (V)",
    "Time (min)": "Treatment time (min)",
    "polarity(unipolar-1/bipolar-2)": "Polarity mode",
    "duty cycle +ve (%)": "Positive duty cycle (%)",
    "duty cycle -ve (%)": "Negative duty cycle (%)",
    "basic duty cycle (%)": "Basic duty cycle (%)",
}

st.set_page_config(page_title="MAO Process Prediction & Trial Analysis", page_icon="🧪", layout="wide")
st.title("MAO Process Prediction & Trial Analysis")
st.caption("Estimate coating thickness and surface roughness from historical MAO trials.")


def feature_help(title: str, description: str) -> None:
    """Client-facing feature explanation, collapsed by default and easy to hide."""
    with st.expander(f"What does {title} do?", expanded=False):
        st.write(description)


@st.cache_resource(show_spinner=False)
def load_engine_cached(dataset_path: str, dataset_signature: str, db_path: str, db_signature: str) -> ModelSuite:
    base = read_dataset(dataset_path)
    new_trials = trials_to_frame(db_path)
    merged = combine_training_data(base, new_trials)
    return ModelSuite(merged)


def file_signature(path: Path) -> str:
    if not path.exists():
        return "missing"
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def current_engine(dataset_path: Path) -> ModelSuite:
    ensure_database(DB_PATH)
    return load_engine_cached(
        str(dataset_path), file_signature(dataset_path), str(DB_PATH), file_signature(DB_PATH)
    )


def fmt(value: object, digits: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def pdf_safe(value: object) -> str:
    """Map characters unsupported by ReportLab's built-in Helvetica font."""
    text = str(value)
    return (text.replace("₃", "3").replace("₄", "4").replace("μ", "u").replace("µ", "u")
            .replace("−", "-").replace("–", "-").replace("•", "-").replace("→", "->"))


def model_metrics_table(suite: ModelSuite, mode: str) -> pd.DataFrame:
    """Return client-facing validation results without exposing model implementation names."""
    frame = suite.metrics.loc[(suite.metrics["mode"] == mode) & suite.metrics["selected"]].copy()
    if frame.empty:
        return frame
    return frame[["family", "Validation rows", "MAE Low", "MAE High", "R² Low", "R² High"]].rename(
        columns={
            "family": "Coating property",
            "Validation rows": "Historical trials checked",
            "MAE Low": "Average error (minimum, µm)",
            "MAE High": "Average error (maximum, µm)",
            "R² Low": "R² (minimum)",
            "R² High": "R² (maximum)",
        }
    ).reset_index(drop=True)


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    safe = pdf_safe(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe, style)


def make_prediction_pdf(
    payload: dict,
    predictions: dict,
    mode: str,
    suite: ModelSuite,
    warnings: list[str],
    similar: pd.DataFrame,
    current_delta: float | None,
) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm,
        topMargin=14 * mm, bottomMargin=14 * mm,
        title="MAO prediction report", author="MAO Prediction & Trial Intelligence",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CenterTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22, spaceAfter=6))
    styles.add(ParagraphStyle(name="SmallNote", parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor("#444444")))
    story = [
        Paragraph("MAO Coating Prediction Report", styles["CenterTitle"]),
        Paragraph(f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}", styles["Normal"]),
        Paragraph(f"Prediction mode: {MODE_LABELS[mode]}", styles["Normal"]),
        Spacer(1, 4 * mm),
        Paragraph("Predicted coating properties", styles["Heading2"]),
    ]
    outputs = [
        ["Outcome", "Predicted min (um)", "Predicted max (um)", "CV MAE min / max", "CV Q90 abs error min / max"],
    ]
    for family, targets in FAMILIES.items():
        unc = suite.uncertainty[mode][family]
        outputs.append([
            pdf_safe(family),
            fmt(predictions[targets[0]]),
            fmt(predictions[targets[1]]),
            f"{fmt(unc[targets[0]]['MAE'])} / {fmt(unc[targets[1]]['MAE'])}",
            f"{fmt(unc[targets[0]]['Q90 absolute error'])} / {fmt(unc[targets[1]]['Q90 absolute error'])}",
        ])
    out_table = Table(outputs, colWidths=[25 * mm, 30 * mm, 30 * mm, 34 * mm, 44 * mm], repeatRows=1)
    out_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263746")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7C2CA")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F7")]),
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([out_table, Spacer(1, 4 * mm)])

    # Visual interval chart, deliberately labelled as model output rather than measured data.
    try:
        import matplotlib.pyplot as plt
        chart_buffer = io.BytesIO()
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
        for ax, family, targets in zip(axes, ["Thickness", "Ra"], [THICKNESS_TARGETS, ROUGHNESS_TARGETS]):
            low = predictions[targets[0]]
            high = predictions[targets[1]]
            ax.hlines(0, low, high, linewidth=8)
            ax.plot([low, high], [0, 0], "o")
            ax.set_xlim(min(0, low * 0.95), max(high * 1.1, high + 0.1))
            ax.set_yticks([])
            ax.set_xlabel("µm")
            ax.set_title(f"Predicted {family} range")
            ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(chart_buffer, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        chart_buffer.seek(0)
        story.extend([Paragraph("Prediction range visual", styles["Heading2"]), PDFImage(chart_buffer, width=170 * mm, height=63 * mm), Spacer(1, 3 * mm)])
    except Exception:
        pass

    story.append(Paragraph("Input settings", styles["Heading2"]))
    input_rows = [["Parameter", "Value"]]
    for feat in (CORE_FEATURES if mode == "process" else ALL_FEATURES):
        value = payload.get(feat)
        if feat == "polarity(unipolar-1/bipolar-2)" and value is not None:
            value = "Unipolar" if float(value) == 1.0 else "Bipolar"
        else:
            value = fmt(value, 3)
        input_rows.append([pdf_safe(DISPLAY_NAMES.get(feat, feat)), pdf_safe(value)])
    input_table = Table(input_rows, colWidths=[110 * mm, 50 * mm], repeatRows=1)
    input_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263746")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C4CDD3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7F8")]),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.extend([input_table, Spacer(1, 3 * mm)])
    if current_delta is not None:
        story.append(_paragraph(f"Current change (final - initial): {current_delta:.3f}", styles["Normal"]))
        story.append(Spacer(1, 2 * mm))

    story.append(Paragraph("Historical validation results", styles["Heading2"]))
    metrics = model_metrics_table(suite, mode)
    metrics_rows = [["Property", "Trials", "Avg. error min", "Avg. error max", "R² min", "R² max"]]
    for _, row in metrics.iterrows():
        metrics_rows.append([
            row["Coating property"], str(int(row["Historical trials checked"])),
            fmt(row["Average error (minimum, µm)"]), fmt(row["Average error (maximum, µm)"]),
            fmt(row["R² (minimum)"], 3), fmt(row["R² (maximum)"], 3),
        ])
    metric_table = Table(metrics_rows, colWidths=[27 * mm, 17 * mm, 31 * mm, 31 * mm, 24 * mm, 24 * mm], repeatRows=1)
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263746")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C4CDD3")),
        ("FONTSIZE", (0, 0), (-1, -1), 7), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.extend([metric_table, Spacer(1, 2 * mm), _paragraph("Uncertainty values are empirical out-of-fold errors from historical trials, not guaranteed confidence intervals or certified process tolerances.", styles["SmallNote"]), Spacer(1, 3 * mm)])

    story.append(Paragraph("Range and support warnings", styles["Heading2"]))
    if warnings:
        for warning in warnings:
            story.append(_paragraph("- " + warning, styles["BodyText"]))
            story.append(Spacer(1, 1 * mm))
    else:
        story.append(Paragraph("No marginal range or nearest-trial support warning was triggered. This does not establish process safety or feasibility.", styles["BodyText"]))
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("Closest historical trials", styles["Heading2"]))
    if not similar.empty:
        cols = ["_source_label", "Similarity distance"] + THICKNESS_TARGETS + ROUGHNESS_TARGETS
        cols = [c for c in cols if c in similar.columns]
        rows = [[c.replace("_source_label", "Trial") for c in cols]]
        for _, row in similar.head(5).iterrows():
            rows.append([fmt(row.get(c), 3) if c != "_source_label" else pdf_safe(row.get(c, "")) for c in cols])
        rows = [[pdf_safe(cell) for cell in row] for row in rows]
        table = LongTable(rows, colWidths=[32 * mm, 20 * mm, 22 * mm, 22 * mm, 22 * mm, 22 * mm][:len(rows[0])], repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263746")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C4CDD3")),
            ("FONTSIZE", (0, 0), (-1, -1), 6.5), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("No complete-feature historical trials were available for retrieval.", styles["Normal"]))
    story.extend([
        Spacer(1, 5 * mm),
        _paragraph("Decision limitation: this report screens historical relationships. It does not replace engineering review, a controlled MAO trial, or independent validation on new data.", styles["SmallNote"]),
    ])
    doc.build(story)
    return buffer.getvalue()


def numeric_widget(label: str, default: float, step: float, min_value: float = 0.0, key: str | None = None, max_value: float | None = None) -> float:
    kwargs = {"label": label, "min_value": min_value, "value": float(default), "step": float(step), "format": "%.3f" if step < 1 else "%.1f"}
    if max_value is not None:
        kwargs["max_value"] = float(max_value)
    if key:
        kwargs["key"] = key
    return float(st.number_input(**kwargs))


# Uploaded workbooks persist locally and replace the active historical workbook; they do not append to it.
ACTIVE_DATASET_PATH = DATA_DIR / "active_dataset.xlsx"
DATA_SOURCE_FLAG = DATA_DIR / "use_uploaded_dataset.flag"


def validate_uploaded_workbook(content: bytes) -> None:
    """Reject unrelated/malformed workbooks before replacing the active data source."""
    try:
        raw = pd.read_excel(io.BytesIO(content))
    except Exception as exc:
        raise ValueError(f"The file could not be read as an Excel workbook: {exc}") from exc
    if raw.empty:
        raise ValueError("The workbook has no data rows.")
    required_columns = CORE_FEATURES + ALL_TARGETS
    missing = [name for name in required_columns if name not in raw.columns]
    if missing:
        preview = ", ".join(missing[:6])
        if len(missing) > 6:
            preview += f", and {len(missing) - 6} more"
        raise ValueError(f"This is not a compatible MAO workbook. Missing required columns: {preview}.")
    for pair_name, pair in (("thickness", THICKNESS_TARGETS), ("Ra", ROUGHNESS_TARGETS)):
        low = pd.to_numeric(raw[pair[0]], errors="coerce")
        high = pd.to_numeric(raw[pair[1]], errors="coerce")
        valid = low.notna() & high.notna() & (low >= 0) & (high >= low)
        if int(valid.sum()) < 3:
            raise ValueError(f"At least three valid measured {pair_name} ranges are required to train this model.")


with st.sidebar:
    st.header("MAO data source")
    with st.expander("How workbook uploads work", expanded=False):
        st.write(
            "Uploading a workbook replaces the active historical workbook; it does not append to the bundled workbook. "
            "The original workbook is left unchanged. Saved trials in this computer's trial database are included separately. "
            "The selected upload remains active after refreshing or restarting the app until you switch back to the original workbook. "
            "The model is rebuilt from the active workbook and saved trial records. Use the same MAO parameter and measured-outcome column names."
        )
    uploaded_dataset = st.file_uploader(
        "Upload replacement MAO workbook (.xlsx)", type=["xlsx"], key="replacement_dataset_upload"
    )
    if uploaded_dataset is not None:
        content = uploaded_dataset.getvalue()
        old_content = ACTIVE_DATASET_PATH.read_bytes() if ACTIVE_DATASET_PATH.exists() else None
        if old_content != content:
            try:
                validate_uploaded_workbook(content)
                temporary_path = DATA_DIR / "active_dataset.pending.xlsx"
                temporary_path.write_bytes(content)
                temporary_path.replace(ACTIVE_DATASET_PATH)
                DATA_SOURCE_FLAG.write_text("uploaded", encoding="utf-8")
                st.success("Workbook checked and activated. The model will rebuild using this workbook and saved trial records.")
            except Exception as exc:
                st.error(f"Upload rejected; the previous active data source was kept. {exc}")

    if ACTIVE_DATASET_PATH.exists() and DATA_SOURCE_FLAG.exists():
        st.caption(f"Active workbook: {ACTIVE_DATASET_PATH.name} (uploaded)")
        if st.button("Switch to original workbook", use_container_width=True, key="switch_to_original_workbook"):
            DATA_SOURCE_FLAG.unlink(missing_ok=True)
            st.rerun()
    elif ACTIVE_DATASET_PATH.exists():
        st.caption(f"Active workbook: {DATASET_DEFAULT.name} (original)")
        if st.button("Use last uploaded workbook", use_container_width=True, key="switch_to_uploaded_workbook"):
            DATA_SOURCE_FLAG.write_text("uploaded", encoding="utf-8")
            st.rerun()
    else:
        st.caption(f"Active workbook: {DATASET_DEFAULT.name} (original)")
    st.caption("Saved trial records remain stored on this computer.")
    dataset_path = ACTIVE_DATASET_PATH if (ACTIVE_DATASET_PATH.exists() and DATA_SOURCE_FLAG.exists()) else DATASET_DEFAULT

if not dataset_path.exists():
    st.error("MAO DATASET.xlsx was not found beside app.py. Upload the workbook in the sidebar to continue.")
    st.stop()

with st.spinner("Preparing historical trials and checking prediction performance. The first run can take a little while..."):
    try:
        suite = current_engine(dataset_path)
    except Exception as exc:
        st.exception(exc)
        st.stop()

last_prediction = st.session_state.get("last_prediction")
if last_prediction and (last_prediction.get("dataset_path") != str(dataset_path) or last_prediction.get("dataset_signature") != file_signature(dataset_path)):
    st.session_state.pop("last_prediction", None)

if st.session_state.get("app_notice"):
    st.info(st.session_state.pop("app_notice"))

# Keep the main screen focused on MAO work rather than software implementation details.
tab_predict, tab_history, tab_validation, tab_whatif, tab_database = st.tabs([
    "Prediction", "Historical trials", "Prediction checks", "Parameter exploration", "Trial records"
])

with tab_predict:
    feature_help(
        "Prediction",
        "Enter the MAO settings to estimate minimum and maximum coating thickness and surface roughness (Ra). "
        "Before-trial prediction uses process settings. After-trial analysis can also use measured initial and final current. "
        "Estimates are based on past trials and must be checked against real measurements."
    )
    st.subheader("MAO process settings")
    mode_label = st.radio(
        "Prediction timing",
        [MODE_LABELS["process"], MODE_LABELS["current"]],
        horizontal=True,
        help="Use process settings before a trial. Select the after-trial option only when measured initial and final current readings are available.",
    )
    mode = "process" if mode_label == MODE_LABELS["process"] else "current"
    polarity_label = st.radio("Polarity mode", ["Unipolar", "Bipolar"], horizontal=True)
    polarity = 1.0 if polarity_label == "Unipolar" else 2.0

    with st.form("prediction_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("#### Electrolyte")
            koh = numeric_widget("KOH (g/l)", 2.0, 0.5, key="p_koh")
            napo4 = numeric_widget("Na₃PO₄ (g/l)", 0.0, 0.5, key="p_napo4")
            nasio3 = numeric_widget("Na₂SiO₃ (g/l)", 6.0, 0.5, key="p_nasio3")
        with c2:
            st.markdown("#### Voltage")
            pos_voltage = numeric_widget("Positive voltage (V)", 480.0, 5.0, key="p_posv", max_value=1000)
            if polarity == 1.0:
                neg_voltage = 0.0
                st.number_input("Negative voltage (V)", value=0.0, disabled=True, key="p_negv_uni")
            else:
                neg_voltage = numeric_widget("Negative voltage (V)", 100.0, 5.0, key="p_negv", max_value=500)
        with c3:
            st.markdown("#### Timing & frequency")
            freq = numeric_widget("Frequency (Hz)", 1000.0, 10.0, key="p_freq", max_value=5000)
            duration = numeric_widget("Treatment time (min)", 25.0, 1.0, key="p_time", max_value=240)
        with c4:
            st.markdown("#### Duty cycle")
            basic_duty = numeric_widget("Basic duty cycle (%)", 50.0, 1.0, key="p_basic", max_value=100)
            pos_duty = numeric_widget("Positive duty cycle (%)", 50.0, 1.0, key="p_posd", max_value=100)
            if polarity == 1.0:
                neg_duty = 0.0
                st.number_input("Negative duty cycle (%)", value=0.0, disabled=True, key="p_negd_uni")
            else:
                neg_duty = numeric_widget("Negative duty cycle (%)", 50.0, 1.0, key="p_negd", max_value=100)
        if mode == "current":
            st.markdown("#### Current readings (post-trial mode)")
            cc1, cc2 = st.columns(2)
            with cc1:
                initial_current = numeric_widget("Initial current", 20.0, 1.0, key="p_initial_current", max_value=300)
            with cc2:
                final_current = numeric_widget("Final current", 7.0, 1.0, key="p_final_current", max_value=300)
        else:
            initial_current = None
            final_current = None
            st.info("Current readings are not used in pre-trial prediction. Switch to current-aware mode for post-trial analysis.")
        submit_prediction = st.form_submit_button("Predict coating properties", type="primary", use_container_width=True)

    if submit_prediction:
        payload = {
            "KOH (g/l)": koh,
            "Na₃PO₄ (g/l)": napo4,
            "Na₂SiO₃ (g/l)": nasio3,
            "Voltage (V) +ve": pos_voltage,
            "Voltage (V) -ve": neg_voltage,
            "Frequency (Hz)": freq,
            "Time (min)": duration,
            "polarity(unipolar-1/bipolar-2)": polarity,
            "duty cycle +ve (%)": pos_duty,
            "duty cycle -ve (%)": neg_duty,
            "basic duty cycle (%)": basic_duty,
            "initial current": initial_current,
            "final current": final_current,
        }
        pred = suite.predict(payload, mode)
        warnings = suite.range_warnings(payload, mode)
        similar = suite.similar_trials(payload, mode, 5)
        delta = None if initial_current is None or final_current is None else final_current - initial_current
        st.session_state["last_prediction"] = {
            "payload": payload, "predictions": pred, "mode": mode,
            "warnings": warnings, "similar": similar, "current_delta": delta,
            "dataset_signature": file_signature(dataset_path), "dataset_path": str(dataset_path),
        }

    result = st.session_state.get("last_prediction")
    if result:
        st.divider()
        st.subheader("Prediction results")
        p = result["predictions"]
        r1, r2 = st.columns(2)
        r1.metric("Predicted thickness range", f"{p[THICKNESS_TARGETS[0]]:.2f} – {p[THICKNESS_TARGETS[1]]:.2f} µm")
        r2.metric("Predicted Ra range", f"{p[ROUGHNESS_TARGETS[0]]:.2f} – {p[ROUGHNESS_TARGETS[1]]:.2f} µm")
        st.caption("These are estimates from historical trials, not guaranteed process outcomes.")

        u1, u2 = st.columns(2)
        for col, family, targets, title in [
            (u1, "Thickness", THICKNESS_TARGETS, "Thickness validation error"),
            (u2, "Ra", ROUGHNESS_TARGETS, "Ra validation error"),
        ]:
            with col:
                unc = suite.uncertainty[result["mode"]][family]
                st.markdown(f"**{title} (historical cross-validation)**")
                st.write(f"Minimum endpoint MAE: {unc[targets[0]]['MAE']:.2f} µm; 90th-percentile absolute error: {unc[targets[0]]['Q90 absolute error']:.2f} µm.")
                st.write(f"Maximum endpoint MAE: {unc[targets[1]]['MAE']:.2f} µm; 90th-percentile absolute error: {unc[targets[1]]['Q90 absolute error']:.2f} µm.")
                envelope_low = max(0.0, p[targets[0]] - unc[targets[0]]["Q90 absolute error"])
                envelope_high = p[targets[1]] + unc[targets[1]]["Q90 absolute error"]
                st.write(f"Empirical CV error envelope (not a confidence interval): {envelope_low:.2f}–{envelope_high:.2f} µm.")
        st.caption("The 90th-percentile historical absolute error is an empirical error envelope; it is not a calibrated confidence interval.")

        if result["current_delta"] is not None:
            d1, d2, d3 = st.columns(3)
            initial = result["payload"]["initial current"]
            final = result["payload"]["final current"]
            d1.metric("Initial current", f"{initial:.2f}")
            d2.metric("Final current", f"{final:.2f}")
            d3.metric("Current change (final − initial)", f"{result['current_delta']:+.2f}")
            if initial not in (None, 0):
                st.write(f"Relative current change: {(result['current_delta'] / initial) * 100:+.1f}% of initial current.")

        if result["warnings"]:
            st.warning("**Input support warnings**\n\n" + "\n\n".join(result["warnings"]))
        else:
            st.success("No marginal range or nearest-trial support warning was triggered. This is not a safety or feasibility certification.")

        st.markdown("#### Closest historical trials — actual results")
        with st.expander("How to read this comparison", expanded=False):
            st.write("These are recorded trials with settings closest to the current inputs. Their thickness and Ra columns are actual historical measurements, not estimates. A smaller setting-distance value means a closer match; it does not guarantee the same outcome.")
        if result["similar"].empty:
            st.info("No complete-input historical trials were available for comparison.")
        else:
            similar_view = result["similar"].rename(columns=DISPLAY_NAMES).rename(columns={"_source_label": "Trial", "Similarity distance": "Setting distance (lower is closer)"})
            st.dataframe(similar_view, use_container_width=True, hide_index=True)


        pdf_bytes = make_prediction_pdf(
            result["payload"], result["predictions"], result["mode"], suite,
            result["warnings"], result["similar"], result["current_delta"],
        )
        with st.expander("What is included in the PDF report?", expanded=False):
            st.write("The report contains entered settings, estimated thickness and Ra ranges, historical validation errors, warnings, and the closest comparable historical trials.")
        st.download_button(
            "Download PDF report", data=pdf_bytes,
            file_name=f"MAO_prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf", type="primary",
        )

with tab_history:
    feature_help(
        "Historical trials",
        "Search the selected historical workbook and saved trial records. Review recorded process settings, remarks, "
        "and measured thickness/Ra results. Historical measurements are observations, not predictions."
    )
    st.subheader("Search historical and saved trials")
    search = st.text_input("Search by remarks, trial label, or numeric value", key="history_search")
    hist = suite.data.copy()
    if search.strip():
        mask = hist[REMARK_COLUMN].astype(str).str.contains(search.strip(), case=False, na=False)
        mask |= hist["_source_label"].astype(str).str.contains(search.strip(), case=False, na=False)
        # Allow searching a number that appears in any numeric field.
        for col in CORE_FEATURES + CURRENT_FEATURES + ALL_TARGETS:
            mask |= hist[col].map(lambda x: search.strip() in str(x) if pd.notna(x) else False)
        hist = hist.loc[mask]
    st.caption(f"Showing {len(hist):,} matching records.")
    show_cols = ["_source_label"] + CORE_FEATURES + CURRENT_FEATURES + ALL_TARGETS + [REMARK_COLUMN]
    hist_view = hist[[c for c in show_cols if c in hist.columns]].rename(columns=DISPLAY_NAMES).rename(columns={"_source_label": "Trial", REMARK_COLUMN: "Notes"})
    st.dataframe(hist_view, use_container_width=True, hide_index=True)
    if st.session_state.get("last_prediction"):
        st.markdown("#### Compare latest prediction with closest trials")
        nearest = st.session_state["last_prediction"]["similar"]
        if not nearest.empty:
            st.dataframe(nearest[[c for c in ["_source_label", "Similarity distance"] + ALL_TARGETS + [REMARK_COLUMN] if c in nearest.columns]], use_container_width=True, hide_index=True)

with tab_validation:
    feature_help(
        "Prediction checks",
        "These checks compare model estimates with historical measurements not used to fit each validation fold. "
        "Lower MAE means smaller average absolute error; an R² closer to 1 generally means the model follows the measured variation more closely. "
        "Scores are preliminary and do not guarantee accuracy on future trials. Missing-data and unusual-record checks are provided below."
    )
    st.subheader("Prediction performance")
    validation_mode_label = st.radio("Validation mode", [MODE_LABELS["process"], MODE_LABELS["current"]], horizontal=True, key="validation_mode")
    validation_mode = "process" if validation_mode_label == MODE_LABELS["process"] else "current"
    st.dataframe(model_metrics_table(suite, validation_mode).style.format({
        "Average error (minimum, µm)": "{:.3f}", "Average error (maximum, µm)": "{:.3f}",
        "R² (minimum)": "{:.3f}", "R² (maximum)": "{:.3f}",
    }), use_container_width=True, hide_index=True)
    st.caption("Average error is measured in micrometres; lower is better. R² nearer to 1 generally indicates a closer fit to the historical measurements. Scores are preliminary and do not guarantee future results.")

    validation_subset = suite.validation_predictions.loc[suite.validation_predictions["mode"] == validation_mode]
    if not validation_subset.empty:
        st.markdown("#### Measured results versus estimates")
        with st.expander("What this graph shows", expanded=False):
            st.write("Each dot compares an estimate with the measurement recorded for a historical trial. Dots closer to the diagonal line indicate closer estimates. Past performance does not guarantee future accuracy.")
        cv_target = st.selectbox("Coating property to compare", list(ALL_TARGETS), key="cv_target")
        cv_plot_df = validation_subset.loc[validation_subset["outcome"] == cv_target]
        fig = px.scatter(cv_plot_df, x="Actual", y="Predicted", hover_data=["Absolute error"],
                         title=f"Historical measured values versus estimates — {cv_target}",
                         labels={"Actual": "Measured value (µm)", "Predicted": "Estimated value (µm)"})
        finite = pd.concat([cv_plot_df["Actual"], cv_plot_df["Predicted"]]).replace([np.inf, -np.inf], np.nan).dropna()
        if not finite.empty:
            lo, hi = float(finite.min()), float(finite.max())
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal", line={"dash": "dash"}))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Process setting versus measured outcome")
    with st.expander("What this graph shows", expanded=False):
        st.write("Each point is a historical trial with both values recorded. This graph shows association in the dataset; it does not prove that changing one setting causes the outcome to change.")
    g1, g2 = st.columns(2)
    with g1:
        x_feature = st.selectbox("Process setting", CORE_FEATURES, key="relationship_feature", format_func=lambda value: DISPLAY_NAMES.get(value, value))
    with g2:
        y_target = st.selectbox("Measured outcome", ALL_TARGETS, key="relationship_target")
    plot_df = suite.data[[x_feature, y_target, "_source_label"]].dropna()
    if not plot_df.empty:
        plot = px.scatter(plot_df, x=x_feature, y=y_target, hover_data=["_source_label"], title=f"Historical relationship: {DISPLAY_NAMES.get(x_feature, x_feature)} vs {y_target}", labels={x_feature: DISPLAY_NAMES.get(x_feature, x_feature)})
        st.plotly_chart(plot, use_container_width=True)
    else:
        st.info("No rows have both the selected parameter and outcome.")

    st.markdown("#### How current changes relate to measured outcomes")
    with st.expander("What this analysis shows", expanded=False):
        st.write("Current change is calculated as final current minus initial current. The graph compares that change with measured coating outcomes across historical trials; it describes a relationship and does not prove causation.")
    current_audit = suite.data.loc[suite.data[CURRENT_FEATURES].notna().all(axis=1)].copy()
    if current_audit.empty:
        st.info("No records contain both current readings.")
    else:
        current_audit["Current change (final - initial)"] = current_audit["final current"] - current_audit["initial current"]
        current_audit["Current change (%)"] = np.where(
            current_audit["initial current"] != 0,
            current_audit["Current change (final - initial)"] / current_audit["initial current"] * 100.0, np.nan
        )
        cstat1, cstat2, cstat3 = st.columns(3)
        cstat1.metric("Trials with both currents", f"{len(current_audit):,}")
        cstat2.metric("Median current change", f"{current_audit['Current change (final - initial)'].median():.2f}")
        cstat3.metric("Current decreased", f"{int((current_audit['Current change (final - initial)'] < 0).sum()):,}")
        current_outcome = st.selectbox("Outcome to compare with current change", ALL_TARGETS, key="current_change_outcome")
        current_plot_data = current_audit[["Current change (final - initial)", current_outcome, "_source_label"]].dropna()
        if not current_plot_data.empty:
            current_fig = px.scatter(current_plot_data, x="Current change (final - initial)", y=current_outcome, hover_data=["_source_label"], title=f"Historical current change vs {current_outcome}")
            st.plotly_chart(current_fig, use_container_width=True)
        current_view = current_audit[["_source_label", "initial current", "final current", "Current change (final - initial)", "Current change (%)"] + ALL_TARGETS].sort_values("Current change (final - initial)")
        current_view = current_view.rename(columns=DISPLAY_NAMES).rename(columns={"_source_label": "Trial"})
        st.dataframe(current_view, use_container_width=True, hide_index=True)

    st.markdown("#### Missing measurements")
    with st.expander("Why this matters", expanded=False):
        st.write("Missing values reduce the number of trials available for some predictions. The application handles missing input values during modelling, but an outcome cannot be learned from a row where that outcome was not measured.")
    missing_df = suite.data[CORE_FEATURES + CURRENT_FEATURES + ALL_TARGETS].isna().sum().rename("Missing rows").to_frame()
    missing_df["Missing %"] = (missing_df["Missing rows"] / len(suite.data) * 100).round(1)
    st.dataframe(missing_df.sort_values("Missing rows", ascending=False), use_container_width=True)
    thickness_pair = suite.data[THICKNESS_TARGETS].dropna()
    ra_pair = suite.data[ROUGHNESS_TARGETS].dropna()
    invalid_thickness = int((thickness_pair.iloc[:, 0] > thickness_pair.iloc[:, 1]).sum())
    invalid_ra = int((ra_pair.iloc[:, 0] > ra_pair.iloc[:, 1]).sum())
    dq1, dq2, dq3 = st.columns(3)
    dq1.metric("Rows with all 4 outcomes", int(suite.data[ALL_TARGETS].notna().all(axis=1).sum()))
    dq2.metric("Thickness min > max", invalid_thickness)
    dq3.metric("Ra min > max", invalid_ra)

    if st.button("Find unusual historical records", key="run_outlier_review"):
        with st.spinner("Reviewing unusual records..."):
            st.session_state["outlier_review"] = suite.outlier_rows()
    if "outlier_review" in st.session_state:
        outliers = st.session_state["outlier_review"]
        st.write(f"Records flagged for review: {len(outliers)}. A flag does not mean a measurement is wrong; no records are automatically deleted.")
        if not outliers.empty:
            cols = ["_source_label"] + CORE_FEATURES + ALL_TARGETS + ["Outlier score", REMARK_COLUMN]
            st.dataframe(outliers[[c for c in cols if c in outliers.columns]], use_container_width=True, hide_index=True)

with tab_whatif:
    feature_help(
        "Parameter exploration",
        "Change one process setting and see how the model's estimated thickness or Ra changes while the other listed settings stay fixed. "
        "Target search ranks candidate settings that may match a requested outcome range. These are screening tools, not proof of causation, safety, or an optimal recipe."
    )
    st.subheader("What-if analysis")
    st.write("Change one setting at a time. All other process settings shown below are held constant for the curve.")
    latest = st.session_state.get("last_prediction")
    if latest:
        base_values = latest["payload"].copy()
    else:
        base_values = {
            "KOH (g/l)": 2.0, "Na₃PO₄ (g/l)": 0.0, "Na₂SiO₃ (g/l)": 6.0,
            "Voltage (V) +ve": 480.0, "Voltage (V) -ve": 0.0,
            "Frequency (Hz)": 1000.0, "Time (min)": 25.0,
            "polarity(unipolar-1/bipolar-2)": 1.0, "duty cycle +ve (%)": 50.0,
            "duty cycle -ve (%)": 0.0, "basic duty cycle (%)": 50.0,
            "initial current": 20.0, "final current": 7.0,
        }
    selected_parameter = st.selectbox(
        "Parameter to vary", CORE_FEATURES, key="whatif_parameter",
        format_func=lambda value: DISPLAY_NAMES.get(value, value),
    )
    defaults = {
        "KOH (g/l)": 2.0, "Na₃PO₄ (g/l)": 0.0, "Na₂SiO₃ (g/l)": 6.0,
        "Voltage (V) +ve": 480.0, "Voltage (V) -ve": 0.0,
        "Frequency (Hz)": 1000.0, "Time (min)": 25.0,
        "polarity(unipolar-1/bipolar-2)": 1.0, "duty cycle +ve (%)": 50.0,
        "duty cycle -ve (%)": 0.0, "basic duty cycle (%)": 50.0,
    }
    baseline_table = pd.DataFrame([
        {
            "Process setting": feature,
            "Baseline value": float(base_values.get(feature, defaults.get(feature, 0.0))),
            "Treatment": "Varied across the curve" if feature == selected_parameter else "Held constant",
        }
        for feature in CORE_FEATURES
    ])
    baseline_table["Process setting"] = baseline_table["Process setting"].map(lambda value: DISPLAY_NAMES.get(value, value))
    baseline_table["Baseline value"] = baseline_table["Baseline value"].astype(object)
    baseline_table.loc[baseline_table["Process setting"] == "Polarity mode", "Baseline value"] = baseline_table.loc[baseline_table["Process setting"] == "Polarity mode", "Baseline value"].map({1.0: "Unipolar", 2.0: "Bipolar"})
    with st.expander("Show all settings used for this analysis", expanded=True):
        st.dataframe(baseline_table, use_container_width=True, hide_index=True)
        st.caption("The baseline comes from your latest prediction. Before you make one, standard defaults are shown. The selected setting is varied across the historical range; all other settings are held constant. To change the fixed settings, update them in the Prediction tab and submit another prediction. Current readings are not used in this what-if curve.")
    range_min, range_max = suite.bounds["process"].get(selected_parameter, (0.0, 1.0))
    if selected_parameter == "polarity(unipolar-1/bipolar-2)":
        grid_values = [1.0, 2.0]
        st.caption("Polarity is categorical; only the observed settings 1 (unipolar) and 2 (bipolar) are compared.")
    else:
        low = max(0.0, range_min)
        high = range_max
        grid_values = np.linspace(low, high, 25).tolist()
        st.caption(f"Observed range for this parameter: {range_min:g} to {range_max:g}.")
    if st.button("Run what-if curve", type="primary", key="run_whatif"):
        records = []
        for val in grid_values:
            candidate = base_values.copy()
            candidate[selected_parameter] = float(val)
            if float(candidate.get("polarity(unipolar-1/bipolar-2)", 2.0)) == 1.0:
                candidate["Voltage (V) -ve"] = 0.0
                candidate["duty cycle -ve (%)"] = 0.0
            prediction = suite.predict(candidate, "process")
            records.append({selected_parameter: val, **prediction})
        st.session_state["whatif_results"] = {"parameter": selected_parameter, "data": pd.DataFrame(records)}
    if "whatif_results" in st.session_state:
        whatif_result = st.session_state["whatif_results"]
        graph_data = whatif_result["data"]
        selected_y = st.selectbox("Outcome to plot", ALL_TARGETS, key="whatif_outcome")
        line = px.line(
            graph_data, x=whatif_result["parameter"], y=selected_y, markers=True,
            title=f"What-if: {selected_y} versus {DISPLAY_NAMES.get(whatif_result['parameter'], whatif_result['parameter'])}",
            labels={whatif_result["parameter"]: DISPLAY_NAMES.get(whatif_result["parameter"], whatif_result["parameter"])},
        )
        st.plotly_chart(line, use_container_width=True)
        st.dataframe(graph_data, use_container_width=True, hide_index=True)
        st.caption("Marginally in-range values can still form combinations unlike any historical trial. Check the support warnings and verify any candidate experimentally.")

    st.divider()
    st.subheader("Target-based setting search")
    with st.expander("What this search does", expanded=False):
        st.write("Enter the desired thickness and Ra ranges. The tool ranks historical settings and nearby variations by how closely their estimated outcomes approach those targets. Every result must be reviewed and tested; it is not a proven optimum.")
    with st.form("target_search_form"):
        ts1, ts2 = st.columns(2)
        with ts1:
            tmin = st.number_input("Target thickness minimum (µm)", min_value=0.0, value=20.0, step=1.0, key="target_tmin")
            tmax = st.number_input("Target thickness maximum (µm)", min_value=0.0, value=25.0, step=1.0, key="target_tmax")
        with ts2:
            rmin = st.number_input("Target Ra minimum (µm)", min_value=0.0, value=1.0, step=0.1, key="target_rmin")
            rmax = st.number_input("Target Ra maximum (µm)", min_value=0.0, value=2.0, step=0.1, key="target_rmax")
        run_search = st.form_submit_button("Search candidate settings", type="primary")
    if run_search:
        if tmin > tmax or rmin > rmax:
            st.error("For each target, minimum must be less than or equal to maximum.")
        else:
            with st.spinner("Scoring candidate settings..."):
                ranked = suite.target_search(tmin, tmax, rmin, rmax)
            st.session_state["target_search_results"] = ranked
    if "target_search_results" in st.session_state:
        target_results = st.session_state["target_search_results"].copy()
        target_results = target_results.rename(columns={col: DISPLAY_NAMES.get(col, col) for col in target_results.columns})
        st.dataframe(target_results, use_container_width=True, hide_index=True)
        st.warning("Treat these as candidates for engineering review, not prescribed process settings. Predicted closeness to a target is not proof of safe or robust operation.")

with tab_database:
    feature_help(
        "Trial records",
        "Save process settings for new experiments, add actual measured currents and coating results when available, compare settings with historical trials, "
        "and retrain the model after adding real measurements. Records are stored locally on this computer. Only measured outcomes can be used as training targets."
    )
    st.subheader("Trial history")
    st.caption("Records are stored on this computer. A trial without measured outcomes can be saved for tracking, but it cannot teach the model its result until measurements are entered.")
    saved_trials = list_trials(DB_PATH)
    if saved_trials:
        table_rows = []
        for trial in saved_trials:
            row = {"Trial ID": trial["id"], "Created": trial["created_at"], "Remark": trial["remark"]}
            row.update(trial["payload"])
            table_rows.append(row)
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No locally saved trials yet.")

    if st.session_state.get("new_trial_comparison"):
        comparison = st.session_state["new_trial_comparison"]
        st.markdown(f"#### Closest historical settings for trial #{comparison['id']}")
        comparison_view = comparison["similar"].rename(columns=DISPLAY_NAMES).rename(columns={"_source_label": "Trial", "Similarity distance": "Setting distance (lower is closer)"})
        st.dataframe(comparison_view, use_container_width=True, hide_index=True)

    st.markdown("#### Add a trial")
    add_polarity_label = st.radio("New trial polarity", ["Unipolar", "Bipolar"], horizontal=True, key="new_trial_polarity")
    add_polarity = 1.0 if add_polarity_label == "Unipolar" else 2.0
    st.markdown("**Optional measurements**")
    # These toggles must sit outside the form: Streamlit forms do not rerun until submitted.
    record_currents = st.checkbox("Record initial and final current now", key="n_record_currents")
    record_outcomes = st.checkbox("Measured thickness and Ra are available", key="n_record_outcomes")
    with st.form("add_trial_form"):
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            add_koh = numeric_widget("KOH (g/l)", 2.0, 0.5, key="n_koh")
            add_napo4 = numeric_widget("Na₃PO₄ (g/l)", 0.0, 0.5, key="n_napo4")
            add_nasio3 = numeric_widget("Na₂SiO₃ (g/l)", 6.0, 0.5, key="n_nasio3")
        with a2:
            add_posv = numeric_widget("Positive voltage (V)", 480.0, 5.0, key="n_posv", max_value=1000)
            if add_polarity == 1.0:
                add_negv = 0.0
                st.number_input("Negative voltage (V)", value=0.0, disabled=True, key="n_negv_uni")
            else:
                add_negv = numeric_widget("Negative voltage (V)", 100.0, 5.0, key="n_negv", max_value=500)
        with a3:
            add_freq = numeric_widget("Frequency (Hz)", 1000.0, 10.0, key="n_freq", max_value=5000)
            add_time = numeric_widget("Treatment time (min)", 25.0, 1.0, key="n_time", max_value=240)
        with a4:
            add_basic = numeric_widget("Basic duty cycle (%)", 50.0, 1.0, key="n_basic", max_value=100)
            add_posd = numeric_widget("Positive duty cycle (%)", 50.0, 1.0, key="n_posd", max_value=100)
            if add_polarity == 1.0:
                add_negd = 0.0
                st.number_input("Negative duty cycle (%)", value=0.0, disabled=True, key="n_negd_uni")
            else:
                add_negd = numeric_widget("Negative duty cycle (%)", 20.0, 1.0, key="n_negd", max_value=100)
        if record_currents:
            ac1, ac2 = st.columns(2)
            with ac1:
                add_initial = numeric_widget("Initial current", 20.0, 1.0, key="n_initial_current", max_value=300)
            with ac2:
                add_final = numeric_widget("Final current", 7.0, 1.0, key="n_final_current", max_value=300)
        else:
            add_initial, add_final = None, None
        if record_outcomes:
            o1, o2, o3, o4 = st.columns(4)
            with o1:
                add_tmin = st.number_input("Measured thickness min (µm)", min_value=0.0, value=18.0, step=0.5, key="n_tmin")
            with o2:
                add_tmax = st.number_input("Measured thickness max (µm)", min_value=0.0, value=24.0, step=0.5, key="n_tmax")
            with o3:
                add_rmin = st.number_input("Measured Ra min (µm)", min_value=0.0, value=1.0, step=0.1, key="n_rmin")
            with o4:
                add_rmax = st.number_input("Measured Ra max (µm)", min_value=0.0, value=2.0, step=0.1, key="n_rmax")
        else:
            add_tmin = add_tmax = add_rmin = add_rmax = None
        add_remark = st.text_area("Trial remark / notes", key="n_remark")
        save_trial_button = st.form_submit_button("Save trial", type="primary")
    if save_trial_button:
        if record_outcomes and (add_tmin > add_tmax or add_rmin > add_rmax):
            st.error("Measured minimum must not exceed measured maximum.")
        else:
            payload = {
                "KOH (g/l)": add_koh, "Na₃PO₄ (g/l)": add_napo4, "Na₂SiO₃ (g/l)": add_nasio3,
                "Voltage (V) +ve": add_posv, "Voltage (V) -ve": add_negv,
                "Frequency (Hz)": add_freq, "Time (min)": add_time,
                "polarity(unipolar-1/bipolar-2)": add_polarity,
                "duty cycle +ve (%)": add_posd, "duty cycle -ve (%)": add_negd,
                "basic duty cycle (%)": add_basic,
                "initial current": add_initial, "final current": add_final,
                "Thickness Min (μm)": add_tmin, "Thickness Max (μm)": add_tmax,
                "Ra Value Min (μm)": add_rmin, "Ra Value Max (μm)": add_rmax,
            }
            # Compare this new configuration against the history before inserting it.
            comparison = suite.similar_trials(payload, "process", 6)
            trial_id = save_trial(DB_PATH, payload, add_remark)
            st.session_state["new_trial_comparison"] = {"id": trial_id, "similar": comparison.head(5)}
            load_engine_cached.clear()
            st.session_state.pop("last_prediction", None)
            st.session_state["app_notice"] = f"Saved trial #{trial_id}. The model suite has been refreshed; records without measured outcomes are not used as targets."
            st.rerun()

    st.markdown("#### Add measured results to an existing trial")
    if saved_trials:
        trial_ids = [int(t["id"]) for t in saved_trials]
        update_currents = st.checkbox("Update current readings", key="u_update_currents")
        update_outcomes = st.checkbox("Update measured thickness and Ra", key="u_update_outcomes")
        with st.form("update_trial_form"):
            selected_trial_id = st.selectbox("Trial ID to update", trial_ids)
            if update_currents:
                uc1, uc2 = st.columns(2)
                with uc1:
                    update_initial = st.number_input("Initial current", min_value=0.0, value=20.0, step=1.0, key="u_initial_current")
                with uc2:
                    update_final = st.number_input("Final current", min_value=0.0, value=7.0, step=1.0, key="u_final_current")
            else:
                update_initial = update_final = None
            if update_outcomes:
                uc3, uc4, uc5, uc6 = st.columns(4)
                with uc3:
                    update_tmin = st.number_input("Thickness min (µm)", min_value=0.0, value=18.0, step=0.5, key="u_tmin")
                with uc4:
                    update_tmax = st.number_input("Thickness max (µm)", min_value=0.0, value=24.0, step=0.5, key="u_tmax")
                with uc5:
                    update_rmin = st.number_input("Ra min (µm)", min_value=0.0, value=1.0, step=0.1, key="u_rmin")
                with uc6:
                    update_rmax = st.number_input("Ra max (µm)", min_value=0.0, value=2.0, step=0.1, key="u_rmax")
            else:
                update_tmin = update_tmax = update_rmin = update_rmax = None
            update_remark = st.text_input("Replace remark (leave blank to clear)", key="u_remark")
            update_button = st.form_submit_button("Update trial results")
        if update_button:
            if update_outcomes and (update_tmin > update_tmax or update_rmin > update_rmax):
                st.error("Measured minimum must not exceed measured maximum.")
            else:
                updates = {}
                if update_currents:
                    updates.update({"initial current": update_initial, "final current": update_final})
                if update_outcomes:
                    updates.update({
                        "Thickness Min (μm)": update_tmin, "Thickness Max (μm)": update_tmax,
                        "Ra Value Min (μm)": update_rmin, "Ra Value Max (μm)": update_rmax,
                    })
                current_record = next((t for t in saved_trials if int(t["id"]) == int(selected_trial_id)), None)
                if current_record is not None:
                    compare_payload = dict(current_record["payload"])
                    compare_payload.update(updates)
                    compare_df = suite.similar_trials(compare_payload, "process", 6)
                    compare_df = compare_df.loc[compare_df["_source_label"] != f"Database trial #{selected_trial_id}"].head(5) if not compare_df.empty else compare_df
                else:
                    compare_df = pd.DataFrame()
                found = update_trial(DB_PATH, selected_trial_id, updates, update_remark)
                if found:
                    st.session_state["new_trial_comparison"] = {"id": int(selected_trial_id), "similar": compare_df}
                    load_engine_cached.clear()
                    st.session_state.pop("last_prediction", None)
                    st.session_state["app_notice"] = f"Updated trial #{selected_trial_id}. Models were refreshed with the updated trial data."
                    st.rerun()
                else:
                    st.error("That trial ID does not exist.")

    if st.button("Retrain and save model artifacts", type="primary", key="retrain_models"):
        with st.spinner("Retraining and validating models using workbook plus saved trials..."):
            try:
                load_engine_cached.clear()
                suite = current_engine(dataset_path)
                suite.save_artifacts(ARTIFACT_PATH)
                st.session_state.pop("last_prediction", None)
                st.session_state["app_notice"] = f"Retrained and saved model bundle to {ARTIFACT_PATH.relative_to(BASE_DIR)}. Review the new validation metrics before engineering use."
                st.rerun()
            except Exception as exc:
                st.exception(exc)
