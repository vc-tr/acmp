"""Streamlit web demo for ACMP.

Upload comic/manga pages (or a PDF), preview detected panels with bounding-box
overlays, then render an animated motion-comic MP4 — all in the browser.

Run locally:   streamlit run streamlit_app.py
Deploy:        works as-is on Hugging Face Spaces / Streamlit Community Cloud.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from acmp import __version__
from acmp.config import PipelineConfig
from acmp.panels.detector import detect_panels, detect_panels_vertical_scroll
from acmp.utils.reading_order import detect_reading_order, sort_panels_by_reading_order

_IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "bmp", "tiff"]
_PALETTE = [(239, 71, 111), (17, 138, 178), (6, 214, 160), (255, 209, 102), (155, 93, 229)]


def _load_uploads(uploaded) -> list[Image.Image]:
    """Turn uploaded files (images or a single PDF) into a list of page images."""
    if len(uploaded) == 1 and uploaded[0].name.lower().endswith(".pdf"):
        from acmp.ingest.pdf_extractor import extract_pages_from_pdf

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded[0].getvalue())
            pdf_path = Path(tmp.name)
        return extract_pages_from_pdf(pdf_path)

    pages = []
    for up in sorted(uploaded, key=lambda u: u.name):
        if not up.name.lower().endswith(".pdf"):
            pages.append(Image.open(up).convert("RGB"))
    return pages


def _draw_boxes(page: Image.Image, boxes, order: str) -> Image.Image:
    """Overlay numbered panel boxes in reading order."""
    canvas = page.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width = max(2, page.width // 250)
    for i, (x, y, w, h) in enumerate(boxes):
        color = _PALETTE[i % len(_PALETTE)]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
        draw.text((x + 6, y + 6), str(i + 1), fill=color)
    return canvas


def main() -> None:
    st.set_page_config(page_title="ACMP — Motion Comics", page_icon="🎬", layout="wide")
    st.title("🎬 ACMP — Animated Comics / Manga / Manhwa Panels")
    st.caption(
        f"v{__version__} · Detect panels, then render a 9:16 motion-comic video. "
        "CV (panel detection) + optional LLM scene analysis + AI/Ken Burns animation."
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        reading_order = st.selectbox("Reading order", ["auto", "rtl", "ltr", "vertical"], index=0)
        seconds = st.slider("Seconds per panel", 1.0, 8.0, 4.0, 0.5)
        fps = st.select_slider("FPS", [8, 12, 16, 24, 30], value=24)
        use_ai = st.checkbox("AI animation (Wan VACE — slow, needs GPU/weights)", value=False)
        llm = st.selectbox("Scene analysis", ["fallback", "claude", "ollama"], index=0)
        st.markdown("---")
        st.markdown("Tip: leave **AI animation off** for a fast Ken-Burns preview.")

    uploaded = st.file_uploader(
        "Upload pages (images) or a single PDF",
        type=_IMAGE_TYPES + ["pdf"],
        accept_multiple_files=True,
    )
    if not uploaded:
        st.info("⬆️ Upload one or more comic pages to begin.")
        return

    pages = _load_uploads(uploaded)
    st.success(f"Loaded {len(pages)} page(s).")

    order = detect_reading_order(pages) if reading_order == "auto" else reading_order

    tab_panels, tab_video = st.tabs(["🔍 Panel detection", "🎞️ Generate video"])

    with tab_panels:
        st.write(f"Detected reading order: **{order}**")
        cfg = PipelineConfig()
        total = 0
        for idx, page in enumerate(pages):
            if order == "vertical":
                boxes = detect_panels_vertical_scroll(page, cfg.panels)
            else:
                boxes = detect_panels(page, cfg.panels)
            boxes = sort_panels_by_reading_order(boxes, order, page.height)
            total += len(boxes)
            st.image(
                _draw_boxes(page, boxes, order),
                caption=f"Page {idx + 1}: {len(boxes)} panels",
                use_container_width=True,
            )
        st.metric("Total panels", total)

    with tab_video:
        st.write("Render the full pipeline into an MP4.")
        if use_ai:
            st.warning("AI animation is slow and memory-hungry; the page may take minutes.")
        if st.button("🚀 Generate video", type="primary"):
            cfg = PipelineConfig()
            cfg.animation.seconds_per_panel = seconds
            cfg.output.fps = int(fps)
            if reading_order != "auto":
                cfg.input.reading_order = reading_order

            workdir = Path(tempfile.mkdtemp(prefix="acmp_demo_"))
            in_dir = workdir / "pages"
            in_dir.mkdir()
            for i, page in enumerate(pages):
                page.save(in_dir / f"page_{i:03d}.png")
            out_path = workdir / "out.mp4"

            from acmp.pipeline import process_chapter

            with st.spinner("Rendering… (panel detection → scene analysis → animation → encode)"):
                process_chapter(
                    input_path=in_dir,
                    output_path=out_path,
                    config=cfg,
                    use_ai=use_ai,
                    llm_prefer=llm,
                )
            st.success("Done!")
            st.video(str(out_path))
            st.download_button("⬇️ Download MP4", out_path.read_bytes(), "acmp.mp4", "video/mp4")


if __name__ == "__main__":
    main()
