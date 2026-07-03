"""
reports/pdf_generator.py
=========================
Professional PDF report generator using ReportLab.

Generates a per-detection PDF containing:
- Header with project branding
- Detection summary table
- Confidence score visualisation
- Grad-CAM image (if available)
- Frame-wise statistics (for videos)
- Footer with timestamp
"""

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class PDFReportGenerator:
    """
    Generates professional PDF detection reports.

    Args:
        output_dir: Directory where PDF files will be saved.
        logo_path:  Optional path to a logo image for the header.
    """

    # Brand colours (RGB 0-1)
    PRIMARY_R, PRIMARY_G, PRIMARY_B = 0.16, 0.50, 0.73
    ACCENT_R, ACCENT_G, ACCENT_B = 0.91, 0.30, 0.24
    DARK_R, DARK_G, DARK_B = 0.13, 0.13, 0.18
    LIGHT_R, LIGHT_G, LIGHT_B = 0.95, 0.95, 0.97

    def __init__(
        self,
        output_dir: str = "outputs/reports",
        logo_path: Optional[str] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logo_path = logo_path

    def generate(
        self,
        file_name: str,
        prediction: str,
        confidence: float,
        processing_time_ms: float,
        timestamp: Optional[datetime] = None,
        num_frames: int = 0,
        fake_frames: int = 0,
        real_frames: int = 0,
        gradcam_image: Optional[np.ndarray] = None,
        frame_confidences: Optional[List[float]] = None,
        model_name: str = "XceptionNet",
        notes: str = "",
    ) -> Path:
        """
        Generate a PDF report and save it to disk.

        Args:
            file_name:          Source file name being reported on.
            prediction:         ``"FAKE"`` or ``"REAL"``.
            confidence:         Confidence score (0–1).
            processing_time_ms: Total processing time in milliseconds.
            timestamp:          Detection datetime (defaults to now).
            num_frames:         Total frames analysed (videos only).
            fake_frames:        Frames classified as FAKE.
            real_frames:        Frames classified as REAL.
            gradcam_image:      Optional RGB Grad-CAM overlay array.
            frame_confidences:  Per-frame confidence list (for chart).
            model_name:         Model used for detection.
            notes:              Optional free-text notes.

        Returns:
            Path to the saved PDF file.
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm, mm
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table,
                TableStyle, Image as RLImage, HRFlowable,
            )
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
        except ImportError:
            logger.error("ReportLab not installed. Install with: pip install reportlab")
            raise

        if timestamp is None:
            timestamp = datetime.now()

        # Sanitise filename
        safe_name = Path(file_name).stem.replace(" ", "_")
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
        pdf_name = f"report_{safe_name}_{ts_str}.pdf"
        pdf_path = self.output_dir / pdf_name

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        styles = getSampleStyleSheet()
        elements = []

        PRIMARY = colors.Color(self.PRIMARY_R, self.PRIMARY_G, self.PRIMARY_B)
        ACCENT = colors.Color(self.ACCENT_R, self.ACCENT_G, self.ACCENT_B)
        DARK = colors.Color(self.DARK_R, self.DARK_G, self.DARK_B)
        LIGHT_BG = colors.Color(self.LIGHT_R, self.LIGHT_G, self.LIGHT_B)

        FAKE_COLOR = ACCENT
        REAL_COLOR = colors.Color(0.18, 0.69, 0.40)
        verdict_color = FAKE_COLOR if prediction == "FAKE" else REAL_COLOR

        # ── Header ────────────────────────────────────────────────────────────
        header_style = ParagraphStyle(
            "Header",
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=PRIMARY,
            alignment=TA_CENTER,
            spaceAfter=4,
        )
        sub_style = ParagraphStyle(
            "Sub",
            fontName="Helvetica",
            fontSize=10,
            textColor=colors.grey,
            alignment=TA_CENTER,
            spaceAfter=12,
        )

        elements.append(Paragraph("DeepFake Detection Report", header_style))
        elements.append(Paragraph("Powered by XceptionNet + Grad-CAM", sub_style))
        elements.append(HRFlowable(width="100%", thickness=2, color=PRIMARY))
        elements.append(Spacer(1, 0.4 * cm))

        # ── Verdict banner ────────────────────────────────────────────────────
        verdict_style = ParagraphStyle(
            "Verdict",
            fontName="Helvetica-Bold",
            fontSize=28,
            textColor=verdict_color,
            alignment=TA_CENTER,
            spaceAfter=4,
        )
        elements.append(Paragraph(f"🔍 {prediction}", verdict_style))
        conf_style = ParagraphStyle(
            "Conf",
            fontName="Helvetica",
            fontSize=14,
            textColor=verdict_color,
            alignment=TA_CENTER,
            spaceAfter=16,
        )
        elements.append(Paragraph(f"Confidence: {confidence:.1%}", conf_style))

        # ── Summary table ─────────────────────────────────────────────────────
        summary_data = [
            ["Field", "Value"],
            ["File Name", file_name],
            ["Prediction", prediction],
            ["Confidence", f"{confidence:.1%}"],
            ["Model", model_name],
            ["Processing Time", f"{processing_time_ms:.0f} ms"],
            ["Timestamp", timestamp.strftime("%Y-%m-%d %H:%M:%S")],
        ]

        if num_frames > 0:
            summary_data += [
                ["Frames Analysed", str(num_frames)],
                ["Fake Frames", f"{fake_frames}  ({fake_frames/num_frames:.1%})"],
                ["Real Frames", f"{real_frames}  ({real_frames/num_frames:.1%})"],
            ]

        if notes:
            summary_data.append(["Notes", notes])

        table_style = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("BACKGROUND", (0, 1), (-1, -1), LIGHT_BG),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("ROUNDEDCORNERS", [3, 3, 3, 3]),
        ])

        table = Table(summary_data, colWidths=[5 * cm, 12 * cm])
        table.setStyle(table_style)
        elements.append(table)
        elements.append(Spacer(1, 0.5 * cm))

        # ── Confidence bar ────────────────────────────────────────────────────
        section_style = ParagraphStyle(
            "Section",
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=DARK,
            spaceAfter=6,
        )
        elements.append(Paragraph("Confidence Score", section_style))

        bar_data = [["Real ←", f"{confidence:.1%}", "→ Fake"]]
        bar_table = Table(bar_data, colWidths=[3 * cm, 11 * cm, 3 * cm])
        bar_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (0, 0), "RIGHT"),
            ("ALIGN", (1, 0), (1, 0), "CENTER"),
            ("ALIGN", (2, 0), (2, 0), "LEFT"),
            ("BACKGROUND", (1, 0), (1, 0), verdict_color),
            ("TEXTCOLOR", (1, 0), (1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(bar_table)
        elements.append(Spacer(1, 0.5 * cm))

        # ── Grad-CAM image ────────────────────────────────────────────────────
        if gradcam_image is not None:
            elements.append(Paragraph("Grad-CAM Explanation", section_style))
            elements.append(
                Paragraph(
                    "Red/yellow regions indicate areas that most influenced "
                    "the DeepFake classification decision.",
                    styles["Normal"],
                )
            )
            elements.append(Spacer(1, 0.3 * cm))
            try:
                pil_img = Image.fromarray(gradcam_image.astype(np.uint8))
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format="JPEG", quality=85)
                img_buffer.seek(0)
                rl_img = RLImage(img_buffer, width=12 * cm, height=8 * cm)
                elements.append(rl_img)
                elements.append(Spacer(1, 0.4 * cm))
            except Exception as exc:
                logger.warning("Could not embed Grad-CAM image: %s", exc)

        # ── Frame confidence chart (text fallback) ────────────────────────────
        if frame_confidences and len(frame_confidences) > 0:
            elements.append(Paragraph("Frame-wise Fake Probability", section_style))
            avg = sum(frame_confidences) / len(frame_confidences)
            peak = max(frame_confidences)
            chart_data = [
                ["Metric", "Value"],
                ["Frames Analysed", str(len(frame_confidences))],
                ["Average Fake Prob", f"{avg:.3f}"],
                ["Peak Fake Prob", f"{peak:.3f}"],
                ["Frames > 50%", str(sum(1 for c in frame_confidences if c > 0.5))],
            ]
            chart_table = Table(chart_data, colWidths=[8 * cm, 9 * cm])
            chart_table.setStyle(table_style)
            elements.append(chart_table)
            elements.append(Spacer(1, 0.4 * cm))

        # ── Footer ────────────────────────────────────────────────────────────
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
        footer_style = ParagraphStyle(
            "Footer",
            fontName="Helvetica",
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER,
            spaceBefore=6,
        )
        elements.append(
            Paragraph(
                f"Generated by DeepFake Detection System v1.0 | "
                f"{timestamp.strftime('%B %d, %Y at %H:%M:%S')}",
                footer_style,
            )
        )

        doc.build(elements)
        logger.info("PDF report saved: %s", pdf_path)
        return pdf_path
