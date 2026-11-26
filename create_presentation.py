"""
create_presentation.py
Generates a minimal, modern, widescreen 16:9 PowerPoint presentation for TourLingo AI.
"""

import os
import sys

# Ensure site-packages are accessible
sys.path.insert(0, r"C:\Users\saksham tandon\anaconda3\Lib\site-packages")

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Clean Modern Palette
    BG_COLOR = RGBColor(248, 250, 252)       # Light Slate #F8FAFC
    PRIMARY_COLOR = RGBColor(30, 41, 59)     # Deep Slate #1E293B
    ACCENT_COLOR = RGBColor(37, 99, 235)     # Royal Blue #2563EB
    MUTED_COLOR = RGBColor(100, 116, 139)    # Slate Muted #64748B
    CARD_BG = RGBColor(255, 255, 255)        # Pure White
    BORDER_COLOR = RGBColor(226, 232, 240)   # Light border #E2E8F0

    blank_layout = prs.slide_layouts[6]

    def set_background(slide):
        bg_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg_shape.fill.solid()
        bg_shape.fill.fore_color.rgb = BG_COLOR
        bg_shape.line.fill.background() # No border
        return bg_shape

    def add_header(slide, title_text, category="CSE-AIML-P26 | NLP PROJECT"):
        # Category tag
        cat_box = slide.shapes.add_textbox(Inches(1.0), Inches(0.6), Inches(11.3), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_COLOR
        p_cat.font.name = "Segoe UI"

        # Main Title
        title_box = slide.shapes.add_textbox(Inches(1.0), Inches(0.95), Inches(11.3), Inches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(26)
        p_title.font.bold = True
        p_title.font.color.rgb = PRIMARY_COLOR
        p_title.font.name = "Segoe UI"

    # =========================================================================
    # SLIDE 1: TITLE SLIDE (Minimal & Sleek)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_background(s1)

    # Decorative subtle accent line
    line = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(2.2), Inches(1.5), Inches(0.08))
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT_COLOR
    line.line.fill.background()

    # Title & Subtitle Box
    tbox = s1.shapes.add_textbox(Inches(1.0), Inches(2.5), Inches(11.0), Inches(2.5))
    tf1 = tbox.text_frame
    tf1.word_wrap = True

    p1 = tf1.paragraphs[0]
    p1.text = "TourLingo AI"
    p1.font.size = Pt(46)
    p1.font.bold = True
    p1.font.color.rgb = PRIMARY_COLOR
    p1.font.name = "Segoe UI"

    p2 = tf1.add_paragraph()
    p2.text = "Multilingual AI Assistant for Tourist Information & Local Language Translation"
    p2.font.size = Pt(20)
    p2.font.color.rgb = ACCENT_COLOR
    p2.font.name = "Segoe UI"
    p2.space_before = Pt(12)

    # Metadata card
    card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(5.2), Inches(11.3), Inches(1.4))
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = BORDER_COLOR

    c_box = s1.shapes.add_textbox(Inches(1.3), Inches(5.35), Inches(10.7), Inches(1.1))
    c_tf = c_box.text_frame
    c_p1 = c_tf.paragraphs[0]
    c_p1.text = "Project ID: CSEAIML2026-P26  |  Domain: Natural Language Processing (NLP)"
    c_p1.font.size = Pt(13)
    c_p1.font.bold = True
    c_p1.font.color.rgb = PRIMARY_COLOR
    c_p1.font.name = "Segoe UI"

    c_p2 = c_tf.add_paragraph()
    c_p2.text = "Team Members: Member 1 (NLP & Intent)  •  Member 2 (NMT & Deep Learning)  •  Member 3 (Retrieval & Full-Stack)"
    c_p2.font.size = Pt(12)
    c_p2.font.color.rgb = MUTED_COLOR
    c_p2.font.name = "Segoe UI"
    c_p2.space_before = Pt(6)

    # =========================================================================
    # SLIDE 2: THE PROBLEM (Minimal Contrast)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_background(s2)
    add_header(s2, "Problem Statement: Challenges Faced by Tourists")

    col_data = [
        ("Language Barrier", "Foreign and domestic tourists struggle to converse with local vendors, auto drivers, and artisans who speak regional languages."),
        ("Information Fragmentation", "Crucial details like temple dress codes, camera fees, Friday closures, and genuine transport routes are scattered or inaccurate."),
        ("No Connectivity Fallback", "Most AI travel assistants fail completely when mobile cellular network drops in crowded bazaars or remote heritage monuments.")
    ]

    for i, (heading, desc) in enumerate(col_data):
        x = Inches(1.0 + i * 3.9)
        c = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(2.2), Inches(3.6), Inches(4.3))
        c.fill.solid()
        c.fill.fore_color.rgb = CARD_BG
        c.line.color.rgb = BORDER_COLOR

        t = s2.shapes.add_textbox(x + Inches(0.3), Inches(2.5), Inches(3.0), Inches(3.7))
        tf = t.text_frame
        tf.word_wrap = True

        p_num = tf.paragraphs[0]
        p_num.text = f"0{i+1}"
        p_num.font.size = Pt(28)
        p_num.font.bold = True
        p_num.font.color.rgb = ACCENT_COLOR
        p_num.font.name = "Segoe UI"

        p_h = tf.add_paragraph()
        p_h.text = heading
        p_h.font.size = Pt(18)
        p_h.font.bold = True
        p_h.font.color.rgb = PRIMARY_COLOR
        p_h.font.name = "Segoe UI"
        p_h.space_before = Pt(14)

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(13)
        p_d.font.color.rgb = MUTED_COLOR
        p_d.font.name = "Segoe UI"
        p_d.space_before = Pt(10)

    # =========================================================================
    # SLIDE 3: THE SOLUTION: TOURLINGO AI
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_background(s3)
    add_header(s3, "TourLingo AI: Solution Architecture Overview")

    pillars = [
        ("NLU Intent Engine", "Predicts tourist intent (attraction, transport, etiquette) and extracts destination entities."),
        ("Two-Way Translator", "Split-screen communication with voice playback (TTS) and fine-tuned Seq2Seq model."),
        ("Geo-Spatial Recommender", "Ranks nearby attractions and food using Haversine great-circle distance."),
        ("Offline Phrasebook", "Zero-network fuzzy string matcher with Roman phonetic pronunciation.")
    ]

    for i, (title, text) in enumerate(pillars):
        row = i // 2
        col = i % 2
        x = Inches(1.0 + col * 5.8)
        y = Inches(2.2 + row * 2.3)

        c = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(5.5), Inches(2.0))
        c.fill.solid()
        c.fill.fore_color.rgb = CARD_BG
        c.line.color.rgb = BORDER_COLOR

        t = s3.shapes.add_textbox(x + Inches(0.4), y + Inches(0.3), Inches(4.7), Inches(1.4))
        tf = t.text_frame
        tf.word_wrap = True

        pt = tf.paragraphs[0]
        pt.text = title
        pt.font.size = Pt(16)
        pt.font.bold = True
        pt.font.color.rgb = ACCENT_COLOR
        pt.font.name = "Segoe UI"

        pd = tf.add_paragraph()
        pd.text = text
        pd.font.size = Pt(12)
        pd.font.color.rgb = PRIMARY_COLOR
        pd.font.name = "Segoe UI"
        pd.space_before = Pt(6)

    # =========================================================================
    # SLIDE 4: CORE NLP PIPELINE
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_background(s4)
    add_header(s4, "Core NLP & Machine Learning Pipeline")

    steps = [
        ("1. Preprocessing", "Regex tokenization, customized stopword whitelisting, and Porter stemming."),
        ("2. Feature Extraction", "Sublinear TF-IDF (1 + log TF) capturing unigrams, bigrams, and trigrams (1, 3)."),
        ("3. Classification", "Calibrated LinearSVC finding optimal hyperplane across 8 tourist intent classes."),
        ("4. Semantic Retrieval", "Granular passage chunking (timings, etiquette, transport) via Cosine Similarity.")
    ]

    for i, (st, desc) in enumerate(steps):
        x = Inches(1.0 + i * 2.9)
        c = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(2.3), Inches(2.65), Inches(4.2))
        c.fill.solid()
        c.fill.fore_color.rgb = CARD_BG
        c.line.color.rgb = BORDER_COLOR

        t = s4.shapes.add_textbox(x + Inches(0.25), Inches(2.6), Inches(2.15), Inches(3.6))
        tf = t.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = st
        p1.font.size = Pt(15)
        p1.font.bold = True
        p1.font.color.rgb = PRIMARY_COLOR
        p1.font.name = "Segoe UI"

        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(12)
        p2.font.color.rgb = MUTED_COLOR
        p2.font.name = "Segoe UI"
        p2.space_before = Pt(12)

    # =========================================================================
    # SLIDE 5: DEEP LEARNING NMT MODEL
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_background(s5)
    add_header(s5, "Neural Machine Translation (NMT) Fine-Tuning")

    # Left card: Dataset & Architecture
    c_left = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(2.2), Inches(5.5), Inches(4.4))
    c_left.fill.solid()
    c_left.fill.fore_color.rgb = CARD_BG
    c_left.line.color.rgb = BORDER_COLOR

    t_l = s5.shapes.add_textbox(Inches(1.4), Inches(2.5), Inches(4.7), Inches(3.8))
    tf_l = t_l.text_frame
    tf_l.word_wrap = True

    p = tf_l.paragraphs[0]
    p.text = "Model & Training Setup"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = ACCENT_COLOR
    p.font.name = "Segoe UI"

    bullets_l = [
        "Dataset: IIT Bombay English-Hindi Corpus (10,000 pairs)",
        "Architecture: MarianMT (opus-mt-en-hi) Seq2Seq",
        "Parameters: 6 Encoder layers + 6 Decoder layers",
        "Optimizer: AdamW (lr = 2e-5) with linear schedule",
        "Decoding: 4-beam search with brevity penalty"
    ]
    for b in bullets_l:
        pb = tf_l.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(12)
        pb.font.color.rgb = PRIMARY_COLOR
        pb.space_before = Pt(8)

    # Right card: Training Convergence
    c_right = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(2.2), Inches(5.5), Inches(4.4))
    c_right.fill.solid()
    c_right.fill.fore_color.rgb = CARD_BG
    c_right.line.color.rgb = BORDER_COLOR

    t_r = s5.shapes.add_textbox(Inches(7.2), Inches(2.5), Inches(4.7), Inches(3.8))
    tf_r = t_r.text_frame
    tf_r.word_wrap = True

    p_r = tf_r.paragraphs[0]
    p_r.text = "Training Convergence Results"
    p_r.font.size = Pt(18)
    p_r.font.bold = True
    p_r.font.color.rgb = ACCENT_COLOR
    p_r.font.name = "Segoe UI"

    bullets_r = [
        "Total Training Steps: 1,875 steps (3 Full Epochs)",
        "Initial Loss: 3.377  ──►  Final Loss: 2.781 (-17.6%)",
        "Validation Loss: 3.985  ──►  3.956 (Consistent drop)",
        "Best Validation BLEU: 9.17 reached at Step 1,250",
        "Model Output: 303 MB PyTorch checkpoint saved locally"
    ]
    for b in bullets_r:
        pb = tf_r.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(12)
        pb.font.color.rgb = PRIMARY_COLOR
        pb.space_before = Pt(8)

    # =========================================================================
    # SLIDE 6: EVALUATION & RESULTS (Clean Stats)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_background(s6)
    add_header(s6, "Quantitative Evaluation & Key Metrics")

    stat_cards = [
        ("72.22%", "Intent Accuracy", "8 diverse tourist classes (Up from 53.8%)"),
        ("78.0%", "Macro Precision", "High confidence on tourist questions"),
        ("1.000", "Recommender MRR", "Gold standard spot ranked at Position 1"),
        ("100%", "Hit Rate@3", "Relevant recommendation in top-3 results")
    ]

    for i, (val, title, sub) in enumerate(stat_cards):
        x = Inches(1.0 + i * 2.9)
        c = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(2.2), Inches(2.65), Inches(2.1))
        c.fill.solid()
        c.fill.fore_color.rgb = CARD_BG
        c.line.color.rgb = BORDER_COLOR

        t = s6.shapes.add_textbox(x + Inches(0.2), Inches(2.35), Inches(2.25), Inches(1.8))
        tf = t.text_frame
        tf.word_wrap = True

        pv = tf.paragraphs[0]
        pv.text = val
        pv.font.size = Pt(28)
        pv.font.bold = True
        pv.font.color.rgb = ACCENT_COLOR
        pv.font.name = "Segoe UI"

        pt = tf.add_paragraph()
        pt.text = title
        pt.font.size = Pt(13)
        pt.font.bold = True
        pt.font.color.rgb = PRIMARY_COLOR
        pt.space_before = Pt(4)

        ps = tf.add_paragraph()
        ps.text = sub
        ps.font.size = Pt(10)
        ps.font.color.rgb = MUTED_COLOR
        ps.space_before = Pt(2)

    # Bottom summary box
    summary_box = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(4.7), Inches(11.3), Inches(1.9))
    summary_box.fill.solid()
    summary_box.fill.fore_color.rgb = CARD_BG
    summary_box.line.color.rgb = BORDER_COLOR

    st_box = s6.shapes.add_textbox(Inches(1.4), Inches(4.9), Inches(10.5), Inches(1.5))
    st_tf = st_box.text_frame
    st_tf.word_wrap = True

    p_sum_title = st_tf.paragraphs[0]
    p_sum_title.text = "Key Academic Achievements:"
    p_sum_title.font.size = Pt(14)
    p_sum_title.font.bold = True
    p_sum_title.font.color.rgb = PRIMARY_COLOR

    p_sum_body = st_tf.add_paragraph()
    p_sum_body.text = "• Perfect 100% precision on safety-critical classes: Emergency SOS (F1: 0.83) and Currency & Budget (F1: 0.75).\n• Food recommendation recall reached 88% with zero false alarms on religious etiquette queries.\n• Machine translation evaluated with SacreBLEU and chrF across multi-language pairs."
    p_sum_body.font.size = Pt(11)
    p_sum_body.font.color.rgb = MUTED_COLOR
    p_sum_body.space_before = Pt(4)

    # =========================================================================
    # SLIDE 7: THREE-WAY TEAM RESPONSIBILITIES
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_background(s7)
    add_header(s7, "Team Contribution & Role Division")

    roles = [
        ("Member 1", "NLP & Intent Classification", [
            "Text normalization & PorterStemmer",
            "Trigram TF-IDF (1, 3) feature extraction",
            "Calibrated LinearSVC Classifier",
            "Rule-based slot-filling entity extractor",
            "Achieved 72.2% intent test accuracy"
        ]),
        ("Member 2", "NMT Model & Offline Logic", [
            "MarianMT Seq2Seq Transformer model",
            "Training on IIT Bombay Parallel Corpus",
            "1,875 steps, loss: 3.37 ➔ 2.78",
            "Fuzzy Levenshtein & Jaccard phrase matcher",
            "Roman phonetic transliteration engine"
        ]),
        ("Member 3", "Retrieval & Full-Stack System", [
            "Granular passage chunking for RAG Q&A",
            "Cosine Similarity in Vector Space",
            "Haversine geodesic distance recommender",
            "Flask REST APIs + Web Speech (TTS/STT)",
            "Automated test suite (All 6 tests passing)"
        ])
    ]

    for i, (m_name, domain, tasks) in enumerate(roles):
        x = Inches(1.0 + i * 3.9)
        c = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(2.2), Inches(3.6), Inches(4.4))
        c.fill.solid()
        c.fill.fore_color.rgb = CARD_BG
        c.line.color.rgb = BORDER_COLOR

        t = s7.shapes.add_textbox(x + Inches(0.3), Inches(2.4), Inches(3.0), Inches(4.0))
        tf = t.text_frame
        tf.word_wrap = True

        pm = tf.paragraphs[0]
        pm.text = m_name
        pm.font.size = Pt(18)
        pm.font.bold = True
        pm.font.color.rgb = ACCENT_COLOR

        pd = tf.add_paragraph()
        pd.text = domain
        pd.font.size = Pt(12)
        pd.font.bold = True
        pd.font.color.rgb = PRIMARY_COLOR
        pd.space_before = Pt(2)

        for tk in tasks:
            ptk = tf.add_paragraph()
            ptk.text = "• " + tk
            ptk.font.size = Pt(10.5)
            ptk.font.color.rgb = MUTED_COLOR
            ptk.space_before = Pt(6)

    # =========================================================================
    # SLIDE 8: CONCLUSION & DEMO
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_background(s8)
    add_header(s8, "Conclusion & Live Demonstration")

    c_concl = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(2.2), Inches(11.3), Inches(4.4))
    c_concl.fill.solid()
    c_concl.fill.fore_color.rgb = CARD_BG
    c_concl.line.color.rgb = BORDER_COLOR

    t_c = s8.shapes.add_textbox(Inches(1.5), Inches(2.5), Inches(10.3), Inches(3.8))
    tf_c = t_c.text_frame
    tf_c.word_wrap = True

    p_end = tf_c.paragraphs[0]
    p_end.text = "Summary of TourLingo AI"
    p_end.font.size = Pt(20)
    p_end.font.bold = True
    p_end.font.color.rgb = ACCENT_COLOR

    points = [
        "Complete Solution: Successfully bridges language, cultural, and informational barriers for travelers.",
        "Rigorously Evaluated: Achieved 72.2% intent classification accuracy, 1.000 MRR on recommendations, and 1,875-step NMT convergence.",
        "Resilient & Practical: Fully functional offline phrasebook ensures safety and survival even with zero cellular connectivity.",
        "Modern Interface: Responsive web app featuring dual-screen vendor chat, audio synthesis, and live NLP inspection."
    ]

    for pt in points:
        p = tf_c.add_paragraph()
        p.text = "✔ " + pt
        p.font.size = Pt(13)
        p.font.color.rgb = PRIMARY_COLOR
        p.space_before = Pt(10)

    p_demo = tf_c.add_paragraph()
    p_demo.text = "\nThank You!  ──  Ready for Live System Demonstration & Viva Questions"
    p_demo.font.size = Pt(14)
    p_demo.font.bold = True
    p_demo.font.color.rgb = ACCENT_COLOR
    p_demo.space_before = Pt(14)

    output_path = os.path.join(os.path.dirname(__file__), "TourLingo_AI_Presentation.pptx")
    prs.save(output_path)
    print(f"Presentation generated successfully: {output_path}")

if __name__ == "__main__":
    create_presentation()
