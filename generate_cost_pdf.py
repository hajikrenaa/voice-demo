# -*- coding: utf-8 -*-
"""Generate the professional Voice AI cost report as a submittable PDF."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, ListFlowable, ListItem
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---- Fonts (Arial on Win11 includes the rupee glyph) ----
try:
    pdfmetrics.registerFont(TTFont('AR', 'C:/Windows/Fonts/arial.ttf'))
    pdfmetrics.registerFont(TTFont('AR-B', 'C:/Windows/Fonts/arialbd.ttf'))
    pdfmetrics.registerFont(TTFont('AR-I', 'C:/Windows/Fonts/ariali.ttf'))
    pdfmetrics.registerFontFamily('AR', normal='AR', bold='AR-B', italic='AR-I')
    BASE, BOLD, ITAL = 'AR', 'AR-B', 'AR-I'
except Exception:
    BASE, BOLD, ITAL = 'Helvetica', 'Helvetica-Bold', 'Helvetica-Oblique'

R = '₹'  # rupee

NAVY = colors.HexColor('#1F3A5F')
TEAL = colors.HexColor('#2C7A7B')
LIGHT = colors.HexColor('#EAF0F6')
GREY = colors.HexColor('#666666')
LINEC = colors.HexColor('#B8C4D0')

styles = getSampleStyleSheet()
def S(name, **kw):
    return ParagraphStyle(name, **kw)

title_st = S('t', fontName=BOLD, fontSize=20, textColor=NAVY, leading=24)
sub_st = S('s', fontName=BASE, fontSize=9.5, textColor=GREY, leading=14)
h2_st = S('h2', fontName=BOLD, fontSize=12.5, textColor=NAVY, spaceBefore=14, spaceAfter=6, leading=15)
body_st = S('b', fontName=BASE, fontSize=9.5, textColor=colors.black, leading=14, alignment=TA_LEFT)
note_st = S('n', fontName=ITAL, fontSize=8.5, textColor=GREY, leading=12)
cell = S('c', fontName=BASE, fontSize=8.8, leading=11)
cellb = S('cb', fontName=BOLD, fontSize=8.8, leading=11)
cellr = S('cr', fontName=BASE, fontSize=8.8, leading=11, alignment=2)
cellrb = S('crb', fontName=BOLD, fontSize=8.8, leading=11, alignment=2)
hcell = S('hc', fontName=BOLD, fontSize=8.8, leading=11, textColor=colors.white)
hcellr = S('hcr', fontName=BOLD, fontSize=8.8, leading=11, textColor=colors.white, alignment=2)

def P(t, st=body_st): return Paragraph(t, st)

def make_table(data, col_widths, header=True, zebra=True):
    t = Table(data, colWidths=col_widths, hAlign='LEFT')
    cmds = [
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
        ('LINEBELOW', (0,0), (-1,-1), 0.4, LINEC),
        ('LINEAFTER', (0,0), (-2,-1), 0.4, LINEC),
        ('BOX', (0,0), (-1,-1), 0.6, LINEC),
    ]
    if header:
        cmds += [('BACKGROUND', (0,0), (-1,0), NAVY)]
    if zebra:
        for r0 in range(1, len(data)):
            if r0 % 2 == 0:
                cmds.append(('BACKGROUND', (0,r0), (-1,r0), LIGHT))
    t.setStyle(TableStyle(cmds))
    return t

story = []

# ---------- Header ----------
story.append(P('Voice AI Agent &mdash; Operating Cost Analysis', title_st))
story.append(Spacer(1, 4))
story.append(HRFlowable(width='100%', thickness=2, color=TEAL, spaceAfter=8))
story.append(P('<b>Prepared by:</b> Hajik &nbsp;&nbsp;|&nbsp;&nbsp; <b>Date:</b> 11 June 2026', sub_st))
story.append(P(f'<b>FX reference:</b> USD &rarr; INR @ {R}95.26 (market rate, 11 Jun 2026) &nbsp;&nbsp;|&nbsp;&nbsp; '
               '<b>Pricing basis:</b> Public list / pay-as-you-go rates, June 2026', sub_st))
story.append(Spacer(1, 10))

# ---------- 1. Executive Summary ----------
story.append(P('1.&nbsp;&nbsp;Executive Summary', h2_st))
story.append(P('This report sets out the full operating cost of the AI voice-calling agent &mdash; covering the AI / voice '
               'processing stack, AWS cloud infrastructure, and telephony.', body_st))
story.append(Spacer(1, 4))
exec_items = [
    f'<b>AI + voice cost per minute:</b> {R}4.88 on the current model, reducing to {R}3.11 on the optimised model '
    '(gpt-realtime-mini) &mdash; a ~36% unit-cost reduction.',
    f'<b>AWS infrastructure:</b> a fixed ~{R}1,400 / month, which becomes negligible per-minute as call volume grows.',
    '<b>Telephony (Vobiz):</b> a per-minute pass-through, typically the largest single variable; shown parametrically '
    'pending the carrier&rsquo;s confirmed rate, with an illustrative all-in worked example in &sect;7&ndash;8.',
    f'<b>Indicative all-in cost</b> (incl. illustrative {R}0.50/min telephony, moderate volume): '
    f'{R}5.66/min current &rarr; {R}3.89/min optimised.',
]
story.append(ListFlowable([ListItem(P(x, body_st), leftIndent=6) for x in exec_items],
                          bulletType='bullet', start='square', bulletColor=TEAL, bulletFontSize=6))

# ---------- 2. Scope & Architecture ----------
story.append(P('2.&nbsp;&nbsp;Scope &amp; Architecture', h2_st))
story.append(P('The platform places outbound AI voice calls and conducts a scripted, interactive conversation. '
               'Each minute of call time consumes the following metered services:', body_st))
story.append(Spacer(1, 4))
arch = [
    [P('Layer', hcell), P('Provider / Service', hcell), P('Role', hcell)],
    [P('Dialogue &amp; speech understanding', cell), P('OpenAI Realtime API (gpt-realtime)', cell), P('Understands caller audio, generates the agent&rsquo;s replies as text', cell)],
    [P('Speech transcription', cell), P('OpenAI Whisper-1', cell), P('Produces the caller-side transcript for call records', cell)],
    [P('Speech synthesis (voice)', cell), P('ElevenLabs Flash v2.5', cell), P('Converts the agent&rsquo;s text into natural speech', cell)],
    [P('Cloud hosting', cell), P('AWS EC2 (ap-south-1, Mumbai)', cell), P('Runs the application backend + web dashboard', cell)],
    [P('Telephony', cell), P('Vobiz', cell), P('Carries the call over the phone network', cell)],
]
story.append(make_table(arch, [4.2*cm, 4.6*cm, 8.0*cm]))

# ---------- 3. Unit Cost Per Minute ----------
story.append(P('3.&nbsp;&nbsp;Unit Cost &mdash; Per Minute of Call', h2_st))
unit = [
    [P('Component', hcell), P('Rate (list)', hcell), P('Current<br/>gpt-realtime', hcellr), P('Optimised<br/>gpt-realtime-mini', hcellr)],
    [P('OpenAI &mdash; audio input', cell), P('$32 / $10 per 1M tok', cell), P(f'{R}1.83', cellr), P(f'{R}0.57', cellr)],
    [P('OpenAI &mdash; text output', cell), P('$24 / $2.40 per 1M tok', cell), P(f'{R}0.57', cellr), P(f'{R}0.06', cellr)],
    [P('Whisper-1 transcription', cell), P('$0.006 / min', cell), P(f'{R}0.57', cellr), P(f'{R}0.57', cellr)],
    [P('ElevenLabs Flash v2.5', cell), P('$0.05 / 1k chars', cell), P(f'{R}1.90', cellr), P(f'{R}1.90', cellr)],
    [P('AI + voice sub-total / min', cellb), P('', cell), P(f'{R}4.88', cellrb), P(f'{R}3.11', cellrb)],
]
t3 = make_table(unit, [5.6*cm, 4.4*cm, 3.4*cm, 3.4*cm])
t3.setStyle(TableStyle([('BACKGROUND', (0,5), (-1,5), colors.HexColor('#D9E6D9'))]))
story.append(t3)
story.append(Spacer(1, 3))
story.append(P('Assumptions: ~600 audio-input tokens/min, ~250 text-output tokens/min, ~400 characters of synthesised '
               'speech/min. Usage-based; varies with conversation length and verbosity. USD equivalents: $0.051/min '
               '(current), $0.033/min (optimised).', note_st))

# ---------- 4. AWS ----------
story.append(P('4.&nbsp;&nbsp;AWS Cloud Infrastructure &mdash; Fixed Monthly', h2_st))
aws = [
    [P('Item', hcell), P('Basis', hcell), P('USD / mo', hcellr), P('INR / mo', hcellr)],
    [P('EC2 compute (t2.micro)', cell), P('~$0.0124/hr &times; 730 hr', cell), P('$9.05', cellr), P(f'{R}862', cellr)],
    [P('EBS storage (20 GB, gp3)', cell), P('~$0.0924 / GB-mo', cell), P('$1.85', cellr), P(f'{R}176', cellr)],
    [P('Public IPv4 address', cell), P('$0.005/hr (AWS, since 2024)', cell), P('$3.65', cellr), P(f'{R}348', cellr)],
    [P('Data transfer out', cell), P('~64 kbps audio; minimal', cell), P('~$0.10&ndash;1.00', cellr), P(f'{R}10&ndash;95', cellr)],
    [P('AWS total', cellb), P('', cell), P('~$14.7', cellrb), P(f'~{R}1,400', cellrb)],
]
t4 = make_table(aws, [5.0*cm, 5.4*cm, 3.2*cm, 3.2*cm])
t4.setStyle(TableStyle([('BACKGROUND', (0,5), (-1,5), colors.HexColor('#D9E6D9'))]))
story.append(t4)
story.append(Spacer(1, 3))
story.append(P('Instance: t2.micro (1 vCPU, 1 GB RAM, 20 GB EBS), ap-south-1 (Mumbai), on-demand Linux. AWS is a fixed '
               'cost, independent of call volume; per-minute it falls sharply as usage rises (see &sect;6). Free-tier '
               'eligibility, if any, is excluded as it is time-limited.', note_st))

# ---------- 5. Telephony ----------
story.append(P('5.&nbsp;&nbsp;Telephony (Vobiz) &mdash; Pass-Through Variable', h2_st))
story.append(P('Vobiz is billed per connected minute at a rate set by the carrier and destination. As the confirmed rate '
               'is pending, the table shows the all-in impact at representative rates:', body_st))
story.append(Spacer(1, 4))
tel = [
    [P('Vobiz rate', hcell), P('Added cost / min', hcellr), P('Impact on a 2.5-min call', hcellr)],
    [P(f'{R}0.25 / min', cell), P(f'{R}0.25', cellr), P(f'{R}0.63', cellr)],
    [P(f'{R}0.50 / min', cell), P(f'{R}0.50', cellr), P(f'{R}1.25', cellr)],
    [P(f'{R}1.00 / min', cell), P(f'{R}1.00', cellr), P(f'{R}2.50', cellr)],
    [P(f'{R}2.00 / min', cell), P(f'{R}2.00', cellr), P(f'{R}5.00', cellr)],
]
story.append(make_table(tel, [5.0*cm, 5.9*cm, 5.9*cm]))

# ---------- 6. Blended per minute ----------
story.append(P('6.&nbsp;&nbsp;Blended Cost per Minute, by Volume (excl. telephony)', h2_st))
blend = [
    [P('Monthly volume', hcell), P('AWS / min', hcellr), P('Current / min', hcellr), P('Optimised / min', hcellr)],
    [P('1,250 min (~500 calls)', cell), P(f'{R}1.12', cellr), P(f'{R}6.00', cellr), P(f'{R}4.23', cellr)],
    [P('5,000 min (~2,000 calls)', cell), P(f'{R}0.28', cellr), P(f'{R}5.16', cellr), P(f'{R}3.39', cellr)],
    [P('20,000 min (~8,000 calls)', cell), P(f'{R}0.07', cellr), P(f'{R}4.95', cellr), P(f'{R}3.18', cellr)],
]
story.append(make_table(blend, [5.8*cm, 3.4*cm, 3.7*cm, 3.9*cm]))
story.append(Spacer(1, 3))
story.append(P('Call estimates assume an average call length of 2.5 minutes.', note_st))

# ---------- 7. Indicative all-in ----------
story.append(P('7.&nbsp;&nbsp;Indicative All-In Cost (incl. telephony)', h2_st))
story.append(P(f'Worked example at an <b>illustrative Vobiz rate of {R}0.50/min</b> &mdash; replace with the confirmed '
               'carrier rate for the final figure:', body_st))
story.append(Spacer(1, 4))
allin = [
    [P('Monthly volume', hcell), P('AI + AWS / min', hcellr), P(f'+ Vobiz ({R}0.50)', hcellr), P('All-in / min', hcellr), P('All-in / 2.5-min call', hcellr)],
    [P('1,250 min', cell), P(f'{R}6.00', cellr), P(f'{R}0.50', cellr), P(f'{R}6.50', cellrb), P(f'{R}16.25', cellr)],
    [P('5,000 min', cell), P(f'{R}5.16', cellr), P(f'{R}0.50', cellr), P(f'{R}5.66', cellrb), P(f'{R}14.15', cellr)],
    [P('20,000 min', cell), P(f'{R}4.95', cellr), P(f'{R}0.50', cellr), P(f'{R}5.45', cellrb), P(f'{R}13.63', cellr)],
]
story.append(make_table(allin, [3.2*cm, 3.2*cm, 3.0*cm, 3.0*cm, 4.4*cm]))
story.append(Spacer(1, 3))
story.append(P(f'Optimised model (gpt-realtime-mini) all-in /min at {R}0.50 Vobiz: {R}4.73 (1,250 min), {R}3.89 '
               f'(5,000 min), {R}3.68 (20,000 min).', note_st))

# ---------- 8. Monthly projection ----------
story.append(P('8.&nbsp;&nbsp;Monthly Cost Projection', h2_st))
proj = [
    [P('Scenario', hcell), P('Minutes/mo', hcellr), P('Current<br/>(AI+AWS)', hcellr), P('Optimised<br/>(AI+AWS)', hcellr), P(f'All-in current*<br/>(incl. Vobiz {R}0.50)', hcellr)],
    [P('Pilot', cell), P('1,250', cellr), P(f'{R}7,500', cellr), P(f'{R}5,288', cellr), P(f'{R}8,125', cellr)],
    [P('Growth', cell), P('5,000', cellr), P(f'{R}25,800', cellr), P(f'{R}16,950', cellr), P(f'{R}28,300', cellr)],
    [P('Scale', cell), P('20,000', cellr), P(f'{R}99,000', cellr), P(f'{R}63,600', cellr), P(f'{R}1,09,000', cellr)],
]
story.append(make_table(proj, [2.6*cm, 2.6*cm, 3.2*cm, 3.2*cm, 5.2*cm]))
story.append(Spacer(1, 3))
story.append(P('*Illustrative all-in uses Vobiz @ ' + R + '0.50/min on the current model; replace with the confirmed '
               'carrier rate. Add telephony = (Vobiz rate × total minutes).', note_st))

# ---------- 9. Optimisation ----------
story.append(P('9.&nbsp;&nbsp;Cost-Optimisation Roadmap', h2_st))
opt = [
    [P('Lever', hcell), P('Est. saving', hcellr), P('Trade-off', hcell)],
    [P('Switch to gpt-realtime-mini', cell), P(f'{R}1.77 / min (~36%)', cellr), P('Slightly lower reasoning; validate on scripted flow', cell)],
    [P('Reduce average call duration', cell), P('Proportional, all costs', cellr), P('Tighter scripting (also cuts telephony)', cell)],
    [P('Disable caller transcription (if unused)', cell), P(f'{R}0.57 / min', cellr), P('Loses caller-side transcript records', cell)],
    [P('ElevenLabs subscription tier (at volume)', cell), P('Lower per-char rate', cellr), P('Monthly commitment', cell)],
    [P('Optimise prompt caching', cell), P('Marginal, free', cellr), P('None', cell)],
]
story.append(make_table(opt, [5.6*cm, 3.6*cm, 7.6*cm]))

# ---------- 10. Assumptions ----------
story.append(P('10.&nbsp;&nbsp;Assumptions &amp; Disclaimers', h2_st))
disc = [
    'Token / character consumption is usage-based and varies with conversation length and style; figures use moderate-conversation averages.',
    'AI provider rates are public list / pay-as-you-go (June 2026); volume or committed-use discounts may apply.',
    'AWS instance type inferred from server specs (1 vCPU / 1 GB &rarr; t2.micro); free-tier eligibility excluded as time-limited.',
    f'Telephony (Vobiz) confirmed rate pending; all-in figures use an illustrative {R}0.50/min and must be updated with the carrier rate.',
    'FX rate as stated; INR figures move with the exchange rate.',
]
story.append(ListFlowable([ListItem(P(x, body_st), leftIndent=6) for x in disc],
                          bulletType='1', bulletFormat='%s.', bulletFontName=BOLD, bulletColor=NAVY))

# ---------- Footer ----------
def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINEC)
    canvas.setLineWidth(0.5)
    canvas.line(2*cm, 1.4*cm, A4[0]-2*cm, 1.4*cm)
    canvas.setFont(BASE, 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(2*cm, 1.0*cm, 'Voice AI Agent — Operating Cost Analysis  |  Prepared by Hajik  |  Confidential')
    canvas.drawRightString(A4[0]-2*cm, 1.0*cm, 'Page %d' % doc.page)
    canvas.restoreState()

doc = SimpleDocTemplate(
    'D:/voice testing for twillow/Voice_AI_Cost_Report.pdf', pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm,
    title='Voice AI Agent - Operating Cost Analysis', author='Hajik',
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print('PDF written: Voice_AI_Cost_Report.pdf')
