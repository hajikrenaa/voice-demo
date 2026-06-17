# Voice AI Agent — Operating Cost Analysis

**Prepared by:** Hajik
**Date:** 11 June 2026
**FX reference:** USD → INR @ **₹95.26** (RBI/market rate, 11 Jun 2026)
**Pricing basis:** Public list / pay-as-you-go rates, June 2026

---

## 1. Executive Summary

This report sets out the full operating cost of the AI voice-calling agent — covering the AI/voice processing stack, AWS cloud infrastructure, and telephony.

- **AI + voice cost per minute:** **₹4.88** on the current model, reducing to **₹3.11** on the optimised model (`gpt-realtime-mini`) — a **~36% unit-cost reduction**.
- **AWS infrastructure:** a **fixed ~₹1,400/month**, which becomes negligible per-minute as call volume grows.
- **Telephony (Vobiz):** a **per-minute pass-through** that is typically the largest single variable; presented parametrically below pending the carrier's confirmed rate.
- **Headline blended cost (excl. telephony), at moderate volume:** **₹5.16/min** (current) or **₹3.39/min** (optimised).

---

## 2. Scope & Architecture

The platform places outbound AI voice calls and conducts a scripted, interactive conversation. Each minute of call time consumes four metered services:

| Layer | Provider / Service | Role |
|---|---|---|
| Speech understanding + dialogue | OpenAI Realtime API (`gpt-realtime`) | Understands caller audio, generates the agent's replies as text |
| Speech transcription | OpenAI Whisper-1 | Produces the caller-side transcript for call records |
| Speech synthesis (voice) | ElevenLabs Flash v2.5 | Converts the agent's text into natural speech |
| Cloud hosting | AWS EC2 (ap-south-1, Mumbai) | Runs the application backend + web dashboard |
| Telephony | Vobiz | Carries the call over the phone network |

---

## 3. Unit Cost — Per Minute of Call

*Assumptions: ~600 audio-input tokens/min, ~250 text-output tokens/min, ~400 characters of synthesised speech/min. Usage-based; varies with conversation length and verbosity.*

| Component | Rate (list) | **Current — `gpt-realtime`** | **Optimised — `gpt-realtime-mini`** |
|---|---|---:|---:|
| OpenAI — audio input | $32 / $10 per 1M tok | ₹1.83 | ₹0.57 |
| OpenAI — text output | $24 / $2.40 per 1M tok | ₹0.57 | ₹0.06 |
| Whisper-1 transcription | $0.006 / min | ₹0.57 | ₹0.57 |
| ElevenLabs Flash v2.5 | $0.05 / 1k chars | ₹1.90 | ₹1.90 |
| **AI + voice sub-total / min** | | **₹4.88** | **₹3.11** |

*(USD equivalents: $0.051/min current; $0.033/min optimised.)*

---

## 4. AWS Cloud Infrastructure — Fixed Monthly

*Instance: t2.micro (1 vCPU, 1 GB RAM, 20 GB EBS), ap-south-1 (Mumbai), on-demand Linux.*

| Item | Basis | USD / month | INR / month |
|---|---|---:|---:|
| EC2 compute (t2.micro) | ~$0.0124/hr × 730 hr | $9.05 | ₹862 |
| EBS storage (20 GB, gp3) | ~$0.0924/GB-mo | $1.85 | ₹176 |
| Public IPv4 address | $0.005/hr (AWS, since 2024) | $3.65 | ₹348 |
| Data transfer out | ~64 kbps audio; volume-linked, minimal | ~$0.10–1.00 | ₹10–95 |
| **AWS total** | | **~$14.7** | **~₹1,400 / month** |

**Note:** AWS is a *fixed* cost, independent of call volume. Per-minute it falls sharply as usage rises (see §6). The current instance is modestly sized (1 GB RAM, ~58% disk used) — adequate for the present workload; a larger instance would be advisable only at significantly higher concurrency.

---

## 5. Telephony (Vobiz) — Pass-Through Variable

Vobiz is billed per connected minute at a rate set by the carrier and destination. As this is a direct pass-through and the confirmed rate is pending, the table below shows the **all-in impact at representative rates** so the figure can be slotted in:

| Vobiz rate | Added cost / min | Impact on a 2.5-min call |
|---|---:|---:|
| ₹0.25 / min | ₹0.25 | ₹0.63 |
| ₹0.50 / min | ₹0.50 | ₹1.25 |
| ₹1.00 / min | ₹1.00 | ₹2.50 |
| ₹2.00 / min | ₹2.00 | ₹5.00 |

> *Telephony is excluded from the blended totals in §6 and §7. Insert the confirmed Vobiz rate to obtain the final all-in figure.*

---

## 6. Blended Cost per Minute, by Volume *(excl. telephony)*

Because AWS is fixed, the effective per-minute cost depends on monthly volume:

| Monthly volume | AWS / min | **Current /min** | **Optimised /min** |
|---|---:|---:|---:|
| 1,250 min (~500 calls) | ₹1.12 | **₹6.00** | ₹4.23 |
| 5,000 min (~2,000 calls) | ₹0.28 | **₹5.16** | ₹3.39 |
| 20,000 min (~8,000 calls) | ₹0.07 | **₹4.95** | ₹3.18 |

*Call estimates assume an average call length of 2.5 minutes.*

---

## 7. Monthly Cost Projection *(AI + AWS, excl. telephony)*

| Scenario | Minutes/mo | **Current model** | **Optimised model** | Saving |
|---|---:|---:|---:|---:|
| Pilot | 1,250 | ₹7,500 | ₹5,288 | ₹2,212 |
| Growth | 5,000 | ₹25,800 | ₹16,950 | ₹8,850 |
| Scale | 20,000 | ₹99,000 | ₹63,600 | ₹35,400 |

*Add telephony = (Vobiz rate × total minutes).*

---

## 8. Cost-Optimisation Roadmap

| Lever | Est. saving | Trade-off |
|---|---:|---|
| Switch to `gpt-realtime-mini` | ₹1.77/min (~36%) | Slightly lower reasoning; validate on scripted flow |
| Reduce average call duration | ~Proportional across *all* costs incl. telephony | Tighter scripting |
| Disable caller transcription (if not needed) | ₹0.57/min | Loses caller-side transcript records |
| ElevenLabs subscription tier (at volume) | Lower effective per-character rate | Monthly commitment |
| Optimise prompt caching | Marginal, free | None |

---

## 9. Assumptions & Disclaimers

1. Token/character consumption is **usage-based** and varies with conversation length and style; figures use moderate-conversation averages.
2. AI provider rates are **public list / pay-as-you-go** (June 2026); volume or committed-use discounts may apply.
3. AWS instance type inferred from server specs (1 vCPU / 1 GB → t2.micro); free-tier eligibility, if any, is excluded as it is time-limited.
4. **Telephony (Vobiz) is excluded** from blended totals and shown parametrically pending the carrier's confirmed per-minute rate.
5. FX rate as stated; INR figures move with the exchange rate.

---

*End of report.*
