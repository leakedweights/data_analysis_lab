# Exploratory Data Analysis

Analysis of the SCLDDoS2024 training data (SetA + SetB): 264,770 events and 1,820,091 components.

Plots are in `plots/png/` and `plots/svg/`. Regenerate with `uv run python -m src.eda`.

---

## Q1: How severe is the class imbalance?

![](../plots/png/01_class_imbalance.png)

| Type | Count | Share |
|------|-------|-------|
| Normal traffic | 249,017 | 94.1% |
| Suspicious traffic | 13,380 | 5.1% |
| DDoS attack | 2,373 | 0.9% |

The dataset is heavily imbalanced. DDoS attacks — the class we care about most — make up less than 1% of events. The ratio between Normal and DDoS is roughly 105:1.

**Implication for modeling:** Standard classifiers will be biased toward predicting "Normal traffic." We will need resampling (SMOTE, undersampling), class weighting, or anomaly-detection-style approaches. Evaluation must use precision/recall/F1 per class rather than accuracy.

---

## Q2: Which features best distinguish DDoS attacks from normal traffic?

![](../plots/png/02_feature_medians.png)

![](../plots/png/03_feature_distributions.png)

### Strongest discriminating features

| Feature | Normal (median) | Suspicious (median) | DDoS (median) | Signal |
|---------|----------------|--------------------|--------------:|--------|
| **Avg packet len** | 1,278 | 1,254 | **143** | Very strong — DDoS packets are ~9x smaller |
| **Avg source IP count** | 1 | **7** | 1 | Distinguishes Suspicious from the other two |
| **Detect count** | 1 (mean 6) | 1 (mean 5) | 1 (mean **118**) | DDoS events have far more components on average |
| Packet speed | 64,550 | 64,200 | 65,426 | Weak at median, but DDoS has a much heavier tail (mean 157K vs 71K) |
| Data speed | 78 | 77 | 57 | Moderate — DDoS tends lower |

**Key insight:** `Avg packet len` is by far the strongest single signal. DDoS attacks use many small packets (median 143 bytes vs 1,278 for normal). This aligns with known DDoS patterns: amplification and flooding attacks send high volumes of small packets.

`Detect count` has identical medians (1) across all types, but the *mean* for DDoS is 118 — meaning a subset of DDoS events generate a massive number of detection components. This is a useful feature but its discrimination power is concentrated in the tail.

`Avg source IP count` is the best feature for identifying Suspicious traffic specifically (median 7 vs 1 for both Normal and DDoS).

---

## Q3: What are the attack types?

![](../plots/png/04_attack_types.png)

### DDoS event attack codes (left panel)

The top DDoS attack methods are:
1. **SYN Attack** (795 events, 33%) — classic TCP SYN flood
2. **DNS-related** (DNS, DNS + High volume traffic — combined ~460 events)
3. **NTP** (133) — NTP amplification
4. **Generic UDP** (122) — UDP flooding

These are well-known DDoS attack vectors. SYN floods dominate, followed by DNS and NTP amplification.

### Component-level attack codes (right panel)

At the component level, "High volume traffic" accounts for 94% of all detections (1.7M of 1.8M). The remaining 6% include Suspicious traffic, Generic UDP, DNS, and CLDAP. This extreme concentration means the component-level `Attack code` field has limited discriminative value on its own — most detections regardless of event type are flagged as "High volume traffic."

---

## Q4: What temporal patterns exist?

![](../plots/png/05_temporal_patterns.png)

![](../plots/png/06_event_duration.png)

### Volume patterns

- **Normal traffic** has a massive spike around December 2022 (~19K events in a single day), otherwise steady at ~300-1500/day.
- **Suspicious traffic** is concentrated in the first ~4 months (Aug-Nov 2022), with peaks of ~400/day, then drops to near-zero.
- **DDoS attacks** are sparse and bursty — most days have zero, but occasional spikes of 50-300 events appear. The largest bursts are in mid-2023.

**Implication:** The temporal distribution shifts significantly across the observation period. Suspicious traffic virtually disappears after early 2023, while DDoS spikes appear later. Models should not rely on time-based features or they will fail to generalize.

### Duration

DDoS events are significantly longer than normal traffic:
- DDoS median duration: **4 seconds** (mean 370s)
- Normal median duration: **1 second** (mean 48s)

The large gap between DDoS median (4s) and mean (370s) indicates that while most DDoS events are short, a subset are sustained attacks lasting minutes to hours.

---

## Q5: How are components related to events?

![](../plots/png/07_components_per_event.png)

- 62% of events have only 1 component (single detection)
- 38% have more than 1 component
- Only 4.3% have more than 10 components

However, **DDoS events average 118 components** compared to 6 for Normal. This means DDoS attacks trigger many more individual detections per event, which makes sense: a sustained attack generates repeated alerts over its duration.

**Implication:** Aggregating component-level features per event (e.g., component count, variance of packet sizes within an event) could yield strong features for classification.

---

## Q6: What is the correlation structure?

![](../plots/png/08_correlation.png)

Two correlation clusters emerge:
1. **Packet speed ↔ Data speed (0.84)** and **Packet speed ↔ Source IP count (0.67)**: Higher packet rates correlate with higher data rates and more source IPs.
2. **Avg packet len** is essentially uncorrelated with everything else (-0.09 to 0.03).
3. **Detect count** is independent of all traffic features (all < 0.05).

**Implication:** Packet speed and Data speed are largely redundant — consider dropping one or using PCA. Avg packet len and Detect count provide independent information, making them valuable for a classifier. No multicollinearity issues with the most discriminative features.

---

## Q7: Which ports are targeted?

![](../plots/png/09_port_analysis.png)

**Overall** top ports: 4500 (IKE/IPsec), 443 (HTTPS), 0 (reserved), 80 (HTTP), 53 (DNS).

**DDoS-specific** ports differ significantly: port **0** leads (619 events), followed by 443, 53, 10052 (Zabbix), and 80. Port 0 is unusual in normal traffic but common in DDoS (malformed/spoofed packets), making it a useful indicator.

---

## Q8: How concentrated is the victim targeting?

![](../plots/png/10_victim_concentration.png)

- ~25,000 unique victim IPs in the training data
- **777 IPs (3%) account for 80% of all events** — highly concentrated
- The single most-targeted DDoS victim (IP_50122) received 162 DDoS events

**Implication:** A small number of IPs are repeatedly targeted. Per-IP features (historical attack count, time since last event) could be strong signals, but may overfit to the training set IPs and fail on unseen victims in the test/genericity sets.

---

## Summary of Key Findings

1. **Extreme class imbalance (105:1)** requires careful handling — class weights, resampling, or appropriate metrics.
2. **Avg packet len is the strongest discriminator** — DDoS uses small packets (143 bytes vs 1,278 normal).
3. **Detect count (mean, not median)** separates DDoS well — DDoS events trigger ~20x more components.
4. **Avg source IP count** uniquely identifies Suspicious traffic.
5. **Temporal patterns are non-stationary** — Suspicious traffic disappears over time, DDoS is bursty. Avoid time-dependent features.
6. **Packet speed and Data speed are highly correlated (0.84)** — consider dimensionality reduction.
7. **Port 0 is a DDoS indicator** — rarely seen in normal traffic.
8. **Victim IP targeting is concentrated** — but may not generalize across sets.
