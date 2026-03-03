## Lab02-Intro: Haladó adatelemzési módszerek laboratórium

### Preliminary Schedule (Előzetes ütemterv)

The course follows a structured timeline involving online consultations and in-person (jelenléti) milestone presentations:

| Week | Type | Description |
| --- | --- | --- |
| **1** | Online | Introduction, task description, and assignment.
| **2** | Online | Consultation during class time (Today).
| **3** | In-person | <br>**M1:** Project plan presentation, Data preparation plan.
| **4** | Online | Consultation during class time.
| **5** | In-person | <br>**M2:** Data preparation presentation, Data visualization plan.
| **6** | Online | Simonyi Conference (Dean's break); consultation outside class time.
| **7** | In-person | <br>**M3:** Data visualization presentation, Feature engineering/GT models plan.
| **8** | - | Easter Break.
| **9** | Online | Consultation during class time.
| **10** | In-person | <br>**M4:** GT models presentation, GT evaluation plan.
| **11** | Online | Consultation during class time.
| **12** | In-person | <br>**M5:** GT evaluation presentation, Application plan.
| **13** | Online | Consultation during class time.
| **14** | In-person | <br>**M6:** Application presentation.
| **15** | - | Opportunity for makeup.

---

### Dataset Overview

The data is provided in `.gzip` compressed CSV files.

**File Distribution:**

* **Training (Betanításhoz):** Sets A and B (components and events) .


* **Testing (Teszteléshez):** Set C (components and events) .


* **Genericity Check (Generikusság ellenőrzéséhez):** Set D (components and events) .


#### Data Structure

* Attack ID: egyedi esemény azonosító
* Detect count: az összetevő azonosító száma (az eseményen belül)
* Card: hálózati kártya azonosító
* Victim IP: anonimizált cél IP
* Port number: cél port száma
* Attack code: az összetevő jellege
* Significant flag: a DDoS detektor belső használatú flagje (számunkra irreleváns)
* Packet speed: csomagráta (pps)
* Data speed: adatráta (bps)
* Avg packet len: átlagos csomaghossz (byte)
* Source IP count: forrás IP-k száma
* Time: az esemény összetevőjének kezdő ideje (dátummal)

---