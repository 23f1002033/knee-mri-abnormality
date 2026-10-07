import json
import re

import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
D4_LABELS = ["Synovitis", "Contusion", "Fracture"]

SYSTEM = """You read knee MRI radiology reports and extract 12 findings for one knee.
Reports can be in any language. Answer only with JSON, no other text.

For each finding return [severity, certainty, evidence].
severity:
  0 = the report explicitly says it is normal, intact or absent; quote those words
  1 = minimal, trace, degenerative, intrasubstance signal, chronic or old
  2 = mild, small, low grade, grade 1
  3 = moderate, partial tear, grade 2, or a tear or finding that is present with no severity words
  4 = severe, large, massive, complete or full-thickness, grade 3 or 4, displaced
  null = the report does not mention this finding; never use 0 for an unmentioned finding
certainty: 2 = definite, 1 = probable or likely, 0 = possible, suspected, cannot be excluded
evidence: at most 8 words copied from the report in its original language, or ""

Findings:
ACL, MCL: injury or tear of that ligament.
Medial Meniscus, Lateral Meniscus: tear or abnormality of that meniscus. Signal that does not
  reach the surface is severity 1.
Medial OA, Lateral OA: cartilage loss of the femoral condyle or tibial plateau on that side.
PF OA: cartilage loss of the patella or trochlea.
Effusion: joint fluid. Synovitis: synovial thickening or inflammation.
Baker's: popliteal cyst. Contusion: bone bruise or marrow edema without a fracture line.
Fracture: fracture line or cortical break. An old or healed fracture is severity 1.

If the report covers both knees, use the knee the report is mainly about.
Return one JSON object with exactly these keys: """ + ", ".join(f'"{k}"' for k in LABELS) + """
Example value: "ACL": [4, 2, "complete tear of the ACL"]"""


def prompt(tok, report):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Report:\n" + report}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse(text):
    t = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        d = json.loads(t)
        out = {}
        for k in LABELS:
            v = d.get(k)
            ok = isinstance(v, list) and len(v) >= 2
            out[k] = (as_int(v[0]), as_int(v[1]), v[2] if len(v) > 2 else "") if ok else (None, None, "")
        return out, "json", 12
    except Exception:
        pass
    out, hits = {}, 0
    for k in LABELS:
        m = re.search('"' + re.escape(k) + r'"\s*:\s*\[\s*(null|-?\d+)\s*,\s*(null|-?\d+)'
                      r'(?:\s*,\s*"((?:[^"\\]|\\.)*)")?', t)
        if m:
            hits += 1
            out[k] = (as_int(m.group(1)), as_int(m.group(2)), m.group(3) or "")
        else:
            out[k] = (None, None, "")
    return (out, "regex", hits) if hits else (None, "fail", 0)


def score_table(lab, d4=False):
    out = pd.DataFrame({"StudyInstanceUID": lab["StudyInstanceUID"]})
    for k in LABELS:
        sev = lab[f"{k}|sev"]
        cert = lab[f"{k}|cert"].fillna(1)
        ev = lab[f"{k}|ev"].fillna("").astype(str).str.strip()
        found = lab[f"{k}|ev_found"].fillna(False).astype(bool)
        unmentioned = sev.isna() | ((sev == 0) & ((ev == "") | ~found))
        if d4 and k in D4_LABELS:
            unmentioned |= sev == 0
        s = (sev + 0.25 * cert).where(sev > 0, 0.0)
        out[k] = s.where(~unmentioned, 0.5)
    return out
