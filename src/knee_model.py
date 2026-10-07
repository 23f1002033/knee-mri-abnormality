import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def submission_frame(ids, probs):
    df = pd.DataFrame(np.clip(np.asarray(probs, dtype=float), 0.0, 1.0), columns=LABELS)
    df.insert(0, "StudyInstanceUID", list(ids))
    return df


def check_submission(sub, sample, test_ids):
    problems = []
    if list(sub.columns) != list(sample.columns):
        problems.append("columns differ from sample_submission")
    if set(sub["StudyInstanceUID"]) != set(test_ids):
        problems.append("study ids differ from test.csv")
    if sub["StudyInstanceUID"].duplicated().any():
        problems.append("duplicate study ids")
    if sub[LABELS].isna().any().any():
        problems.append("missing predictions")
    return problems
