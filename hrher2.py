import pandas as pd


def hrher2_slide_stems(clinical_csv=None):
    if clinical_csv is None:
        from paths import CLINICAL
        clinical_csv = CLINICAL
    df = pd.read_csv(clinical_csv)
    hr = (df["ER_Status_By_IHC"] == "positive") | (df["PR"] == 1)
    her2neg = df["HER2Calc"] == "negative"
    sub = df[hr & her2neg]
    return set(sub["slide"].astype(str))


if __name__ == "__main__":
    stems = hrher2_slide_stems()
    print(f"HR+/HER2- slides: {len(stems)}")
    import collections
    patients = {s[:12] for s in stems}
    print(f"HR+/HER2- unique patients: {len(patients)}  (paper n=535)")
