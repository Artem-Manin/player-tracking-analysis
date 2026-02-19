import pandas as pd

# ----------------------------------------
# Load files
# ----------------------------------------
rule_df = pd.read_csv("figures/gk_intervals_3min.csv")
cluster_df = pd.read_csv("figures/gk_intervals_clustered.csv")

rule_df["start"] = pd.to_datetime(rule_df["start"])
cluster_df["start"] = pd.to_datetime(cluster_df["start"])

# ----------------------------------------
# Normalize rule labels
# ----------------------------------------
rule_df["role_rule"] = rule_df["gk"].apply(
    lambda x: "GK" if x == "YES" else "Field"
)

# ----------------------------------------
# Merge
# ----------------------------------------
merged = pd.merge(
    rule_df[["start", "role_rule"]],
    cluster_df[["start", "role"]],
    on="start",
)

merged.rename(columns={"role": "role_cluster"}, inplace=True)

# ----------------------------------------
# Agreement
# ----------------------------------------
merged["agree"] = merged["role_rule"] == merged["role_cluster"]

agreement_rate = merged["agree"].mean() * 100

print(f"\nAgreement rate: {agreement_rate:.1f}%")

# ----------------------------------------
# Mismatches
# ----------------------------------------
mismatch = merged[merged["agree"] == False]

print("\nMismatched windows:")
print(mismatch[["start", "role_rule", "role_cluster"]])

# ----------------------------------------
# Save
# ----------------------------------------
merged.to_csv("figures/gk_method_comparison.csv", index=False)

print("\nSaved: figures/gk_method_comparison.csv")
