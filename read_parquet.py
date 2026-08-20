import pandas as pd

df = pd.read_parquet("global_cyclic_order.parquet")

# Keep even row positions only
df = df.iloc[1::2].reset_index(drop=True)

# Renumber question_id sequentially from 1
df["question_id"] = df.index + 1

df.to_parquet("global_cyclic_order.parquet", index=False)
print(df.to_string())