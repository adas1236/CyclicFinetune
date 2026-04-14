import polars as pl

df = pl.read_parquet("/fs/nexus-scratch/adas1236/geo_finetune/data/parquet/cyclic_order.parquet")
df = df.with_columns(
    roles = pl.lit({"center": 0, "b": 1, "c": 2})
)
print(df)
df.write_parquet("/fs/nexus-scratch/adas1236/geo_finetune/data/parquet/cyclic_order_new.parquet")