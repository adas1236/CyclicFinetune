import polars as pl
import pprint

df = pl.read_parquet("/fs/nexus-scratch/adas1236/geo_finetune/data/parquet/earth_cyclic_order.parquet")
# Calculate lengths, count unique occurrences, and sort by length
length_counts = (
    df["location_names"]
    .list.len()
    .alias("list_length")
    .value_counts()
    .sort("list_length")
)

print(length_counts)

cols = df[0].to_dicts()[0]

pprint.pprint(cols)

df = pl.read_parquet("/fs/nexus-scratch/adas1236/geo_finetune/data/parquet/spatial_questions_train.parquet")
cols = df[0].to_dicts()[0]

pprint.pprint(cols)