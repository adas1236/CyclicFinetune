import numpy as np
import polars as pl
import random

url = "https://raw.githubusercontent.com/datasets/world-cities/master/data/world-cities.csv"

df_names = pl.read_csv(url)
NAME_LIST = df_names.select("name").sample(8000).to_series().to_list()

print(f"Loaded {len(NAME_LIST)} names. Example: {NAME_LIST[:5]}")


MAX_COORD = 1000

NUM_GENERATED = 15000

QUESTION_FORMATS = [
    ["From {}, in which rotational direction would you go from {} to {}", [0, 1, 2]],
    ["In which rotational direction would you go from {} to {} from {}", [1, 2, 0]],
    ["With respect to a centroid in {}, is moving from {} to {} clockwise or counterclockwise?", [0, 1, 2]],
    ["Is moving from {} to {} clockwise or counterclockwise with respect to a centroid in {}?", [1, 2, 0]],
    ["From {}, are {} and {} in a clockwise or counterclockwise order?", [0, 1, 2]],
    ["Are {} and {} in a clockwise or counterclockwise order from {}?", [1, 2, 0]],
    ["If you're standing in {}, are {} and {} arranged clockwise or counterclockwise around you?", [0, 1, 2]],
    ["Are {} and {} arranged clockwise or counterclockwise around you if you're standing in {}?", [1, 2, 0]],
    ["Centered at {}, does the path from {} to {} go clockwise or counterclockwise?", [0, 1, 2]],
    ["Does the path from {} to {} go clockwise or counterclockwise when centered at {}?", [1, 2, 0]],
    ["Relative to {}, is the rotation from {} to {} clockwise or counterclockwise?", [0, 1, 2]],
    ["Is the rotation from {} to {} clockwise or counterclockwise relative to {}?", [1, 2, 0]],
    ["Pivoting around {}, which direction (clockwise or counterclockwise) takes you from {} to {}?", [0, 1, 2]],
    ["Which direction (clockwise or counterclockwise) takes you from {} to {} when pivoting around {}?", [1, 2, 0]],
    ["Starting at {}, if you orbit {} to reach {}, did you travel clockwise or counterclockwise?", [1, 0, 2]],
    ["If {} is the origin, does a sweep from {} to {} go clockwise or counterclockwise?", [0, 1, 2]],
    ["Does a sweep from {} to {} go clockwise or counterclockwise if {} is the origin?", [1, 2, 0]],
    ["Taking {} as the focal point, is the arc from {} to {} drawn clockwise or counterclockwise?", [0, 1, 2]],
    ["Is the arc from {} to {} drawn clockwise or counterclockwise when taking {} as the focal point?", [1, 2, 0]],
    ["Considering {} as the central axis, do you rotate clockwise or counterclockwise to get from {} to {}?", [0, 1, 2]],
    ["Do you rotate clockwise or counterclockwise to get from {} to {} considering {} as the central axis?", [1, 2, 0]],
    ["Placing a clock face at {}, would a hand sweep clockwise or counterclockwise to go from {} to {}?", [0, 1, 2]],
    ["Would a hand sweep clockwise or counterclockwise to go from {} to {} if you placed a clock face at {}?", [1, 2, 0]],
    ["To travel from {} to {} around {}, do you take a clockwise or counterclockwise route?", [1, 2, 0]],
    ["Around {}, do you take a clockwise or counterclockwise route to travel from {} to {}?", [0, 1, 2]],
    ["Moving from {} to {}, is the angular displacement clockwise or counterclockwise with respect to {}?", [1, 2, 0]],
    ["With respect to {}, is the angular displacement clockwise or counterclockwise when moving from {} to {}?", [0, 1, 2]],
    ["Imagine looking from {}. Does a transition from {} to {} move clockwise or counterclockwise?", [0, 1, 2]]
]

def generate_simple_polygon(n):
    points = np.random.uniform(-MAX_COORD, MAX_COORD, size = (n, 2))
    
    centroid = np.mean(points, axis=0)

    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    
    return np.round(sorted_points, 4)

def representative_point(geometry: dict) -> tuple[float, float]:
    gtype = geometry["type"].lower()
    coords = geometry["coordinates"]

    if gtype == "point":
        return (coords[0][0], coords[0][1])
    elif gtype == "line":
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    elif gtype == "polygon":
        ring = coords
        if len(ring) > 1 and ring[0] == ring[-1]:
            ring = ring[:-1]
        xs = [c[0] for c in ring]
        ys = [c[1] for c in ring]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    else:
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

data_rows = []
generated_count = 0

for _ in range(NUM_GENERATED):
    chosen_names = [a, b, c] = random.sample(NAME_LIST, 3)

    geometries = []

    for _ in range(3):
        geometry_type = random.choice(['point', 'line', 'polygon'])

        if geometry_type == 'point':
            n = 1
        
        if geometry_type == 'line':
            n = 2
        
        if geometry_type == 'polygon':
            n = random.randint(3, 10)
        
        coords = generate_simple_polygon(n).tolist()
        geometries.append(
            {"coordinates": coords, "type": geometry_type}
        )
    
    form, order = random.choice(QUESTION_FORMATS)
    question = form.format(chosen_names[order[0]], chosen_names[order[1]], chosen_names[order[2]])

    a_point = representative_point(geometries[0])
    b_point = representative_point(geometries[1])
    c_point = representative_point(geometries[2])

    det = (b_point[0] - a_point[0]) * (c_point[1] - a_point[1]) - (c_point[0] - a_point[0]) * (b_point[1] - a_point[1])

    if det == 0:
        continue 

    data_rows.append({
        "question_id": generated_count,
        "question": question,
        "location_names": chosen_names,
        "geometries": geometries,
        "answer": "counterclockwise" if det > 0 else "clockwise",
        "roles": {"center": 0, "b": 1, "c": 2} # Order in the actual list, which is always [0, 1, 2]
    })

    generated_count += 1

df = pl.DataFrame(data_rows)
df.write_parquet("spatial_questions.parquet")
print("Dataset Created (sample below):")
print(df.head(5))