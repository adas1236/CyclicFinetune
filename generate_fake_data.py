import numpy as np
import polars as pl
import random
import json
import sys

def generate_simple_polygon(n, max_coord: int):
    points = np.random.uniform(-max_coord, max_coord, size = (n, 2))
    
    centroid = np.mean(points, axis=0)

    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    
    return np.round(sorted_points, 4)

def compute_pairwise(center, src, dest):
    c_pt = representative_point(center)
    s_pt = representative_point(src)
    d_pt = representative_point(dest)
    
    det = (s_pt[0] - c_pt[0]) * (d_pt[1] - c_pt[1]) - (d_pt[0] - c_pt[0]) * (s_pt[1] - c_pt[1])

    return -1 if det > 0 else 1

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

def generate_random(num_locations: int, max_coord: int):
    geometries = []

    for _ in range(num_locations):
        geometry_type = random.choice(['point', 'line', 'polygon'])

        if geometry_type == 'point':
            n = 1
        
        if geometry_type == 'line':
            n = 2
        
        if geometry_type == 'polygon':
            n = random.randint(3, 10)
        
        coords = generate_simple_polygon(n, max_coord).tolist()
        geometries.append(
            {"coordinates": coords, "type": geometry_type}
        )
    
    return geometries

def generate_ccw(num_locations: int, max_coord: int):
    angles_good = False
    while not angles_good:
        angles_raw = np.random.sample(size = num_locations - 1)
        sorted_angles = sorted(angles_raw, reverse=False)
        angles_good = max(np.diff(sorted_angles)) < 0.5
    angles = np.array(sorted_angles) * 2 * np.pi
    
    radii = np.random.uniform(0, max_coord, size = (num_locations - 1))
    center_point = generate_simple_polygon(1, max_coord)

    points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles))) + center_point
    points = points.tolist()

    geometries = [
        {"coordinates": center_point.tolist(), "type": "point"}
    ]
    
    for point in points:
        geometries.append({"coordinates": [point], "type": "point"})
    
    return geometries

def generate_cw(num_locations: int, max_coord: int):
    angles_good = False
    while not angles_good:
        angles_raw = np.random.sample(size = num_locations - 1)
        sorted_angles = sorted(angles_raw, reverse=True)
        angles_good = min(np.diff(sorted_angles)) > -0.5
    angles = np.array(sorted_angles) * 2 * np.pi
    
    radii = np.random.uniform(0, max_coord, size = (num_locations - 1))
    center_point = generate_simple_polygon(1, max_coord)

    points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles))) + center_point
    points = points.tolist()

    geometries = [
        {"coordinates": center_point.tolist(), "type": "point"}
    ]
    
    for point in points:
        geometries.append({"coordinates": [point], "type": "point"})
    
    return geometries

def generate_neither(num_locations: int, max_coord: int):
    while True:
        geometries = generate_random(num_locations, max_coord)
        pairwise_res = [
            compute_pairwise(geometries[0], geometries[i], geometries[i + 1]) for i in range(1, num_locations - 1)
        ]
        if not all(x == -1 for x in pairwise_res) and not all(x == 1 for x in pairwise_res):
            return geometries

def generate_balanced(save_name, num_train_per_n, max_n, name_list, max_coord, question_formats):
    data_rows = []
    generated_count = 0

    for num_locations in range(3, max_n + 1):
        order_types = 3
        if num_locations == 3:
            order_types = 2
        for order_type in range(order_types):
            for _ in range(num_train_per_n // order_types):
                chosen_names = random.sample(name_list, num_locations)
                
                form, order = random.choice(question_formats[str(num_locations)])
                question = form.format(
                    *[chosen_names[elem] for elem in order]
                )

                if order_type == 0:
                    geometries = generate_cw(num_locations, max_coord)
                    res = 'clockwise'
                
                if order_type == 1:
                    geometries = generate_ccw(num_locations, max_coord)
                    res = 'counterclockwise'
                
                if order_type == 2:
                    geometries = generate_neither(num_locations, max_coord)
                    res = 'neither'

                data_rows.append({
                    "question_id": generated_count,
                    "question": question,
                    "location_names": chosen_names,
                    "geometries": geometries,
                    "answer": res,
                })

                generated_count += 1

    df = pl.DataFrame(data_rows)
    df.write_parquet(save_name)
    print(f"Dataset Created at {save_name} (sample below):")
    print(df.head(5))

def generate_natural(save_name, num_train_per_n, max_n, name_list, max_coord, question_formats):
    data_rows = []
    generated_count = 0

    for num_locations in range(3, max_n + 1):
        for _ in range(num_train_per_n):
            chosen_names = random.sample(name_list, num_locations)
            
            form, order = random.choice(question_formats[str(num_locations)])
            question = form.format(
                *[chosen_names[elem] for elem in order]
            )

            geometries = generate_random(num_locations, max_coord)
            
            pairwise_res = [compute_pairwise(geometries[0], geometries[i + 1], geometries[i + 2]) for i in range(num_locations - 2)]
            res = 'neither'

            if all(i == -1 for i in pairwise_res):
                res = 'counterclockwise'
            
            if all(i == 1 for i in pairwise_res):
                res = 'clockwise'

            data_rows.append({
                "question_id": generated_count,
                "question": question,
                "location_names": chosen_names,
                "geometries": geometries,
                "answer": res,
            })

            generated_count += 1

    df = pl.DataFrame(data_rows)
    df.write_parquet(save_name)
    print(f"Dataset Created at {save_name} (sample below):")
    print(df.head(5))

def main():
    url = "https://raw.githubusercontent.com/datasets/world-cities/master/data/world-cities.csv"

    df_names = pl.read_csv(url)
    name_list = df_names.select("name").sample(8000).to_series().to_list()

    print(f"Loaded {len(name_list)} names. Example: {name_list[:5]}")

    max_coord = 1000
    num_train_per_n = 4000
    num_val_natural_per_n = 1000
    num_val_balanced_per_n = 1000

    max_n = 10

    question_formats = {}

    with open ("question_formats.json") as f:
        question_formats = json.load(f)

    generate_balanced(
        save_name='data/parquet/spatial_questions_train.parquet',
        num_train_per_n=num_train_per_n,
        max_n=max_n,
        name_list=name_list,
        max_coord=max_coord,
        question_formats=question_formats
    )

    generate_balanced(
        save_name='data/parquet/spatial_questions_val_balanced.parquet',
        num_train_per_n=num_val_balanced_per_n,
        max_n=max_n,
        name_list=name_list,
        max_coord=max_coord,
        question_formats=question_formats
    )

    generate_natural(
        save_name='data/parquet/spatial_questions_val_natural.parquet',
        num_train_per_n=num_val_natural_per_n,
        max_n=max_n,
        name_list=name_list,
        max_coord=max_coord,
        question_formats=question_formats
    )

if __name__ == '__main__':
    main()