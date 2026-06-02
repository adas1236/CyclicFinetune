import osmnx as ox
import polars as pl
import random
import itertools
import json

def get_coords(place_name):
    gdf = ox.geocode_to_gdf(place_name)
    centroid = gdf.geometry.centroid.iloc[0]
    return (centroid.y, centroid.x)  # (lat, lon)

def cyclic_order(coords_list):
    area = 0.0
    for i in range(len(coords_list)):
        x1, y1 = coords_list[i][1], coords_list[i][0]
        x2, y2 = coords_list[(i + 1) % len(coords_list)][1], coords_list[(i + 1) % len(coords_list)][0]
        area += x1 * y2 - x2 * y1
    is_ccw = area > 0

    # Flip result for Southern Hemisphere 
    if all(lat < 0 for lat, lon in coords_list):
        is_ccw = not is_ccw

    return "clockwise" if is_ccw else "counterclockwise"

# 20 countries, 10 cities each
countries_cities = {
    "USA": [
        "New York City, New York, USA",
        "Los Angeles, California, USA",
        "Chicago, Illinois, USA",
        "Houston, Texas, USA",
        "Phoenix, Arizona, USA",
        "Philadelphia, Pennsylvania, USA",
        "San Antonio, Texas, USA",
        "San Diego, California, USA",
        "Dallas, Texas, USA",
        "San Jose, California, USA",
    ],
    "United Kingdom": [
        "London, England, UK",
        "Birmingham, England, UK",
        "Manchester, England, UK",
        "Leeds, England, UK",
        "Glasgow, Scotland, UK",
        "Sheffield, England, UK",
        "Bradford, England, UK",
        "Liverpool, England, UK",
        "Edinburgh, Scotland, UK",
        "Bristol, England, UK",
    ],
    "Australia": [
        "Sydney, New South Wales, Australia",
        "Melbourne, Victoria, Australia",
        "Brisbane, Queensland, Australia",
        "Perth, Western Australia, Australia",
        "Adelaide, South Australia, Australia",
        "Gold Coast, Queensland, Australia",
        "Newcastle, New South Wales, Australia",
        "Wollongong, New South Wales, Australia",
        "Canberra, Australian Capital Territory, Australia",
        "Hobart, Tasmania, Australia",
    ],
    "Canada": [
        "Toronto, Ontario, Canada",
        "Montreal, Quebec, Canada",
        "Vancouver, British Columbia, Canada",
        "Calgary, Alberta, Canada",
        "Edmonton, Alberta, Canada",
        "Ottawa, Ontario, Canada",
        "Winnipeg, Manitoba, Canada",
        "Quebec City, Quebec, Canada",
        "Hamilton, Ontario, Canada",
        "Kitchener, Ontario, Canada",
    ],
    "India": [
        "Mumbai, Maharashtra, India",
        "Delhi, India",
        "Bengaluru, Karnataka, India",
        "Hyderabad, Telangana, India",
        "Ahmedabad, Gujarat, India",
        "Chennai, Tamil Nadu, India",
        "Kolkata, West Bengal, India",
        "Surat, Gujarat, India",
        "Pune, Maharashtra, India",
        "Jaipur, Rajasthan, India",
    ],
    "China": [
        "Beijing, China",
        "Shanghai, China",
        "Guangzhou, China",
        "Shenzhen, China",
        "Chengdu, China",
        "Chongqing, China",
        "Tianjin, China",
        "Nanjing, China",
        "Wuhan, China",
        "Xi'an, China",
    ],
    "Japan": [
        "Tokyo, Japan",
        "Yokohama, Japan",
        "Osaka, Japan",
        "Nagoya, Japan",
        "Sapporo, Japan",
        "Kobe, Japan",
        "Kyoto, Japan",
        "Fukuoka, Japan",
        "Kawasaki, Japan",
        "Saitama, Japan",
    ],
    "Germany": [
        "Berlin, Germany",
        "Hamburg, Germany",
        "Munich, Germany",
        "Cologne, Germany",
        "Frankfurt, Germany",
        "Stuttgart, Germany",
        "Düsseldorf, Germany",
        "Dortmund, Germany",
        "Essen, Germany",
        "Leipzig, Germany",
    ],
    "France": [
        "Paris, France",
        "Marseille, France",
        "Lyon, France",
        "Toulouse, France",
        "Nice, France",
        "Nantes, France",
        "Strasbourg, France",
        "Montpellier, France",
        "Bordeaux, France",
        "Lille, France",
    ],
    "Brazil": [
        "São Paulo, Brazil",
        "Rio de Janeiro, Brazil",
        "Brasília, Brazil",
        "Salvador, Brazil",
        "Fortaleza, Brazil",
        "Belo Horizonte, Brazil",
        "Manaus, Brazil",
        "Curitiba, Brazil",
        "Recife, Brazil",
        "Porto Alegre, Brazil",
    ],
    "Russia": [
        "Moscow, Russia",
        "Saint Petersburg, Russia",
        "Novosibirsk, Russia",
        "Yekaterinburg, Russia",
        "Nizhny Novgorod, Russia",
        "Kazan, Russia",
        "Chelyabinsk, Russia",
        "Omsk, Russia",
        "Samara, Russia",
        "Rostov-on-Don, Russia",
    ],
    "South Africa": [
        "Johannesburg, South Africa",
        "Cape Town, South Africa",
        "Durban, South Africa",
        "Pretoria, South Africa",
        "Port Elizabeth, South Africa",
        "Bloemfontein, South Africa",
        "Nelspruit, South Africa",
        "Pietermaritzburg, South Africa",
        "Kimberley, South Africa",
        "Polokwane, South Africa",
    ],
    "Nigeria": [
        "Lagos, Nigeria",
        "Kano, Nigeria",
        "Ibadan, Nigeria",
        "Abuja, Nigeria",
        "Benin City, Nigeria",
        "Port Harcourt, Nigeria",
        "Ilorin, Nigeria",
        "Maiduguri, Nigeria",
        "Zaria, Nigeria",
        "Abeokuta, Nigeria",
    ],
    "Mexico": [
        "Mexico City, Mexico",
        "Guadalajara, Mexico",
        "Monterrey, Mexico",
        "Puebla, Mexico",
        "Tijuana, Mexico",
        "León, Mexico",
        "Juárez, Mexico",
        "Torreón, Mexico",
        "San Luis Potosí, Mexico",
        "Querétaro, Mexico",
    ],
    "Argentina": [
        "Buenos Aires, Argentina",
        "Córdoba, Argentina",
        "Rosario, Argentina",
        "Mendoza, Argentina",
        "La Plata, Argentina",
        "San Miguel de Tucumán, Argentina",
        "Mar del Plata, Argentina",
        "Salta, Argentina",
        "Santa Fe, Argentina",
        "San Juan, Argentina",
    ],
    "Spain": [
        "Madrid, Spain",
        "Barcelona, Spain",
        "Valencia, Spain",
        "Seville, Spain",
        "Zaragoza, Spain",
        "Málaga, Spain",
        "Murcia, Spain",
        "Palma, Spain",
        "Las Palmas, Spain",
        "Bilbao, Spain",
    ],
    "Italy": [
        "Rome, Italy",
        "Milan, Italy",
        "Naples, Italy",
        "Turin, Italy",
        "Palermo, Italy",
        "Genoa, Italy",
        "Bologna, Italy",
        "Florence, Italy",
        "Bari, Italy",
        "Catania, Italy",
    ],
    "Indonesia": [
        "Jakarta, Indonesia",
        "Surabaya, Indonesia",
        "Bandung, Indonesia",
        "Medan, Indonesia",
        "Semarang, Indonesia",
        "Palembang, Indonesia",
        "Makassar, Indonesia",
        "Batam, Indonesia",
        "Pekanbaru, Indonesia",
        "Bogor, Indonesia",
    ],
    "Turkey": [
        "Istanbul, Turkey",
        "Ankara, Turkey",
        "Izmir, Turkey",
        "Bursa, Turkey",
        "Antalya, Turkey",
        "Adana, Turkey",
        "Gaziantep, Turkey",
        "Konya, Turkey",
        "Kayseri, Turkey",
        "Mersin, Turkey",
    ],
}

with open('question_formats.json', 'r') as f:
    question_formats = json.load(f)

# templates use 3 through 10 city names
templates_by_size = {size: question_formats[str(size)] for size in range(3, 11)}

def generate_dataset(countries_map, save_name):
    data_rows = []
    qid = 0

    # For each country, create questions where all locations come from the same country
    for country, cities in countries_map.items():
        max_size = min(len(cities), 10)
        for size in range(3, max_size + 1):
            templates = templates_by_size[size]
            for combo in itertools.combinations(cities, size):
                city_shorts = [city.split(",")[0] for city in combo]

                try:
                    coords_list = [get_coords(city) for city in combo]
                except Exception:
                    # skip combinations where geocoding fails for any city
                    continue

                answer = cyclic_order(coords_list)

                template_str, indices = random.choice(templates)
                question = template_str.format(*[city_shorts[i] for i in indices])

                data_rows.append({
                    "question_id": qid,
                    "question": question,
                    "location_names": list(combo),
                    "geometries": [
                        {"type": "point", "coordinates": list(coords)} for coords in coords_list
                    ],
                    "answer": answer,
                    "region": country,
                })

                qid += 1

    # setup the data frame and save it
    df = pl.DataFrame(data_rows)
    select_cols = ["question_id", "question", "location_names", "geometries", "answer", "region"]
    existing = [c for c in select_cols if c in df.columns]
    df = df.select(existing)

    df.write_parquet(save_name)


generate_dataset(countries_cities, "global_cyclic_order.parquet")