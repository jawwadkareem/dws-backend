import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_car_data(json_file):
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    cars = []
    descriptions = []

    for info in data:
        car = {}
        if 'make' in info and info['make']:
            car["brand"] = info['make']
        if 'model' in info and info['model']:
            car["model"] = info['model']
        if 'buildDate' in info and info['buildDate']:
            # Extract year from buildDate (e.g., "07/15" -> "2015")
            car["year"] = "20" + info['buildDate'].split('/')[-1]
        if 'fuelType' in info and info['fuelType']:
            # Map fuelType (e.g., "F" to "Petrol")
            fuel_map = {"F": "Petrol", "D": "Diesel", "E": "Electric", "H": "Hybrid"}
            car["fuel"] = fuel_map.get(info['fuelType'], info['fuelType'])
        if 'odometer' in info and info['odometer']:
            car["mileage"] = f"{info['odometer']} km"
        if 'price' in info and info['price']:
            car["price"] = f"${info['price']:,}"
        if 'transmission' in info and info['transmission']:
            car["transmission"] = info['transmission']
        if 'driveType' in info and info['driveType']:
            # Map driveType (e.g., "P" to "FWD")
            drive_map = {"P": "FWD", "R": "RWD", "A": "AWD"}
            car["drive_type"] = drive_map.get(info['driveType'], info['driveType'])
        if 'cyls' in info and info['cyls']:
            car["cylinders"] = info['cyls']
        if 'seats' in info and info['seats']:
            car["seats"] = info['seats']
        if 'stockNumber' in info and info['stockNumber']:
            car["stock_no"] = info['stockNumber']
        if '_id' in info and info['_id'].get('$oid'):
            car["url"] = f"https://direct-wholesale-cars.vercel.app/car_Details?id={info['_id']['$oid']}"

        # Compose description for embedding
        description_parts = []
        if 'brand' in car:
            description_parts.append(f"Brand: {car['brand']}")
        if 'model' in car:
            description_parts.append(f"Model: {car['model']}")
        if 'year' in car:
            description_parts.append(f"Year: {car['year']}")
        if 'fuel' in car:
            description_parts.append(f"Fuel: {car['fuel']}")
        if 'mileage' in car:
            description_parts.append(f"Mileage: {car['mileage']}")
        if 'price' in car:
            description_parts.append(f"Price: {car['price']}")
        if 'transmission' in car:
            description_parts.append(f"Transmission: {car['transmission']}")
        if 'drive_type' in car:
            description_parts.append(f"Drive Type: {car['drive_type']}")
        if 'cylinders' in car:
            description_parts.append(f"Cylinders: {car['cylinders']}")
        if 'seats' in car:
            description_parts.append(f"Seats: {car['seats']}")

        descriptions.append(", ".join(description_parts))
        cars.append(car)

    return cars, descriptions

def get_embedding(text):
    return model.encode(text)