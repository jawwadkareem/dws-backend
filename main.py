import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from data_loader import load_car_data, get_embedding

# Initialize OpenAI Grok client
client = OpenAI(
    api_key="xai-y1wct4shqtO1QC0H2kXWZsHLbbqMHOSWigNC2Bb440OYfoTGhlj8Yiitd1ek6qqfVsF5ylAhfM4Xer8A",
    base_url="https://api.x.ai/v1"
)

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load car data and index
car_data, _ = load_car_data("directwholesalecars.carads.json")
index = faiss.read_index("asian_cars_faiss_index.idx")

# Search function
def search_similar_cars(query, top_k=50):
    query_embedding = np.array([get_embedding(query)], dtype=np.float32)
    distances, indices = index.search(query_embedding, top_k)
    return [car_data[i] for i in indices[0]]

# Chat memory
conversation_history = []

def ask_grok(question, cars):
    global conversation_history

    formatted_cars = []
    for car in cars:
        parts = []
        if 'brand' in car:
            parts.append(f"Brand: {car['brand']}")
        if 'model' in car:
            parts.append(f"Model: {car['model']}")
        if 'year' in car:
            parts.append(f"Year: {car['year']}")
        if 'fuel' in car:
            parts.append(f"Fuel: {car['fuel']}")
        if 'mileage' in car:
            parts.append(f"Mileage: {car['mileage']}")
        if 'price' in car:
            parts.append(f"Price: {car['price']}")
        if 'transmission' in car:
            parts.append(f"Transmission: {car['transmission']}")
        if 'drive_type' in car:
            parts.append(f"Drive Type: {car['drive_type']}")
        if 'cylinders' in car:
            parts.append(f"Cylinders: {car['cylinders']}")
        if 'seats' in car:
            parts.append(f"Seats: {car['seats']}")
        if 'stock_no' in car and 'price' in car and 'url' in car:
            parts.append(f"👉 [{car['brand']} {car['model']} - {car['price']}] - {car['url']}")
        formatted_cars.append(", ".join(parts))

    formatted_text = "\n\n".join(formatted_cars)

    conversation_history.append({"role": "user", "content": question})
    conversation_history = conversation_history[-5:]

    system_prompt = {
        "role": "system",
        "content": """
You're a helpful, human-sounding car expert. Only recommend cars if the user requests suggestions.
If they say "hi" or greet you, return a friendly reply and ask how you can help — don’t suggest anything unsolicited.

Keep answers brief, natural, and smart. When cars are mentioned, include a link to their page:
👉 [2022 ABARTH 595 - $18,750] - https://direct-wholesale-cars.vercel.app/car_Details?id=689b125f997c2c5c0663eb9d
⚠️ Do NOT use Markdown-style links like [title](url). Just use the format above — square brackets, then a hyphen, then the plain link.
Never give generic answers. If confused, ask clarifying questions.
"""
    }

    conversation_history.append({
        "role": "user",
        "content": f"""
Customer Question:
{question}

Top Matches from Inventory:
{formatted_text}

Instructions:
- Recommend only if the customer is asking about car suggestions or comparisons.
- Be short (2–3 lines), helpful, and conversational — like a friendly car expert.
- If appropriate, include direct links to the cars if the user is interested in purchasing.
- Don’t act like a salesperson; focus on fit and clarity.
- If unsure, ask for more info.
"""
    })

    response = client.chat.completions.create(
        model="grok-2-latest",
        messages=[system_prompt] + conversation_history
    )

    reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": reply})
    conversation_history = conversation_history[-5:]
    return reply

# API input model
class QueryRequest(BaseModel):
    query: str

# API endpoints
@app.post("/query")
async def query_cars(request: QueryRequest):
    results = search_similar_cars(request.query)
    answer = ask_grok(request.query, results)
    return {"message": answer}

@app.get("/")
async def welcome():
    return {"message": "WELCOME TO ASIAN CARS CHATBOT"}

# Run app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)