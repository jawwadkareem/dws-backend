from data_loader import load_car_data, get_embedding
import faiss
import numpy as np

# Load car data
car_data, car_descriptions = load_car_data("directwholesalecars.carads.json")

# Generate embeddings
embeddings = np.array([get_embedding(desc) for desc in car_descriptions], dtype=np.float32)
print("Embeddings shape:", embeddings.shape)

# Create and populate FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save the index
faiss.write_index(index, "asian_cars_faiss_index.idx")
print("✅ FAISS index generated and saved as asian_cars_faiss_index.idx")