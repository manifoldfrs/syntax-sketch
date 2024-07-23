import torch
from transformers import AutoTokenizer, AutoModel
from diffusers import StableDiffusionPipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# 1. Data Preparation
def load_cs_concepts():
    # Load CS concepts and their explanations
    # Return a list of tuples (concept, explanation)
    pass


# 2. Text Embedding
def embed_text(text):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


# 3. Vector Database (simplified using a list)
vector_db = []


def add_to_db(concept, explanation, embedding):
    vector_db.append((concept, explanation, embedding))


# 4. Retrieval System
def retrieve_similar(query, k=3):
    query_embedding = embed_text(query)
    similarities = [
        cosine_similarity([query_embedding], [item[2]])[0][0] for item in vector_db
    ]
    top_k = sorted(
        range(len(similarities)), key=lambda i: similarities[i], reverse=True
    )[:k]
    return [vector_db[i] for i in top_k]


# 5. Image Generation
def generate_image(prompt):
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    image = pipe(prompt).images[0]
    return image


# 6. RAG Application
def explain_cs_concept(user_query):
    similar_concepts = retrieve_similar(user_query)

    # Combine retrieved concepts into a prompt
    combined_explanation = " ".join(
        [f"{concept}: {explanation}" for concept, explanation, _ in similar_concepts]
    )

    prompt = f"Create a clear, simple diagram explaining the computer science concept: {user_query}. "
    prompt += f"Use these related explanations for context: {combined_explanation}"

    # Generate image
    image = generate_image(prompt)

    return image, similar_concepts


# Main application flow
def main():
    # Load and process CS concepts
    cs_concepts = load_cs_concepts()

    # Embed and store concepts
    for concept, explanation in cs_concepts:
        embedding = embed_text(concept + " " + explanation)
        add_to_db(concept, explanation, embedding)

    # Example usage
    user_query = "Explain binary search algorithm"
    image, related_concepts = explain_cs_concept(user_query)

    # Display image (you'll need to implement this based on your environment)
    image.show()

    # Print related concepts
    for concept, explanation, _ in related_concepts:
        print(f"{concept}: {explanation}")


if __name__ == "__main__":
    main()
