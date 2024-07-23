import torch
from transformers import AutoTokenizer, AutoModel
from diffusers import StableDiffusionPipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# 1. Data Preparation
def load_cs_concepts():
    return [
        (
            "Binary Search",
            "An efficient algorithm for finding an item in a sorted list by repeatedly dividing the search interval in half.",
        ),
        (
            "Big O Notation",
            "A mathematical notation that describes the limiting behavior of a function when the argument tends towards a particular value or infinity.",
        ),
        (
            "Linked List",
            "A linear data structure where elements are stored in nodes, and each node points to the next node in the sequence.",
        ),
        (
            "Stack",
            "A last-in, first-out (LIFO) data structure that supports two main operations: push (add an element) and pop (remove the most recently added element).",
        ),
        (
            "Queue",
            "A first-in, first-out (FIFO) data structure that supports two main operations: enqueue (add an element) and dequeue (remove the oldest element).",
        ),
        (
            "Recursion",
            "A method of solving problems where the solution depends on solutions to smaller instances of the same problem.",
        ),
        (
            "Hash Table",
            "A data structure that implements an associative array abstract data type, a structure that can map keys to values using a hash function.",
        ),
        (
            "Depth-First Search",
            "A graph traversal algorithm that explores as far as possible along each branch before backtracking.",
        ),
        (
            "Breadth-First Search",
            "A graph traversal algorithm that explores all the vertices of a graph at the present depth prior to moving on to the vertices at the next depth level.",
        ),
        (
            "Object-Oriented Programming",
            "A programming paradigm based on the concept of 'objects', which can contain data and code. The key principles are encapsulation, inheritance, and polymorphism.",
        ),
        (
            "Sorting Algorithms",
            "Algorithms for arranging elements in a specific order, such as numerical or lexicographical. Common examples include Bubble Sort, Merge Sort, and Quick Sort.",
        ),
        (
            "Binary Tree",
            "A tree data structure in which each node has at most two children, referred to as the left child and the right child.",
        ),
        (
            "Dynamic Programming",
            "A method for solving complex problems by breaking them down into simpler subproblems and storing the results for future use.",
        ),
        (
            "API (Application Programming Interface)",
            "A set of definitions, protocols, and tools for building software. It specifies how software components should interact.",
        ),
        (
            "Multithreading",
            "A programming concept where multiple threads (lightweight processes) run concurrently within a single program, sharing the same resources.",
        ),
        (
            "Database Index",
            "A data structure that improves the speed of data retrieval operations on a database table at the cost of additional writes and storage space.",
        ),
        (
            "Regular Expression",
            "A sequence of characters that defines a search pattern, mainly for use in pattern matching with strings.",
        ),
        (
            "Version Control",
            "A system that records changes to a file or set of files over time so that you can recall specific versions later.",
        ),
        (
            "Cache",
            "A hardware or software component that stores data so that future requests for that data can be served faster.",
        ),
        (
            "Compiler",
            "A program that translates code written in a high-level programming language into low-level machine code.",
        ),
    ]


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
