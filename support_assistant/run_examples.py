from graph import ask


examples = [
    "What is the delivery time for my order?",
    "What is the capital of India?",
]

for query in examples:
    response = ask(query)
    print("\nQUERY:", query)
    print("RAW JSON:", response.model_dump_json())
