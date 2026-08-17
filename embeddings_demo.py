from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


if __name__ == "__main__":
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]
    embeddings = model.encode(sentences)
    print(embeddings.shape)
    # => (3, 384)

    male = model.encode("male")
    female = model.encode("female")
    apple = model.encode("apple")
    banana = model.encode("banana")
    valhalla = model.encode("valhalla")
    diff = male - female

    print(model.similarity(male, female))
    print(model.similarity(male, diff))
    print(model.similarity(female, diff))
    print(model.similarity(male, apple))
    print(model.similarity(male, valhalla))
    print(model.similarity(apple, banana))
    print(model.similarity(apple, valhalla))
    print(model.similarity(model.encode("ooples"), model.encode("banoonoos")))