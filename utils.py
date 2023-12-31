import openai
import numpy as np
import pandas as pd

from config import MAX_SECTION_LEN, QUERY_EMBEDDINGS_MODEL, SEPARATOR, SEPARATOR_LEN


def get_embedding(text: str, model: str):
    result = openai.Embedding.create(model=model, input=text)
    return result["data"][0]["embedding"]


def get_query_embedding(text: str):
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)


def vector_similarity(x, y):
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference.
    """
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(query: str, contexts):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_query_embedding(query)

    document_similarities = sorted(
        [
            (vector_similarity(query_embedding, doc_embedding), doc_index)
            for doc_index, doc_embedding in contexts.items()
        ],
        reverse=True,
    )

    return document_similarities


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(
        question, context_embeddings
    )

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        document_section = df.loc[section_index]

        chosen_sections_len += len(document_section.text) + SEPARATOR_LEN
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.text.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))

    chosen_sections = [sections.to_list()[0] for sections in chosen_sections]

    # header = f"""Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""

    # return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"
    return "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"


def load_embeddings(fname: str):
    """
    Read the document embeddings and their keys from a CSV.

    fname is the path to a CSV with exactly these named columns:
        "source", "timestamp", "0", "1", ... up to the length of the embedding vectors.
    """

    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "source" and c != "timestamp"])
    return {
        (r.source, r.timestamp): [r[str(i)] for i in range(max_dim + 1)]
        for _, r in df.iterrows()
    }


# Reference: https://huggingface.co/spaces/elevenlabs/tts/blob/main/app.py
def pad_buffer(audio):
    # Pad buffer to multiple of 2 bytes
    buffer_size = len(audio)
    element_size = np.dtype(np.int16).itemsize
    if buffer_size % element_size != 0:
        audio = audio + b"\0" * (element_size - (buffer_size % element_size))
    return audio
