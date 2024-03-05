import os

import numpy as np
import spacy
from langchain.docstore.document import Document
from langchain_community.document_loaders import PDFMinerLoader
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-ada-002"  # type: ignore
)

nlp = spacy.load("en_core_web_sm")


def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]["combined_sentence_embedding"]
        embedding_next = sentences[i + 1]["combined_sentence_embedding"]

        # Calculate cosine similarit
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]  # type: ignore

        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        sentences[i]["distance_to_next"] = distance

    # Optionally handle the last sentence
    # sentences[-1]['distance_to_next'] = None  # or a default value

    return distances, sentences


def combine_sentences(sentences, buffer_size=1):
    # Go through each sentence dict
    for i in range(len(sentences)):

        # Create a string that will hold the sentences which are joined
        combined_sentence = ""

        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative (to avoid index out of range like on the first one)
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]["sentence"] + " "

        # Add the current sentence
        combined_sentence += sentences[i]["sentence"]

        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += " " + sentences[j]["sentence"]

        # Then add the whole thing to your dict
        # Store the combined sentence in the current sentence dict
        sentences[i]["combined_sentence"] = combined_sentence

    return sentences


def create_chunks(indices_above_thresh):
    # Initialize the start index
    start_index = 0

    # Create a list to hold the grouped sentences
    chunks = []

    # Iterate through the breakpoints to slice the sentences
    for index in indices_above_thresh:
        # The end index is the current breakpoint
        end_index = index

        # Slice the sentence_dicts from the current start index to the end index
        group = sentences[start_index : end_index + 1]  # type: ignore
        combined_text = " ".join([d["sentence"] for d in group])
        chunks.append(combined_text)

        # Update the start index for the next group
        start_index = index + 1

    # The last group, if any sentences remain
    if start_index < len(sentences):  # type: ignore
        combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])  # type: ignore
        chunks.append(combined_text)

    return chunks


def split_text(text, max_length=49000):
    """
    Splits text into chunks that are less than max_length bytes.
    This example uses a simple approach and may split words. Adjust accordingly.
    """
    # Convert to bytes to ensure we're splitting based on actual byte size
    text_bytes = text.encode("utf-8")
    chunks = []
    output = []

    while len(text_bytes) > max_length:
        # Find the last byte within the limit (simple split, may need refinement)
        split_index = text_bytes.rfind(b" ", 0, max_length)

        # If we can't find a space to split on, force split at max_length
        if split_index == -1:
            split_index = max_length

        # Extract the chunk and add it to the list
        chunk = text_bytes[:split_index].decode("utf-8")
        chunks.append(chunk)

        # Remove the chunk from the original text
        text_bytes = text_bytes[split_index:].lstrip()  # Remove leading spaces

    # Add the remaining text if any
    if text_bytes:
        chunks.append(text_bytes.decode("utf-8"))

    for chunk in chunks:
        doc = nlp(chunk)

        output += doc.sents

    return output


def chunk_file(paths):
    print("NOOO CHUNKING AGAINN!!!!!!!!!!!!!")
    single_sentences_list = []
    for path in paths:
        try:
            loader = PDFMinerLoader(path)
            whole_document = loader.load()
            whole_document = whole_document[0].page_content
            each_file_sentences = split_text(whole_document)
            single_sentences_list += each_file_sentences
        except:
            continue
    single_sentences_list = [str(x) for x in single_sentences_list]
    sentences = [
        {"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)
    ]
    sentences = combine_sentences(sentences)
    embeddings = embedding_model.embed_documents(
        [x["combined_sentence"] for x in sentences]
    )
    for i, sentence in enumerate(sentences):
        sentence["combined_sentence_embedding"] = embeddings[i]
    distances, sentences = calculate_cosine_distances(sentences)
    breakpoint_percentile_threshold = 75
    breakpoint_distance_threshold = np.percentile(
        distances, breakpoint_percentile_threshold
    )
    num_distances_above_theshold = len(
        [x for x in distances if x > breakpoint_distance_threshold]
    )
    indices_above_thresh = [
        i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
    ]
    # Initialize the start index
    start_index = 0

    # Create a list to hold the grouped sentences
    chunks = []

    # Iterate through the breakpoints to slice the sentences
    for index in indices_above_thresh:
        # The end index is the current breakpoint
        end_index = index

        # Slice the sentence_dicts from the current start index to the end index
        group = sentences[start_index : end_index + 1]
        combined_text = " ".join([d["sentence"] for d in group])
        chunks.append(combined_text)

        # Update the start index for the next group
        start_index = index + 1

    # The last group, if any sentences remain
    if start_index < len(sentences):
        combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
        chunks.append(combined_text)
    documents = []
    for i, item in enumerate(chunks):
        chunk = Document(page_content=item, metadata={"index": i})
        documents.append(chunk)
    return documents
