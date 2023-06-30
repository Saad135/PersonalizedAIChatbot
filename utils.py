import openai
import os
import copy
import numpy as np
import pandas as pd

import googleapiclient.discovery

from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

from config import (
    DOC_EMBEDDINGS_MODEL,
    MAX_SECTION_LEN,
    QUERY_EMBEDDINGS_MODEL,
    SEPARATOR,
    SEPARATOR_LEN,
)

load_dotenv()

youtube = googleapiclient.discovery.build(
    "youtube",
    "v3",
    developerKey=os.getenv("yt_api_key"),
)


def get_embedding(text: str, model: str):
    result = openai.Embedding.create(model=model, input=text)
    return result["data"][0]["embedding"]


def get_query_embedding(text: str):
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)


def get_doc_embedding(text: str):
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)


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
            (vector_similarity(query_embedding, context["embedding"]), context["text"])
            for context in contexts
        ],
        reverse=True,
    )

    return document_similarities


def construct_prompt(question, context_embeddings) -> str:
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(
        question, context_embeddings
    )

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_text in most_relevant_document_sections:
        # Add contexts until we run out of space.

        chosen_sections_len += len(section_text) + SEPARATOR_LEN
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + section_text)
        # chosen_sections_indexes.append(str(section_index))

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


def get_channel_videos(playlist_id, num_vids):
    video_ids = []
    page_token = None

    while True:
        request = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,  # Fetch 50 videos at a time
            pageToken=page_token,  # Add pagination
        )
        response = request.execute()

        num_vids = (
            num_vids
            if num_vids != None and num_vids < len(response["items"])
            else len(response["items"])
        )

        video_ids += [
            item["contentDetails"]["videoId"]
            for item in response["items"][0 : int(num_vids)]
            if item["kind"] == "youtube#playlistItem"
        ]

        # Check if there are more videos to fetch
        if "nextPageToken" in response:
            page_token = response["nextPageToken"]
        else:
            break

    return video_ids


def get_transcripts(video_ids, progress):
    transcripts = {}
    for video_id in progress.tqdm(video_ids, desc="Downloading transcripts"):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            transcripts[video_id] = transcript
        except Exception as ex:
            print(f"An error occurred for video: {video_id} [{ex}]")
    return transcripts


def merge_transcripts(transcripts, progress):
    def reset_merged_item():
        return {"text": "", "start": None, "duration": 0.00}

    merged_item = reset_merged_item()
    merged_transcript = []
    merged_transcript_wo_embed = []
    # all_transcripts = []

    # embeddings = {}

    for key in progress.tqdm(
        transcripts.keys(), desc="Generating embeddings for every video"
    ):
        for item in progress.tqdm(
            transcripts[key], desc="Generating embedding for every chunk"
        ):
            merged_item["source"] = (key, item["start"])
            merged_item["text"] += item["text"].replace("\n", " ")
            merged_item["start"] = item["start"]
            merged_item["duration"] += item["duration"]

            if merged_item["duration"] > 30.00:
                merged_transcript_wo_embed += [copy.copy(merged_item)]
                merged_item["embedding"] = get_doc_embedding(merged_item["text"])
                merged_transcript += [merged_item]
                merged_item = reset_merged_item()

        # all_transcripts += [merged_transcript]
        # merged_transcript = []

    return merged_transcript, merged_transcript_wo_embed
