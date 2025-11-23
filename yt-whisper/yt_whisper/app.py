import os
import re
import tempfile
from pathlib import Path
from tempfile import mkdtemp

import streamlit as st
import yt_dlp
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from streamlit.logger import get_logger

from yt_whisper.vtt_utils import merge_webvtt_to_list

load_dotenv(".env")

logger = get_logger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_TOKEN"))


@st.cache_resource
def load_pinecone(index_name="docker-genai"):
    # initialize pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_TOKEN"))

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(index_name)
    return index


@st.cache_data
def process_video(video_url: str) -> dict[str, str]:
    """Process the video and return the transcription"""

    # Extract video ID from URL
    video_id_match = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', video_url)
    if not video_id_match:
        st.error("Please enter a valid youtube url", icon="🚨")
        return None

    video_id = video_id_match.group(1)

    try:
        with st.spinner("Processing your video"):
            logger.info(f"Processing video: {video_url}")

            with tempfile.TemporaryDirectory() as tmp_dir:
                # Configure yt-dlp options - use ffmpeg for HLS as recommended
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(tmp_dir, 'audio.%(ext)s'),
                    'quiet': True,
                    'no_warnings': True,
                    'force_ipv4': True,
                    'external_downloader': 'ffmpeg',  # Use ffmpeg for HLS streams
                    'external_downloader_args': {
                        'ffmpeg': ['-nostdin', '-y']  # Non-interactive, overwrite
                    },
                    'extractor_args': 'youtube:player_client=mweb',  # Recommended by docs
                }

                # Download audio and get video info
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=True)
                    video_title = info.get('title', 'Unknown')
                    thumbnail_url = info.get('thumbnail', '')

                # Find the downloaded file (should match the pattern audio.*)
                import glob
                audio_files = glob.glob(os.path.join(tmp_dir, 'audio.*'))
                if not audio_files:
                    st.error("Failed to download audio file", icon="🚨")
                    logger.error(f"No audio files found in {tmp_dir}")
                    return None

                audio_file = audio_files[0]
                logger.info(f"Downloaded audio file: {audio_file}")
                file_stats = os.stat(audio_file)
                logger.info(f"File size(bytes): {file_stats.st_size}")
                logger.info(f"File name: {audio_file}")
                if file_stats.st_size > 24 * 1024 * 1024:  # 25 MB Limit check
                    # TODO(davidnet): Split and process the video in chunks
                    st.error(
                        "Please select a shorter video, OpenAI has a limit of 25 MB",
                        icon="🚨",
                    )
                    return None
                with open(audio_file, "rb") as audio_file:
                    whisper_transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="vtt",
                    )
                logger.info("Transcription done")

                seconds_to_merge = 8

                transcript = merge_webvtt_to_list(whisper_transcript, seconds_to_merge)

                stride = 3
                video_data = []

                def _upload_to_pinecone(video_data):
                    index = load_pinecone()
                    batch_transcripts = [t["text"] for t in video_data]
                    batch_ids = [t["id"] for t in video_data]
                    batch_metadata = [
                        {
                            "initial_time": t["initial_time"],
                            "title": video_title,
                            "thumbnail": thumbnail_url,
                            "video_url": f"{video_url}&t={t['initial_time']}s",
                            "text": t["text"],
                        }
                        for t in video_data
                    ]
                    embeddings = client.embeddings.create(
                        input=batch_transcripts, model="text-embedding-3-small"
                    )
                    batch_embeds = [e.embedding for e in embeddings.data]
                    to_upsert = list(zip(batch_ids, batch_embeds, batch_metadata))
                    index.upsert(to_upsert)

                for block in range(0, len(transcript), stride):
                    initial_time = transcript[block]["initial_time_in_seconds"]
                    id = f"{video_id}-t{initial_time}"
                    text = " ".join(
                        [t["text"] for t in transcript[block : block + stride]]
                    ).replace("\n", " ")
                    video_data.append(
                        {"initial_time": initial_time, "text": text, "id": id}
                    )
                    if len(video_data) > 64:
                        _upload_to_pinecone(video_data)
                        video_data = []

                if len(video_data) > 0:
                    _upload_to_pinecone(video_data)

                output_transcript_path: Path = (
                    st.session_state.tempfolder / f"{video_id}.txt"
                )
                with open(output_transcript_path, "w") as transcript_file:
                    transcript_file.write(whisper_transcript)

                    return {
                        "video_id": video_id,
                        "title": video_title,
                        "thumbnail": thumbnail_url,
                    }
    except yt_dlp.utils.DownloadError as e:
        st.error(f"Failed to download video. This could be due to regional restrictions, age restrictions, or the video being private/deleted.", icon="🚨")
        logger.error(f"Download error: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred while processing the video: {str(e)}", icon="🚨")
        logger.error(f"Error processing video: {e}")
        return None


def disable(b):
    st.session_state["processing"] = b


def main():
    logger.debug("Rendering app")
    if "tempfolder" not in st.session_state:
        st.session_state.tempfolder = Path(mkdtemp(prefix="yt_transcription_"))
    if "videos" not in st.session_state:
        st.session_state.videos = []
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # "state: ", st.session_state

    st.header("Chat with karpathy's videos")
    st.write(
        "upload karpathy's videos and get a transcription of the video in real-time"
    )

    yt_uri = st.text_input("Youtube URL", "https://youtu.be/l8pRSuU81PU")
    if st.button(
        "Submit",
        type="primary",
        on_click=disable,
        args=(True,),
        disabled=st.session_state.processing,
    ):
        result = process_video(yt_uri)
        if result is not None:
            st.session_state.videos.append(result)
        st.session_state.processing = False
        st.rerun()

    st.header("Processed videos:")
    st.write("Here are the videos you have processed so far:")
    st.write(
        "You can download the transcription of the video by clicking on the corresponding video below"
    )
    for video in st.session_state.videos:
        with st.container(border=True):
            st.title(video["title"])
            st.image(video["thumbnail"], width=320)
            if st.button("Download transcription"):
                with open(
                    st.session_state.tempfolder / f"{video['video_id']}.txt"
                ) as transcript_file:
                    st.download_button(
                        f"Download transcription {video['video_id']}",
                        transcript_file,
                        file_name=video["video_id"],
                    )


if __name__ == "__main__":
    main()
