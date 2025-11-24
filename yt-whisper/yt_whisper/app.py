import os
import re
import tempfile
from pathlib import Path
from tempfile import mkdtemp

import streamlit as st
import yt_dlp
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from streamlit.logger import get_logger

from yt_whisper.vtt_utils import merge_webvtt_to_list

load_dotenv(".env")

logger = get_logger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_TOKEN"))
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


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


def generate_infographic(transcript: str, video_title: str, video_id: str) -> str:
    """Generate an infographic from the video transcript using Nano Banana Pro"""
    try:
        with st.spinner("Generating infographic from transcript..."):
            logger.info("Generating infographic with Nano Banana Pro")

            # Create a summary of the transcript for the infographic
            summary_prompt = f"""
            Create a concise summary of the following video transcript in bullet points (max 5-7 key points):

            Title: {video_title}

            Transcript:
            {transcript[:4000]}
            """

            summary_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=500
            )

            key_points = summary_response.choices[0].message.content

            # Generate infographic prompt
            infographic_prompt = f"""
            Create a professional, visually appealing infographic with the title "{video_title}" at the top.

            Include these key points in an organized, easy-to-read layout:
            {key_points}

            Use a modern design with:
            - Clean typography with legible text
            - Professional color scheme (blue/purple tones)
            - Clear hierarchy and spacing
            - Icons or visual elements to illustrate key concepts
            - 16:9 aspect ratio, 2K resolution
            - Footer text: "Generated from video transcript"
            """

            # Generate infographic using Nano Banana Pro
            response = gemini_client.models.generate_content(
                model='gemini-3-pro-image-preview',
                contents=infographic_prompt
            )

            # Save the generated image
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            image_data = part.inline_data.data
                            infographic_path = st.session_state.tempfolder / f"{video_id}_infographic.png"

                            with open(infographic_path, "wb") as img_file:
                                img_file.write(image_data)

                            logger.info(f"Infographic saved to {infographic_path}")
                            return str(infographic_path)

            logger.warning("No image data found in response")
            return None

    except Exception as e:
        logger.error(f"Error generating infographic: {e}")
        st.error(f"Failed to generate infographic: {str(e)}", icon="🚨")
        return None


@st.cache_data(ttl=3600)  # Cache for 1 hour, allows cache refresh
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
                # Configure yt-dlp options for reliable YouTube audio download
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(tmp_dir, 'audio.%(ext)s'),
                    'overwrites': True,
                    'quiet': True,
                    'no_warnings': True,
                    'fragment_retries': 10,
                    'skip_unavailable_fragments': True,
                    'ignoreerrors': False,
                    'noprogress': True,
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

                # Generate infographic from the transcript
                infographic_path = generate_infographic(
                    whisper_transcript,
                    video_title,
                    video_id
                )

                return {
                    "video_id": video_id,
                    "title": video_title,
                    "thumbnail": thumbnail_url,
                    "infographic_path": infographic_path,
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
        "You can download the transcription and view the generated infographic for each video below"
    )
    for video in st.session_state.videos:
        with st.container(border=True):
            st.title(video["title"])
            st.image(video["thumbnail"], width=320)

            # Display infographic if available
            if video.get("infographic_path") and os.path.exists(video["infographic_path"]):
                st.subheader("Generated Infographic")
                st.image(video["infographic_path"], use_container_width=True)

            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Download transcription - {video['video_id']}", key=f"trans_{video['video_id']}"):
                    with open(
                        st.session_state.tempfolder / f"{video['video_id']}.txt"
                    ) as transcript_file:
                        st.download_button(
                            f"Download transcription {video['video_id']}",
                            transcript_file,
                            file_name=f"{video['video_id']}.txt",
                            key=f"dl_trans_{video['video_id']}"
                        )

            with col2:
                if video.get("infographic_path") and os.path.exists(video["infographic_path"]):
                    with open(video["infographic_path"], "rb") as img_file:
                        st.download_button(
                            f"Download infographic",
                            img_file,
                            file_name=f"{video['video_id']}_infographic.png",
                            mime="image/png",
                            key=f"dl_info_{video['video_id']}"
                        )


if __name__ == "__main__":
    main()
