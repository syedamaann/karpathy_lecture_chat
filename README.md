# Learn better with Lecture Quiz and Smart Timestamps!

I needed a quick way to review and quiz myself on Karpathy’s dense lecture videos. I thought it would be cool to generate custom timestamps for rewatching specific segments based on my quiz responses. So, I made this project using OpenAI's Whisper for video transcription and Pinecone for semantic search to help myself and others have an efficient study workflow, with better revision through targeted video playback.

 <br />

https://github.com/syedamaann/karpathy_lecture_chat/assets/74735966/6ff84ba4-447b-42b6-b6f5-e45ff8f2ff04

 <br />


### Video processing with Whisper and Pinecone

The yt-whisper component processes video files with **Whisper**, generating time-stamped transcripts. These transcripts are then indexed in **Pinecone** for fast and accurate retrieval of specific video segments based on quiz responses. The docker-bot component integrates with the transcription service, providing an interactive chat interface. This bot uses the indexed transcripts in Pinecone to offer precise video segments for rewatching based on user queries.

### Docker for deployment

The architecture uses **Docker** for consistent deployment. **Poetry** manages dependencies for reproducible builds. Each component has separate Dockerfiles and entry points for modular development and testing. Scripts support both local and Docker execution.
