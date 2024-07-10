# Learn better with Lecture Quiz and Smart Timestamps!


I needed a quick way to review and quiz myself on Karpathy’s dense lecture videos. I thought it would be cool to generate custom timestamps for rewatching specific segments based on my quiz responses. So, I made this project using OpenAI's Whisper for video transcription and Pinecone for semantic search to help myself and others have an efficient study workflow, with better revision through targeted video playback.

The yt-whisper component processes video files with Whisper, generating time-stamped transcripts. These transcripts are then indexed in Pinecone for fast and accurate retrieval of specific video segments based on quiz responses. The docker-bot component integrates with the transcription service, providing an interactive chat interface. This bot uses the indexed transcripts in Pinecone to offer precise video segments for rewatching based on user queries.
