# Learn better with Lecture Quiz and Smart Timestamps!

I needed a quick way to review and quiz myself on Karpathy’s dense lecture videos. I thought it would be cool to generate custom timestamps for rewatching specific segments based on my quiz responses. So, I made this project using OpenAI's Whisper for video transcription and Pinecone for semantic search to help myself and others have an efficient study workflow, with better revision through targeted video playback.

 <br />

https://github.com/syedamaann/karpathy_lecture_chat/assets/74735966/6ff84ba4-447b-42b6-b6f5-e45ff8f2ff04

 <br />


### Video processing with Whisper and Pinecone

The yt-whisper component processes video files with **Whisper**, generating time-stamped transcripts. These transcripts are then indexed in **Pinecone** for fast and accurate retrieval of specific video segments based on quiz responses. The docker-bot component integrates with the transcription service, providing an interactive chat interface. This bot uses the indexed transcripts in Pinecone to offer precise video segments for rewatching based on user queries.

### Docker for deployment

The architecture uses **Docker** for consistent deployment. **Poetry** manages dependencies for reproducible builds. Each component has separate Dockerfiles and entry points for modular development and testing. Scripts support both local and Docker execution.

<br />

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/syedamaann/karpathy_lecture_chat.git
cd karpathy_lecture_chat
```

### 2. Set Up Environment Variables and Configure API Key
- Create a `.env` file with the following content:
  ```bash
  OPENAI_API_KEY='your-api-key'
  ```

- Add the API key to your shell configuration:
  ```bash
  echo "export OPENAI_API_KEY='your-api-key'" >> ~/.zshrc
  source ~/.zshrc
  ```

### 3. Using Docker
- **Build and Run with Docker Compose:**
  ```bash
  docker compose up --build
  ```

### 4. Running Locally (Without Docker)
- **Install Dependencies:**
  ```bash
  # For docker-bot
  cd docker-bot
  poetry install
  
  # For yt-whisper
  cd ../yt-whisper
  poetry install
  ```

- **Run the Scripts Locally:**
  ```bash
  # Run docker-bot
  cd docker-bot
  ./scripts/run_locally.sh
  
  # Run yt-whisper
  cd ../yt-whisper
  ./scripts/run_locally.sh
  ```

- **Running Tests:**
  ```bash
  # For docker-bot
  cd docker-bot
  poetry run pytest
 
  # For yt-whisper
  cd ../yt-whisper
  poetry run pytest
  ```

### 5. Accessing the Services
- `yt-whisper` runs on `localhost:8503`
- `docker-bot` runs on `localhost:8504`
