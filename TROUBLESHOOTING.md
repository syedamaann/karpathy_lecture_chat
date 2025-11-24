# Troubleshooting Guide

## FFmpeg Error 183 - RESOLVED

### Problem
The application was encountering `ERROR: ffmpeg exited with code 183` when trying to download YouTube videos.

### Root Cause
Error code 183 on Windows-based systems means "can't create file because it already exists". In the Docker context, this was caused by using ffmpeg as an external downloader, which had issues with temporary file handling.

### Solution Applied
1. **Removed external ffmpeg downloader**: Changed from using ffmpeg as external downloader to yt-dlp's native downloader
2. **Added FFmpegExtractAudio post-processor**: FFmpeg is now used only for post-processing (audio extraction), not for downloading
3. **Added overwrites flag**: Ensures any file conflicts are resolved by overwriting

### Configuration Changes
```python
ydl_opts = {
    'format': 'bestaudio/best',  # Simple, flexible format selector
    'overwrites': True,  # Force overwrite files
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'm4a',
    }],
    'prefer_ffmpeg': True,
    'keepvideo': False,
}
```

## Format Not Available Error - RESOLVED

### Problem
After fixing error 183, encountered: `ERROR: [youtube] Requested format is not available`

### Root Cause
1. The format selector `bestaudio[ext=m4a]` was too restrictive
2. YouTube serves audio as m3u8 streams (HLS protocol), not direct m4a files
3. The strict format filter prevented yt-dlp from selecting available formats

### Solution Applied
1. **Simplified format selector**: Changed from `bestaudio[ext=m4a]` to `bestaudio/best`
2. **Added post-processor**: Use FFmpegExtractAudio to convert downloaded audio to m4a format
3. **Verified format availability**: Tested that format 234 (audio-only) is successfully selected

### How It Works Now

1. **Download**: yt-dlp downloads the best available audio format (m3u8/HLS stream)
2. **Post-process**: FFmpeg extracts and converts the audio to m4a format
3. **Transcribe**: OpenAI Whisper processes the m4a file
4. **Generate Infographic**: Nano Banana Pro creates visual summary
5. **Display**: Both transcript and infographic are shown in the UI

## Complete Workflow

```
User submits YouTube URL
        ↓
yt-dlp downloads audio (format: bestaudio)
        ↓
FFmpeg post-processes to m4a
        ↓
OpenAI Whisper transcribes audio → VTT format
        ↓
Transcript stored in Pinecone (for search)
        ↓
GPT-4 summarizes key points
        ↓
Nano Banana Pro generates infographic
        ↓
Display transcript + infographic in UI
```

## Tested and Verified

- ✅ Video download works with `bestaudio/best` format selector
- ✅ FFmpeg post-processing extracts audio successfully
- ✅ No more error 183 (file exists conflict)
- ✅ No more format not available errors
- ✅ Infographic generation integrated
- ✅ UI displays both transcript and infographic

## If Issues Persist

### 1. Update yt-dlp
If YouTube changes their API again, update yt-dlp:
```bash
docker exec 06_karpathylecturechat-yt-whisper-1 pip install --upgrade yt-dlp
```

### 2. Check Available Formats
To see what formats are available for a specific video:
```bash
docker exec 06_karpathylecturechat-yt-whisper-1 yt-dlp --list-formats "VIDEO_URL"
```

### 3. Test Format Selector
To test if a format selector works:
```bash
docker exec 06_karpathylecturechat-yt-whisper-1 yt-dlp -f "bestaudio/best" --simulate "VIDEO_URL"
```

### 4. Verbose Output
For detailed debugging information:
```python
ydl_opts = {
    'verbose': True,  # Add this for debugging
    # ... rest of options
}
```

## References

- [yt-dlp GitHub Issues - Error 183](https://github.com/yt-dlp/yt-dlp/issues/9059)
- [yt-dlp Format Selection](https://github.com/yt-dlp/yt-dlp#format-selection)
- [yt-dlp Complete Tutorial 2025](https://ostechnix.com/yt-dlp-tutorial/)
- [Using FFmpeg with YT-DLP](https://www.rendi.dev/post/using-ffmpeg-with-yt-dlp)
