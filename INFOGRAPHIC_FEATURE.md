# Infographic Generation Feature

## Overview

This feature automatically generates professional infographics from video transcripts using Google's Nano Banana Pro (Gemini 3 Pro Image Preview) model. After a video is processed and transcribed, the system creates a visual summary that highlights the key points from the lecture.

## How It Works

1. **Transcription**: Video is first transcribed using OpenAI Whisper
2. **Summarization**: GPT-4 creates a concise summary of key points from the transcript
3. **Infographic Generation**: Nano Banana Pro generates a professional 2K infographic with:
   - Clean, legible typography
   - Professional color scheme (blue/purple tones)
   - Clear visual hierarchy
   - Icons and visual elements
   - 16:9 aspect ratio
4. **Display**: The infographic is automatically displayed in the UI and can be downloaded

## Features

- **Automatic Generation**: Infographics are created automatically after transcription
- **Professional Design**: Uses Nano Banana Pro's state-of-the-art image generation
- **High Resolution**: 2K resolution images suitable for presentations and study materials
- **Text Rendering**: Nano Banana Pro excels at rendering legible, stylized text in images
- **Download Support**: Both transcript and infographic can be downloaded

## API Requirements

### Google API Key
You need a Google API key with access to the Gemini API:
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create or select a project
3. Generate an API key
4. Add it to your `.env` file as `GOOGLE_API_KEY`

### Pricing
- Nano Banana Pro costs $0.139 per 2K image
- The feature generates one 2K infographic per video processed

## Technical Details

### Model Used
- **Model ID**: `gemini-3-pro-image-preview`
- **SDK**: `google-genai` Python package
- **Resolution**: 2K (2048x1152)
- **Aspect Ratio**: 16:9

### Implementation
The infographic generation is implemented in `yt_whisper/app.py`:
- `generate_infographic()`: Main function that orchestrates the generation
- Uses GPT-4 for transcript summarization
- Uses Nano Banana Pro for image generation
- Saves images to the temporary folder with naming pattern: `{video_id}_infographic.png`

### Code Flow
```python
# 1. Summarize transcript with GPT-4
summary = create_summary(transcript, video_title)

# 2. Generate infographic with Nano Banana Pro
response = gemini_client.models.generate_content(
    model='gemini-3-pro-image-preview',
    contents=infographic_prompt
)

# 3. Save and display the image
save_image(response.image_data, video_id)
```

## UI Updates

The processed videos section now includes:
- Thumbnail preview
- **Generated Infographic** display (full width)
- Download buttons for both transcript and infographic
- Side-by-side download buttons for easy access

## Error Handling

- If infographic generation fails, the video is still processed and transcript is available
- Errors are logged and displayed to the user
- The system continues to function even if Google API is unavailable

## Future Enhancements

Potential improvements:
- Multiple infographic styles/templates
- User-customizable color schemes
- Interactive infographic editor
- Export to different formats (PDF, SVG)
- Batch infographic regeneration with different prompts

## References

- [Nano Banana Pro Official Blog](https://blog.google/technology/ai/nano-banana-pro/)
- [Gemini API Image Generation Docs](https://ai.google.dev/gemini-api/docs/image-generation)
- [Google AI Studio](https://aistudio.google.com/)
