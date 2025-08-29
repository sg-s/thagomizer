

::: thagomizer.video
    options:
      filters: 
        - "!^_"
      show_root_toc_entry: false
      show_root_heading: false
      show_category_heading: false
      show_object_full_path: false

### Video Bitrate Detection

The `get_video_bitrate` function automatically detects the bitrate of input video files using multiple fallback methods:

1. **Primary method**: Extracts bitrate from the video container metadata
2. **Secondary method**: Extracts bitrate from the video stream metadata  
3. **Fallback method**: Estimates bitrate from file size and duration

This ensures reliable bitrate detection even for files with incomplete metadata.

### Quality-Preserving Transcoding

The `transcode_for_streaming` function now uses a two-pass encoding approach that targets approximately the same bitrate as the input file:

- **First pass**: Analyzes the video content for optimal encoding decisions
- **Second pass**: Encodes with precise bitrate control and quality constraints
- **Quality factor**: Adjustable multiplier (default 1.0 = same size, 0.8 = 80% size, etc.)
- **Rate control**: Uses `maxrate` and `bufsize` parameters for consistent bitrate

This approach preserves visual quality while leveraging AV1's superior compression efficiency.

### Subtitle stream selection

- When multiple English subtitle streams exist (for example, an English [Forced] track and a full English track), `get_english_subtitle_stream` selects the English stream with the highest `NUMBER_OF_FRAMES` value, which generally corresponds to the most complete subtitle track.


