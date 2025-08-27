

::: thagomizer.video
    options:
      filters: 
        - "!^_"
      show_root_toc_entry: false
      show_root_heading: false
      show_category_heading: false
      show_object_full_path: false


### Subtitle stream selection

- When multiple English subtitle streams exist (for example, an English [Forced] track and a full English track), `get_english_subtitle_stream` selects the English stream with the highest `NUMBER_OF_FRAMES` value, which generally corresponds to the most complete subtitle track.


