# HEARSAY - HackGT 13

## Audio ingestion

Install the Python dependencies with `pip install -r requirements.txt`.
`ProcessFiles.load_audio` decodes with SoundFile first. FFmpeg is an optional
runtime fallback for containers that libsndfile cannot read (for example, M4A);
install the `ffmpeg` executable on `PATH` when those formats are required. The
fallback streams raw float32 PCM through stdout and never creates a converted
audio file.

Run the focused ingestion tests with:

```sh
python -m unittest discover -s tests -v
```
