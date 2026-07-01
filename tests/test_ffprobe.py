import subprocess
from src.util.ffprobe import duration_sec


def test_duration_sec(tmp_path):
    wav = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         str(wav)], check=True,
    )
    assert abs(duration_sec(wav) - 1.0) < 0.1
