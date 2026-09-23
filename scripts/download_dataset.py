import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
NSYNTH_DIR = DATA_DIR / "nsynth-valid"
AUDIO_DIR = NSYNTH_DIR / "audio"
METADATA_PATH = NSYNTH_DIR / "examples.json"
TARGET_DIR = DATA_DIR / "raw" / "keyboard"

NSYNTH_URL = (
    "http://download.magenta.tensorflow.org/"
    "datasets/nsynth/nsynth-valid.jsonwav.tar.gz"
)


def download_archive():
    """Download the NSynth validation archive."""

    archive_path = DATA_DIR / "nsynth-valid.tar.gz"

    print(f"Downloading NSynth validation dataset...")
    print(f"URL: {NSYNTH_URL}")

    urllib.request.urlretrieve(NSYNTH_URL, archive_path)

    return archive_path


def validate_archive(archive_path):
    """Check that the downloaded archive is a valid gzip/tar archive."""

    print("Validating NSynth archive...")

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            # Reading the member list forces gzip/tar validation.
            members = tar.getmembers()

        print(f"Archive is valid ({len(members)} files).")
        return True

    except (tarfile.TarError, EOFError, OSError) as exc:
        print(f"Invalid NSynth archive: {exc}")
        return False


def extract_metadata(archive_path):
    """
    Extract only examples.json from the NSynth archive.

    Existing audio files are left untouched.
    """

    print("Extracting examples.json...")

    with tarfile.open(archive_path, "r:gz") as tar:
        metadata_member = None

        for member in tar.getmembers():
            if member.name.endswith("/examples.json"):
                metadata_member = member
                break

        if metadata_member is None:
            raise RuntimeError(
                "Could not find examples.json inside the NSynth archive."
            )

        metadata_member.name = "examples.json"

        with tar.extractfile(metadata_member) as source:
            if source is None:
                raise RuntimeError("Could not read examples.json from archive.")

            with open(METADATA_PATH, "wb") as destination:
                shutil.copyfileobj(source, destination)

    print(f"Metadata saved to: {METADATA_PATH}")


def ensure_dataset():
    """
    Make sure the existing NSynth audio and metadata are available.

    Existing audio is never deleted.
    """

    if not AUDIO_DIR.exists():
        raise RuntimeError(
            f"NSynth audio directory does not exist:\n{AUDIO_DIR}"
        )

    wav_files = list(AUDIO_DIR.glob("*.wav"))

    print(f"Found {len(wav_files):,} existing NSynth WAV files.")

    if len(wav_files) == 0:
        raise RuntimeError("No WAV files found in the NSynth audio directory.")

    if METADATA_PATH.exists():
        print(f"Metadata already exists: {METADATA_PATH}")
        return

    archive_path = DATA_DIR / "nsynth-valid.tar.gz"

    if archive_path.exists():
        print("Existing NSynth archive found.")

        if not validate_archive(archive_path):
            print("Existing archive is corrupted.")
            print("Removing corrupted archive...")
            archive_path.unlink()
            archive_path = None
    else:
        archive_path = None

    if archive_path is None:
        archive_path = download_archive()

        if not validate_archive(archive_path):
            archive_path.unlink(missing_ok=True)

            raise RuntimeError(
                "The NSynth archive downloaded but is invalid/corrupted."
            )

    extract_metadata(archive_path)

def prepare_keyboard_samples():
    """Copy only keyboard-family samples into data/raw/keyboard."""

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print("Parsing metadata and copying keyboard samples...")

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    copied = 0
    missing = 0

    for note_id, note_data in tqdm(metadata.items()):

        is_keyboard = (
            note_data.get("instrument_family") == 4
            or note_data.get("instrument_family_str") == "keyboard"
        )

        if not is_keyboard:
            continue

        src_audio = AUDIO_DIR / f"{note_id}.wav"
        dst_audio = TARGET_DIR / f"{note_id}.wav"

        if not src_audio.exists():
            missing += 1
            continue

        if dst_audio.exists():
            continue

        shutil.copy2(src_audio, dst_audio)
        copied += 1

    print()
    print(f"Successfully prepared {copied:,} keyboard samples.")
    print(f"Output directory: {TARGET_DIR}")

    if missing:
        print(f"Warning: {missing:,} keyboard audio files were missing.")


def download_nsynth_keyboard():
    """
    Prepare NSynth validation data for the keyboard model.

    Existing audio is preserved. Missing metadata is downloaded/extracted,
    then keyboard samples are copied to data/raw/keyboard/.
    """

    print("=" * 60)
    print("NSynth Keyboard Dataset Preparation")
    print("=" * 60)

    ensure_dataset()
    prepare_keyboard_samples()

    print()
    print("Dataset preparation complete.")
    print("Next step:")
    print("  python3 scripts/prepare_dataset.py --instrument keyboard")


if __name__ == "__main__":
    download_nsynth_keyboard()
