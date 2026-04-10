"""
MediBOT Model Download Script
Fetches the Llama-2-7B-Chat GGML model (7 GB) from HuggingFace (primary)
or Google Drive (fallback), then places it at:
    model/llama-2-7b-chat.ggmlv3.q8_0-002.bin

Usage:
    python download_model.py
    python download_model.py --source gdrive   # force Google Drive
    python download_model.py --source hf       # force HuggingFace
    python download_model.py --skip-hash       # skip SHA-256 verification
"""

import os
import sys
import hashlib
import argparse

# ---------------------------------------------------------------------------
# Asset definitions
# ---------------------------------------------------------------------------
_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_BASE, "model")
MODEL_FILE = "llama-2-7b-chat.ggmlv3.q8_0-002.bin"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# HuggingFace — TheBloke/Llama-2-7B-Chat-GGML (GGML q8_0 quantisation)
HF_REPO_ID = "TheBloke/Llama-2-7B-Chat-GGML"
HF_FILENAME = "llama-2-7b-chat.ggmlv3.q8_0.bin"  # single-file version on HF

# Google Drive (split upload — this is the copy your team uploaded)
GDRIVE_FILE_ID = "144I4EHgc7HdnHQlkXyh_7oVJrLgQK8cG"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# SHA-256 of the Google Drive copy (verify after download).
# Run `python -c "import hashlib; ..."` on a known-good copy to update this.
# Set to None to skip hash check with --skip-hash flag.
EXPECTED_SHA256 = None  # fill in after first successful download

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: str, chunk_mb: int = 8) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_mb * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hash(path: str, expected: str) -> bool:
    print(f"  Verifying SHA-256 … (this takes ~30 s for a 7 GB file)")
    actual = sha256_file(path)
    if actual == expected:
        print(f"  ✅ Hash verified: {actual[:16]}…")
        return True
    else:
        print(f"  ❌ Hash mismatch!")
        print(f"     Expected : {expected}")
        print(f"     Got      : {actual}")
        return False


def download_huggingface() -> bool:
    """Download via huggingface_hub (resumes automatically)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  huggingface_hub not installed — skipping HF source.")
        return False

    print(f"  Source: HuggingFace ({HF_REPO_ID} / {HF_FILENAME})")
    print("  This may take 20-60 minutes on a typical connection.")
    try:
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
        )
        # HF saves as HF_FILENAME; rename to match what api_bridge.py expects
        if os.path.basename(local_path) != MODEL_FILE:
            os.rename(local_path, MODEL_PATH)
        print(f"  ✅ Downloaded to: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"  ❌ HuggingFace download failed: {e}")
        return False


def download_gdrive() -> bool:
    """Download via gdown (Google Drive)."""
    try:
        import gdown
    except ImportError:
        print("  gdown not installed — run: pip install gdown")
        return False

    print(f"  Source: Google Drive (id={GDRIVE_FILE_ID})")
    print("  This may take 10-30 minutes depending on your connection.")
    try:
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
        print(f"  ✅ Downloaded to: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"  ❌ Google Drive download failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download MediBOT Llama-2 model")
    parser.add_argument("--source", choices=["hf", "gdrive", "auto"], default="auto",
                        help="Download source (default: try HF first, then GDrive)")
    parser.add_argument("--skip-hash", action="store_true",
                        help="Skip SHA-256 verification after download")
    args = parser.parse_args()

    print("=" * 60)
    print("  MediBOT — Llama-2 Model Downloader")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        size_gb = os.path.getsize(MODEL_PATH) / (1024 ** 3)
        print(f"  Model already present: {MODEL_PATH} ({size_gb:.1f} GB)")
        if not args.skip_hash and EXPECTED_SHA256:
            if not verify_hash(MODEL_PATH, EXPECTED_SHA256):
                print("  Deleting corrupt file and re-downloading …")
                os.remove(MODEL_PATH)
            else:
                return
        else:
            print("  Skipping hash check (no expected hash set or --skip-hash used).")
            return

    success = False

    if args.source == "hf":
        success = download_huggingface()
    elif args.source == "gdrive":
        success = download_gdrive()
    else:  # auto: HF first, GDrive fallback
        print("[1/2] Attempting HuggingFace …")
        success = download_huggingface()
        if not success:
            print("[2/2] Falling back to Google Drive …")
            success = download_gdrive()

    if not success:
        print("\n❌ All download sources failed.")
        print("   Manual option: place the file at:")
        print(f"   {MODEL_PATH}")
        sys.exit(1)

    if not args.skip_hash and EXPECTED_SHA256:
        if not verify_hash(MODEL_PATH, EXPECTED_SHA256):
            print("  Downloaded file appears corrupt. Try again or use --skip-hash.")
            sys.exit(1)
    elif not EXPECTED_SHA256:
        actual = sha256_file(MODEL_PATH)
        print(f"\n  SHA-256 of downloaded file: {actual}")
        print("  Paste this into EXPECTED_SHA256 in download_model.py to enable future checks.")

    print("\n✅ Model ready. You can now start the MediBOT AI engine:")
    print("   python api_bridge.py")


if __name__ == "__main__":
    main()
