"""
EcoPulse - Download the ESC-50 background/negative audio dataset

"""
import argparse
import io
import zipfile
from pathlib import Path

import requests

ESC50_ZIP_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"


def download_esc50(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ESC-50 (~600MB) from {ESC50_ZIP_URL} ...")
    resp = requests.get(ESC50_ZIP_URL, stream=True, timeout=120)
    resp.raise_for_status()

    buf = io.BytesIO()
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=1 << 20):
        buf.write(chunk)
        downloaded += len(chunk)
        print(f"\r  {downloaded / 1e6:.1f} MB", end="", flush=True)
    print()

    print("Extracting audio/ and meta/ ...")
    with zipfile.ZipFile(buf) as z:
        for name in z.namelist():
            if "/audio/" in name and name.endswith(".wav"):
                target = out_dir / Path(name).name
                with z.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())
            elif name.endswith("esc50.csv"):
                with z.open(name) as src, open(out_dir / "esc50_meta.csv", "wb") as dst:
                    dst.write(src.read())

    n_files = len(list(out_dir.glob("*.wav")))
    print(f"Done. {n_files} background .wav files saved to {out_dir}")
    print("Citation: K. J. Piczak, 'ESC: Dataset for Environmental Sound "
          "Classification', Proc. 23rd ACM Intl. Conf. on Multimedia, 2015.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default=r"D:\Researches\Data Odyssey 2026\unique\Data Odyssey 2026 Project\data\background",
    )
    args = parser.parse_args()
    download_esc50(args.out_dir)
