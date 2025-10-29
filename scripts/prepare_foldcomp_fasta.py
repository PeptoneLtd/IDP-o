# Copyright 2025 Peptone Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import platform
import shutil
import stat
import subprocess
import tarfile
import zipfile
from urllib.request import urlretrieve

import foldcomp
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_labels_to_offset_dict(foldcomp_db: str) -> dict[str, int]:
    index = (
        pd.read_csv(f"{foldcomp_db}.index", header=None, sep="\t", names=["id", "start", "end"])
        .drop("end", axis=1)
        .set_index("id")
    )
    lookup = (
        pd.read_csv(
            f"{foldcomp_db}.lookup",
            header=None,
            sep="\t",
            names=["id", "label", "other"],
        )
        .drop("other", axis=1)
        .set_index("id")
    )
    return lookup.join(index).set_index("label").to_dict()["start"]


def extract_fasta(foldcomp_db: str, threads: int = 8):
    try:
        foldcomp_bin = shutil.which("foldcomp")
        if foldcomp_bin is None and os.path.isfile("./foldcomp") and os.access("./foldcomp", os.X_OK):
            foldcomp_bin = "./foldcomp"
    except shutil.Error:
        foldcomp_bin = None

    if foldcomp_bin is None:
        logger.warning("foldcomp command not found, downloading...")
        p_system = platform.system()
        if p_system == "Windows":
            urlretrieve("https://mmseqs.com/foldcomp/foldcomp-windows-x64.zip", "foldcomp.zip")
        elif p_system == "Darwin":
            urlretrieve(
                "https://mmseqs.com/foldcomp/foldcomp-macos-universal.tar.gz",
                "foldcomp.tar.gz",
            )
        elif p_system == "Linux":
            p_machine = platform.machine()
            if p_machine == "x86_64":
                urlretrieve(
                    "https://mmseqs.com/foldcomp/foldcomp-linux-x86_64.tar.gz",
                    "foldcomp.tar.gz",
                )
            elif "arm64" in p_machine.lower():
                urlretrieve(
                    "https://mmseqs.com/foldcomp/foldcomp-linux-arm64.tar.gz",
                    "foldcomp.tar.gz",
                )
            else:
                raise RuntimeError(f"Architecture {p_machine} is not supported")
        else:
            raise RuntimeError(f"Platform {p_system} is not supported")

        if p_system == "Windows":
            with zipfile.open("foldcomp.zip", "r") as z:
                z.extractall(path=".")
            os.remove("foldcomp.zip")
        else:
            with tarfile.open("foldcomp.tar.gz", "r:gz") as t:
                t.extractall(path=".")
                st = os.stat("foldcomp")
                os.chmod("foldcomp", st.st_mode | stat.S_IEXEC)
            os.remove("foldcomp.tar.gz")

        foldcomp_bin = "./foldcomp"
    else:
        logger.info(f"found foldcomp binary at {foldcomp_bin}")

    logger.info("extracting foldcomp fasta")
    subprocess.run([foldcomp_bin, "extract", foldcomp_db, "--fasta", "--threads", str(threads)], check=False)


def create_offset_fasta(foldcomp_db: str, offset_dict: dict[str, int]):
    with (
        open(f"{foldcomp_db}_fasta") as in_f,
        open(f"{foldcomp_db}.fasta", "w") as of,
    ):
        for line in in_f:
            if line.startswith(">"):
                label = line.lstrip(">").rstrip()
                offset = offset_dict[label]
                of.write(f">{offset}\n")
            else:
                of.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare fasta file with foldcomp offset as labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--foldcomp-db", required=False, default="afdb_uniprot_v4")
    parser.add_argument("--threads", required=False, default=8)
    parser.add_argument("--workdir", required=False, default="/data/foldcomp_db")
    args = parser.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    os.chdir(args.workdir)

    if not os.path.isfile(args.foldcomp_db):
        logger.info("downloading foldcomp db")
        foldcomp.setup(args.foldcomp_db)
    else:
        logger.info("foldcomp db found locally, skipping download")

    logger.info("computing offset map")
    offsets = get_labels_to_offset_dict(foldcomp_db=args.foldcomp_db)

    if not os.path.isfile(f"{args.foldcomp_db}_fasta"):
        logger.info("extracting foldcomp fasta")
        extract_fasta(foldcomp_db=args.foldcomp_db, threads=args.threads)
    else:
        logger.info(f"{args.foldcomp_db}_fasta found locally, skipping extraction")

    logger.info("writing offset fasta")
    create_offset_fasta(foldcomp_db=args.foldcomp_db, offset_dict=offsets)
