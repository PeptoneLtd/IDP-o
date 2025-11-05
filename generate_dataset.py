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

import logging
import os
import traceback
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from subprocess import check_call

import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Use IDP-o to generate a dataset of ensemble configurations from a csv/fasta file",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input", "-i", type=str, required=True, help="input csv/fasta file")
    parser.add_argument("--outfolder", "-o", type=str, default="./ensembles", help="path name of the output folder")
    parser.add_argument(
        "-n",
        "--max_structures_in_ensemble",
        type=int,
        default=0,
        help="number of frames to generate for each fragment, default is to not create ensembles",
    )
    parser.add_argument(
        "--fragments_overlap",
        type=int,
        default=1,
        help="number of overlapping residues between fragments",
    )
    parser.add_argument(
        "--column_names",
        type=str,
        default="seq_name,fasta",
        help="names of the useful columns in the csv",
    )
    parser.add_argument("--shuffle", action="store_true", help="generate the ensembles in shuffled order")
    parser.add_argument(
        "--format",
        type=str,
        default="xtc",
        choices=["h5", "xtc", "pdb", "pdb.gz", "dcd"],
        help="format of the output files",
    )
    parser.add_argument("--overwrite", action="store_true", help="overwrite any existing ensemble files")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    name, fasta = args.column_names.split(",")
    if args.input.endswith(".fasta"):
        logger.info(f"reading sequences from '{args.input}'")
        ids, sequences = [], []
        with open(args.input) as f:
            for line in f:
                ids.append(line.strip().removeprefix(">"))
                sequences.append(next(f).strip())
        df = pd.DataFrame({name: ids, fasta: sequences})
        del ids, sequences
    elif args.input.endswith(".csv"):
        logger.info(f"reading '{name}' and '{fasta}' columns from '{args.input}'")
        df = pd.read_csv(args.input, usecols=[name, fasta])
    else:
        raise ValueError(f"unsupported file format: {args.input}")

    logger.info(f"loaded {len(df):_} sequences")
    for column in [fasta, name]:
        if df[column].duplicated().any():
            logger.warning(f"dropping {df[column].duplicated().sum()} duplicated {column} found")
            df = df.drop_duplicates(column).reindex(drop=True)

    counter = 0
    os.makedirs(args.outfolder, exist_ok=True)
    if args.max_structures_in_ensemble <= 0:
        logger.warning("set '--max_structures_in_ensemble' to actually generate the ensembles")
    else:
        if args.shuffle:
            logger.info("shuffling generation order, for easy parallelization")
            df = df.sample(frac=1).reset_index(drop=True)
        for i, row in df.iterrows():
            logger.info(f"++++++++++++++++++++++++++++ {i / len(df):.3%} ++++++++++++++++++++++++++++")
            logger.info(f"name={row[name]}, len={len(row[fasta])}, fasta={row[fasta]}")
            outpath = os.path.join(args.outfolder, f"{row[name]}.{args.format}")
            scratch_folder = os.path.join("/tmp", f"tmp-{row[name]}")
            os.makedirs(os.path.dirname(scratch_folder), exist_ok=True)
            cmd = [
                "python3",
                f"{os.path.dirname(os.path.abspath(__file__))}/scripts/build_ensemble.py",
                "--sequence",
                row[fasta],
                "--outpath",
                outpath,
                "--scratch_folder",
                scratch_folder,
                "--max_structures_in_ensemble",
                str(args.max_structures_in_ensemble),
            ]
            if args.overwrite:
                cmd.append("--overwrite")
            try:
                check_call(cmd)
                counter += 1
            except Exception:
                error_msg = traceback.format_exc()
                logger.warning(error_msg)
                with open(os.path.splitext(outpath)[0] + ".txt", "w") as f:
                    f.write(cmd + "\n" + error_msg + "\n")
    logger.info(f"generated {counter:_} ensembles")
