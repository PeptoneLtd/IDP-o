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
import traceback

import extract_structures_from_foldcomp_database
import fasta_search_in_foldcomp_database
import join_fragments


def main(
    sequence: str,
    foldcomp_fasta: str,
    foldcomp_db: str,
    n_max_structures_per_fragment: int,
    scratch_folder: str,
    outpath: str,
    reduction_factor: int,
    joins_to_attempt_per_pairing: int,
    max_structures_in_ensemble: int,
    exclude_cis_omega: bool = False,
    rmsd_sort: bool = False,
) -> None:
    logger = logging.getLogger()
    standard_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    invalid_aa = [aa for aa in sequence if aa not in standard_amino_acids]
    if len(invalid_aa) > 0:
        raise ValueError(f"Input sequence contains invalid amino acid: {invalid_aa}")

    fragments_dir = os.path.join(scratch_folder, "fragment_ensembles")
    if exclude_cis_omega:
        fragments_dir += "-exclude_cis_omega"
    bytes_start_pkl = os.path.join(scratch_folder, "byte_starts.pkl")

    os.makedirs(scratch_folder, exist_ok=True)
    os.makedirs(fragments_dir, exist_ok=True)

    logger.info("Searching foldcomp database")
    fasta_search_in_foldcomp_database.main(sequence, foldcomp_fasta, bytes_start_pkl, reduction_factor)
    logger.info("extracting structures from foldcomp database")
    extract_structures_from_foldcomp_database.main(
        foldcomp_fasta,
        foldcomp_db,
        fragments_dir,
        n_max_structures_per_fragment,
        bytes_start_pkl,
        exclude_cis_omega,
    )
    logger.info("joining fragments")
    join_fragments.main(
        sequence,
        outpath,
        fragments_dir,
        joins_to_attempt_per_pairing,
        max_structures_in_ensemble,
        rmsd_sort,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Builds a protein ensemble from fragments given a sequence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sequence", required=True)
    parser.add_argument(
        "--foldcomp_fasta",
        help="""fasta file related to a foldcomp compressed database with the format:
  >ByteNumberAtWhichFoldcompEntryRelatingToThisSequenceBegins
  Sequence

  e.g.
  >677
  LPYPAHLEILVQTLRYWIRDVSSL
  denotes the entry relating to LPYPAHLEILVQTLRYWIRDVSSL in the FoldComp database begins at byte 677 in the foldcomp database file""",
        default="/data/afdb/afdb_uniprot_v4.fasta",
    )
    parser.add_argument(
        "--foldcomp_db",
        help="Path to the foldcomp database file",
        default="/data/afdb/afdb_uniprot_v4",
    )
    parser.add_argument(
        "--n_max_structures_per_fragment",
        type=int,
        default=1000,
        help="Max number of structures per fragment to extract",
    )
    parser.add_argument("--outpath", help="Path to write ensemble to", required=True)
    parser.add_argument("--scratch_folder", help="Folder to write intermediates to", required=False, default="/tmp")
    parser.add_argument(
        "--reduction_factor",
        type=int,
        default=1,
        help="Factor by which to reduce the proportion of fasta searched, so 10 means only a tenth of the database will be searched",
    )
    parser.add_argument("--joins_to_attempt_per_pairing", type=int, default=500000)
    parser.add_argument("--max_structures_in_ensemble", type=int, default=100)
    parser.add_argument(
        "--exclude_cis_omega",
        default=False,
        action="store_true",
        help="Whether to exclude structures with cis omega angles",
    )
    parser.add_argument(
        "--rmsd_sort",
        action="store_true",
        default=False,
        help="Whether to sort by rmsd matrix and align each frame, for nicer visualization",
    )
    args = parser.parse_args()
    try:
        logger.info("building ensemble")
        main(
            sequence=args.sequence,
            foldcomp_fasta=args.foldcomp_fasta,
            foldcomp_db=args.foldcomp_db,
            n_max_structures_per_fragment=args.n_max_structures_per_fragment,
            scratch_folder=args.scratch_folder,
            outpath=args.outpath,
            reduction_factor=args.reduction_factor,
            joins_to_attempt_per_pairing=args.joins_to_attempt_per_pairing,
            max_structures_in_ensemble=args.max_structures_in_ensemble,
            exclude_cis_omega=args.exclude_cis_omega,
            rmsd_sort=args.rmsd_sort,
        )
        logger.info("ensemble built")

    except Exception as e:
        logger.error("error while building ensemble", exc_info=True)
        logger.error(traceback.format_exc())
        raise e
