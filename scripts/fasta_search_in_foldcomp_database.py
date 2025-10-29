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
import pickle as pkl

import cupy as cp
import numpy as np

PROTEIN_LETTERS = "ACDEFGHIKLMNPQRSTVWY"


def encode_sequence(sequence):
    return cp.frombuffer(sequence.encode(), dtype=cp.uint8)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def decode_sequence_int8(seq_arr: np.array):
    v = np.full((90,), "", dtype="<U1")
    v[encode_sequence(PROTEIN_LETTERS).get()] = list(PROTEIN_LETTERS)
    return v[seq_arr]


def extract_all_byte_starts(
    foldcomp_fasta,
    fragments,
    chunk_size=320000000,
    reduction_factor: int = 100,
    nmax_char_ints=13,
):
    """
    Searches for all instances of fragment sequences in a specially constructed fasta file
    while the fasta file related to a foldcomp compressed database with the format:
    >ByteNumberAtWhichFoldcompEntryRelatingToThisSequenceBegins
    Sequence

    e.g.
    >677
    LPYPAHLEILVQTLRYWIRDVSSL
    denotes the entry relating to LPYPAHLEILVQTLRYWIRDVSSL in the FoldComp database begins at byte 677 in the foldcomp database file

    returns dictionary list of hits for each fragment with three parts:
    (
      hit_idxs: The byte in the fasta file where the hit to the input sequence begins
      byte_starts: The byte in the FoldComp DB file where the entry for the hit begins
      aa_start_index: The residue index for the first residue in the hit
    )
    """
    n_characters = os.path.getsize(foldcomp_fasta) // reduction_factor
    if chunk_size > n_characters:
        chunk_size = (n_characters // 8) * 8

    start_char = encode_sequence(">").ravel()
    pow10 = 10 ** cp.arange(nmax_char_ints, dtype=cp.int64)[::-1]

    hits = []
    n_chunks = int(np.ceil(n_characters / chunk_size))
    with open(foldcomp_fasta, "rb") as f:
        for chunk_index in range(n_chunks):
            offset = chunk_size * chunk_index
            f.seek(offset)
            tmp_data = f.read(chunk_size)

            # if make sure data is 0-padded to a multiple of 8 bytes for the last chunk
            if len(tmp_data) < chunk_size:
                tmp_data += bytes(chunk_size - len(tmp_data))

            indexs = cp.frombuffer(tmp_data).view(cp.uint8)
            start_bytes = cp.flatnonzero(indexs == start_char)

            chunk_hits = {}
            for sequence in fragments:
                sequence_encoded = encode_sequence(sequence)
                ncoeff = sequence_encoded.shape[0]

                energies = cp.ones(chunk_size - ncoeff, dtype=bool)
                for j in range(ncoeff):
                    energies &= indexs[j : chunk_size - (ncoeff - j)] == sequence_encoded[j]
                hit_idxs = cp.flatnonzero(energies)

                if len(hit_idxs) <= 0:
                    hit_idxs = cp.zeros(len(energies))
                    continue

                # Hits where sequence is split between chunks are dropped
                hit_idxs = hit_idxs[hit_idxs > start_bytes[0]]

                # Map to indexing in foldcomp database using header in the fasta
                byte_index_of_gt_sign = start_bytes[cp.searchsorted(start_bytes, hit_idxs) - 1]
                byte_indexs = (cp.arange(nmax_char_ints)[None] + 1) + byte_index_of_gt_sign[..., None]
                x = indexs[byte_indexs]
                mask = (x >= 48) & (x <= 58)
                n_char_in_bytes_start = mask.sum(1)
                # convert int8 array representation of the code e.g. 2343242\nFE to int64 of 2343242
                byte_starts = ((x - 48) * mask * pow10).sum(1) // 10 ** (nmax_char_ints - n_char_in_bytes_start)
                # get the amino acid index of start of sequence match within that sequence
                aa_start_index = hit_idxs - byte_index_of_gt_sign - n_char_in_bytes_start - 2

                hit_idxs += offset
                chunk_hits[sequence] = (
                    hit_idxs.get(),
                    byte_starts.get(),
                    aa_start_index.get(),
                )
            hits.append(chunk_hits)

    nvalues = len(next(iter(hits[0].values())))
    # if we don't have hits for a fragment, its tuple is ([],[],[])
    hits = {
        fragment: tuple(
            np.concatenate(
                [
                    x.get(
                        fragment,
                        (
                            np.array([], dtype=np.int64),
                            np.array([], dtype=np.int64),
                            np.array([], dtype=np.int64),
                        ),
                    )[i]
                    for x in hits
                ],
            )
            for i in range(nvalues)
        )
        for fragment in fragments
    }
    return hits


def generate_fragments(s, overlap, seq_len):
    shift = seq_len - overlap
    fragments = [s[i * shift : i * shift + seq_len] for i in range((len(s) - seq_len) // shift + 2)]
    if len(fragments[-1]) == overlap and len(fragments) > 1:
        fragments.pop()
    return fragments


def main(sequence, foldcomp_fasta, pkl_outpath, reduction_factor, overlap=2, seq_len=6):
    fragments = generate_fragments(sequence, overlap=overlap, seq_len=seq_len)
    hits = extract_all_byte_starts(
        foldcomp_fasta=foldcomp_fasta,
        fragments=fragments,
        reduction_factor=reduction_factor,
    )
    os.makedirs(os.path.dirname(pkl_outpath), exist_ok=True)
    with open(pkl_outpath, "wb") as f:
        pkl.dump(hits, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get hits in fasta file for byte starts of foldcomp entries. May cut the sequence slightly shorter than the input. To avoid this pad with glycines and then cut later (or do a general fix)",
    )
    parser.add_argument(
        "--foldcomp_fasta",
        help="""fasta file related to a foldcomp compressed database with the format:
  >ByteNumberAtWhichFoldcompEntryRelatingToThisSequenceBegins
  Sequence

  e.g.
  >677
  LPYPAHLEILVQTLRYWIRDVSSL
  denotes the entry relating to LPYPAHLEILVQTLRYWIRDVSSL in the FoldComp database begins at byte 677 in the foldcomp database file""",
        default="/data/afdb_uniprot_v4.fasta",
        required=False,
    )
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--pkl_outpath", help=".pkl outfile", required=True)
    parser.add_argument(
        "--reduction_factor",
        type=int,
        default=1,
        help="Factor by which to reduce the proportion of fasta searched, so 10 means only a tenth of the database will be searched",
    )
    args = parser.parse_args()
    main(args.sequence, args.foldcomp_fasta, args.pkl_outpath, args.reduction_factor)
