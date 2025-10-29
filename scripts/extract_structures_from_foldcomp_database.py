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

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import pickle as pkl
import struct
from os.path import exists

import jax
import mdtraj as md
import numpy as np
from hirola import HashTable
from jax import jit, vmap
from jax import numpy as jnp
from nerfax.foldcomp_constants import AA_N_SC_ATOMS, BACKBONE_BOND_LENGTHS
from nerfax.foldcomp_utils import reconstruct_from_internal_coordinates_pure_sequential, reconstruct_sidechains
from nerfax.utils import build_mdtraj_top

ConvertIntToOneLetterCode = np.array(
    [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
        "B",
        "Z",
        "*",
        "X",
    ],
)
table = HashTable(len(ConvertIntToOneLetterCode), "U1", almost_full=None)
table.add(ConvertIntToOneLetterCode)


def load_backbone_data(fcz, n):
    d = np.frombuffer(fcz.read(8 * n), dtype=np.uint8).reshape(-1, 8).astype(np.uint16)
    aas = (d[:, 0] & 0xF8) >> 3
    # overwrite first three parts of d
    d[:, 0] = ((d[:, 0] & 0x0007) << 8) | (d[:, 1] & 0x00FF)  # omega
    d[:, 1] = ((d[:, 2] & 0x00FF) << 4) | (d[:, 3] & 0x00FF) >> 4  # psi
    d[:, 2] = ((d[:, 3] & 0x000F) << 8) | (d[:, 4] & 0x00FF)  # phi
    # ca_c_n_angle, c_n_ca_angle, n_ca_c_angle = d[:,-3:].T
    # 0-omega, 1-psi, 2-phi, 5-ca_c_n_angle, 6-c_n_ca_angle, 7-n_ca_c_angle
    # mainChainAnglesTorsionsDicretizers is (phi,psi,omega, n_ca_c_angle, ca_c_n_angle, c_n_ca_angle), we reorder to match this
    mainChainAnglesTorsions = d[:, [2, 1, 0, 7, 5, 6]]
    return aas, mainChainAnglesTorsions


sequence_to_aa_index = np.vectorize(list(ConvertIntToOneLetterCode).index)


def _load_backbone_reconstruction_data(fcz, byte_start, l, r):
    ### get metadata
    fcz.seek(byte_start)
    assert struct.unpack("@4s", fcz.read(4))[0] == b"FCMP", "Start tag doesnt match to b'FCMP'"
    fcz.seek(byte_start + 4)
    nResidue = struct.unpack("H", fcz.read(2))[0]
    fcz.seek(byte_start + 12)
    nAnchor = struct.unpack("B", fcz.read(1))[0]
    fcz.seek(byte_start + 24)
    lenTitle = struct.unpack("I", fcz.read(4))[0]
    mainChainAnglesTorsionsDicretizers = np.frombuffer(fcz.read(48), dtype=np.float32).reshape(2, 6)

    ### get backbone data
    start_byte_bb_data = byte_start + 89 + 40 * nAnchor + lenTitle  # 28+48+4*nAnchor+lenTitle+36*nAnchor+1+12
    # angle+torsion data is stored in previous residue data [so will be offset by 1 in aas]
    # Note: the first 8 bytes are just dummy values and are not actually used.
    fcz.seek(start_byte_bb_data + (l - 1) * 8)
    _, mainChainAnglesTorsions = load_backbone_data(fcz, r - l)

    start_byte_sidechain_data = start_byte_bb_data + 8 * nResidue
    return mainChainAnglesTorsionsDicretizers, mainChainAnglesTorsions, start_byte_sidechain_data


def load_backbone_reconstruction_data(fcz, byte_start, l, r):
    mainChainAnglesTorsionsDicretizers, mainChainAnglesTorsions, start_byte_sidechain_data = (
        _load_backbone_reconstruction_data(fcz, byte_start, l, r)
    )
    return mainChainAnglesTorsionsDicretizers, mainChainAnglesTorsions


prefix_cumsum = lambda x: np.pad(np.cumsum(x), (1, 0))


def load_reconstruction_data(fcz, byte_start, l, r, sequence):
    mainChainAnglesTorsionsDicretizers, mainChainAnglesTorsions, start_byte_sidechain_data = (
        _load_backbone_reconstruction_data(fcz, byte_start, l, r)
    )

    ### get sidechain data
    # aas_from_start = sequence_to_aa_index(list(sequence[:r]))
    aas_from_start = table.get(list(sequence[:r]))
    aas = aas_from_start[l:r]

    n_atoms_per_sc = AA_N_SC_ATOMS[aas_from_start]
    l_byte_sc_offset, r_byte_sc_offset = prefix_cumsum(n_atoms_per_sc)[[l, r]]
    sc_byte_size = r_byte_sc_offset - l_byte_sc_offset

    fcz.seek(start_byte_sidechain_data + l_byte_sc_offset)

    sideChainAnglesDiscretized = np.frombuffer(fcz.read(sc_byte_size), dtype=np.uint8)

    return mainChainAnglesTorsionsDicretizers, mainChainAnglesTorsions, aas, sideChainAnglesDiscretized


def reconstruct_backbone(mainChainAnglesTorsionsDicretizers, mainChainAnglesTorsions):
    mins, conf_fs = mainChainAnglesTorsionsDicretizers  # swap so ordering matches

    def process(mainChainAnglesTorsions):
        # (phi,psi,omega, n_ca_c_angle, ca_c_n_angle, c_n_ca_angle) ordering in axis=1 [phi,psi,omega, n_ca_c_angle, ca_c_n_angle, c_n_ca_angle = mainChainAnglesTorsions_cont.T]
        mainChainAnglesTorsions_cont = (mainChainAnglesTorsions * conf_fs + mins) * (jnp.pi / 180)  # convert to radians

        # reorder so we place (anchors)-N->CA->C->... so N first
        # hence we need (CA-C-N, psi, length C-N) then (C-N-CA, omega, length N-CA) then (N-CA-C,phi,CA-C) on repeat
        torsions = mainChainAnglesTorsions_cont[..., [1, 2, 0]]
        angles = mainChainAnglesTorsions_cont[..., [4, 5, 3]]
        angles = (
            jnp.pi - angles
        )  # due to historical scnet reasons, the scnet angle is defined as pi-angle. It seems they've used scnet defn here

        angles, torsions = [x.reshape((x.shape[0], -1)) for x in (angles, torsions)]
        return angles, torsions

    angles, torsions = process(mainChainAnglesTorsions)
    lengths = jnp.broadcast_to(jnp.tile(BACKBONE_BOND_LENGTHS, angles.shape[-1] // 3), angles.shape)

    ref_pos = jnp.array([[-1, -1, 0], [-1, 0, 0], [0, 0, 0]], dtype=float)
    bb_pos = reconstruct_from_internal_coordinates_pure_sequential(
        *map(lambda x: x.reshape(-1), (lengths, angles, torsions)),
        init=ref_pos,
    )
    return bb_pos


def reconstruct(mainChainAnglesTorsionsDicretizers, mainChainAnglesTorsions, aas, sideChainAnglesDiscretized):
    bb_pos = reconstruct_backbone(mainChainAnglesTorsionsDicretizers, mainChainAnglesTorsions)
    pos = reconstruct_sidechains(aas, sideChainAnglesDiscretized, bb_pos)
    return pos


def load_from_foldcomp(fcz, byte_start, l, r, sequence, reconstruct_jit_fn=None):
    mainChainAnglesTorsionsDicretizers, mainChainAnglesTorsions, aas, sideChainAnglesDiscretized = (
        load_reconstruction_data(fcz, byte_start, l, r, sequence)
    )
    if reconstruct_jit_fn is None:
        pos = reconstruct(mainChainAnglesTorsionsDicretizers, mainChainAnglesTorsions, aas, sideChainAnglesDiscretized)
    else:
        # specialised function, precompiled for aas in question, can be ~2000x faster
        # built by build_reconstruct_fn
        pos = reconstruct_jit_fn(
            mainChainAnglesTorsionsDicretizers,
            mainChainAnglesTorsions,
            sideChainAnglesDiscretized,
        )
    return pos


def build_reconstruct_fn(sequence, reconstruct_fn=reconstruct):
    aas = np.vectorize(list(ConvertIntToOneLetterCode).index)(list(sequence))

    @jit
    def _f(mainChainAnglesTorsionsDicretizers, mainChainAnglesTorsions, sideChainAnglesDiscretized):
        with jax.ensure_compile_time_eval():
            return reconstruct_fn(
                mainChainAnglesTorsionsDicretizers,
                mainChainAnglesTorsions,
                aas,
                sideChainAnglesDiscretized,
            )

    return _f


def build_parallel_reconstruct_fn(sequence):
    return build_reconstruct_fn(sequence, reconstruct_fn=vmap(reconstruct, in_axes=(0, 0, None, 0)))


def join_list_of_identical_pytrees(x, joiner_fn=np.concatenate):
    vals = jax.tree.flatten(x)[0]
    first_val, pytree = jax.tree.flatten(x[0])
    nvals = len(first_val)
    x = jax.tree.unflatten(pytree, [joiner_fn(vals[i::nvals]) for i in range(nvals)])
    return x


def extract_data(fcz, fasta_f, fragment_data, nres, exclude_cis_omega=False):
    data = []
    for fasta_byte_start, byte_start, l in zip(*fragment_data, strict=False):
        assert l >= 0 and l < 10**5, f"Start bytes {(fasta_byte_start, byte_start, l)} is not valid"
        fasta_f.seek(fasta_byte_start - l)
        sequence = fasta_f.read(l + nres)
        angles_torsions_discretizers, angles_torsions, aas, sideChainAnglesDiscretized = load_reconstruction_data(
            fcz,
            byte_start,
            l,
            l + nres,
            sequence,
        )
        if exclude_cis_omega:
            omegas = angles_torsions[:, 2] * angles_torsions_discretizers[1, 2] + angles_torsions_discretizers[0, 2]
            if np.all(np.abs(omegas) > 90):  # these should relax to trans omega state
                data.append((angles_torsions_discretizers, angles_torsions, sideChainAnglesDiscretized))
        else:
            data.append((angles_torsions_discretizers, angles_torsions, sideChainAnglesDiscretized))
    data = join_list_of_identical_pytrees(data, joiner_fn=np.stack)
    return data


def compute_ensembles(
    fragments_data,
    foldcomp_fasta="/data/afdb_uniprot_v4.fasta",
    foldcomp_db="/data/afdb_uniprot_v4",
    outfolder="/work/backbone_forcefield/scratch/117/",
    n_max_structures_per_fragment=1000,
    exclude_cis_omega=False,
):
    for fragment, fragment_data in fragments_data.items():
        if not exists(f"{outfolder}/{fragment}.h5"):
            nres = len(fragment)

            ## Subset number of fragments to reasonable number
            nstructures = len(fragment_data[0])
            if nstructures > n_max_structures_per_fragment:
                indexs = np.random.uniform(size=(nstructures,)).argsort()[:n_max_structures_per_fragment]
                fragment_data = jax.tree.map(lambda x: x[indexs], fragment_data)

            ## Extract info from database
            with open(foldcomp_fasta) as fasta_f, open(foldcomp_db, "rb") as fcz:
                structure_data = extract_data(fcz, fasta_f, fragment_data, nres, exclude_cis_omega)

            ## Reconstruct
            jit_fn = build_parallel_reconstruct_fn(fragment)
            pos = jit_fn(*structure_data)

            ## Save
            t = md.Trajectory(pos / 10.0, topology=md.Topology.from_dataframe(build_mdtraj_top(fragment)))
            t.save_hdf5(f"{outfolder}/{fragment}.h5")


def main(foldcomp_fasta, foldcomp_db, outfolder, n_max_structures_per_fragment, byte_starts_path, exclude_cis_omega):
    os.makedirs(outfolder, exist_ok=True)
    data = pkl.load(open(byte_starts_path, "rb"))
    compute_ensembles(data, foldcomp_fasta, foldcomp_db, outfolder, n_max_structures_per_fragment, exclude_cis_omega)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Constructs ensembles of the fragments hit from their start data")
    parser.add_argument(
        "--byte_starts_path",
        help="Path to hit start data in both fasta and foldcomp database files",
        required=True,
        default="example/byte_starts.pkl",
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
    )
    parser.add_argument(
        "--foldcomp_db",
        help="Path to the foldcomp database file",
        default="/data/afdb_uniprot_v4",
    )
    parser.add_argument("--outfolder", help="Folder to write ensemble h5 files to", required=True)
    parser.add_argument(
        "--n_max_structures_per_fragment",
        type=int,
        default=1000,
        help="Max number of structures per fragment to extract",
    )
    parser.add_argument(
        "--exclude_cis_omega",
        default=False,
        action="store_true",
        help="Whether to exclude structures with cis omega angles",
    )
    args = parser.parse_args()

    main(
        args.foldcomp_fasta,
        args.foldcomp_db,
        args.outfolder,
        args.n_max_structures_per_fragment,
        args.byte_starts_path,
        args.exclude_cis_omega,
    )
