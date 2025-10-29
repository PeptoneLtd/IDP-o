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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".96"

from functools import partial
from os.path import exists
from time import time

import jax
import mdtraj as md
import numpy as np
from fasta_search_in_foldcomp_database import generate_fragments
from jax import numpy as jnp
from jax import random, vmap
from nerfax.reduce_utils import reconstruct_from_mdtraj as infer_and_insert_hydrogens
from nerfax.utils import build_mdtraj_top


def affine_alignment(geom, ref_geom):
    translation_geom = geom.mean(0)
    translation_ref_geom = ref_geom.mean(0)

    R = jnp.dot((geom - translation_geom).T, ref_geom - translation_ref_geom)
    U, S, Vt = jnp.linalg.svd(R, full_matrices=False)
    Vt = jnp.where(
        jnp.linalg.det(jnp.dot(U, Vt)) < 0.0,
        Vt.at[-1].multiply(-1),
        Vt,  # jnp.concatenate([Vt[:3], -Vt[3:]], axis=0)
    )
    rotation = jnp.dot(U, Vt)
    return rotation, translation_geom, translation_ref_geom


def align(mobile_pos, ref_pos, mobile_indices, ref_indices):
    l_bb, r_bb = ref_pos[ref_indices], mobile_pos[mobile_indices]
    rot, pre_trans, post_trans = affine_alignment(r_bb, l_bb)
    mobile_pos_aligned = jnp.einsum("ij,bj", rot, mobile_pos - pre_trans) + post_trans
    return mobile_pos_aligned


def check_interactions(xyz, bonds=None, cutoff=0.1):
    distances = jnp.sqrt(((xyz[:, None] - xyz[None, :]) ** 2).sum(-1))
    # violation mask
    mask = distances < cutoff
    # ignore bonds
    # Note bonds is arr of [N,2], where the second value is higher
    mask = mask.at[tuple(bonds.T)].set(False)
    # Ignore self distances, and only look at j>i
    mask = jnp.triu(mask, k=1)
    return mask


compute_rmsd = lambda a, b: ((a - b) ** 2).sum(-1).mean() ** 0.5
get_n_chunks = lambda m, chunk_size: m // chunk_size + (1 if ((m % chunk_size) != 0) else 0)


def pre_join_fragments(data, l, r, overlap=2):
    overlap = 2  # this has been explicitly assumed at various bits of code
    assert l[-overlap:] == r[:overlap], "l and r sequences dont overlap correctly"
    sequence = l + r[overlap:]
    d = build_mdtraj_top(sequence)
    top = md.Topology.from_dataframe(d)
    top.create_standard_bonds()
    bb_indices = np.flatnonzero(d.name.apply(lambda s: s in ("N", "CA", "C", "O")).values).reshape(-1, 4)

    bonds = top.to_dataframe()[1][..., :2].astype(int)
    (lpos, l_bb_indices, lprobs), (rpos, r_bb_indices, rprobs) = (
        (data[k]["coords"], data[k]["bb_indices"], data[k]["probs"]) for k in (l, r)
    )
    return (
        ((lpos, l_bb_indices, lprobs), (rpos, r_bb_indices, rprobs), bonds),
        sequence,
        bb_indices,
    )


def _join_fragments(random_indices, context, static_context):
    overlap = 2

    (l_random_indices, r_random_indices) = random_indices
    (lpos, l_bb_indices), (rpos, r_bb_indices), bonds = context
    lhs, rhs = static_context

    lpos, rpos = (pos[indices] for (pos, indices) in ((lpos, l_random_indices), (rpos, r_random_indices)))
    l_indices, r_indices = (
        bb_indices[indexing].ravel()[2:-2]  # (C, O)_i, (N, CA)_(iplus1), join on peptide bond
        for bb_indices, indexing in [
            (l_bb_indices, jnp.arange(-overlap, 0)),
            (r_bb_indices, jnp.arange(overlap)),
        ]
    )

    def align_and_validate(rpos, lpos, r_indices, l_indices):
        rpos_aligned = align(rpos, lpos, r_indices, l_indices)
        rmsd = compute_rmsd(rpos_aligned[r_indices], lpos[l_indices])
        pos = jnp.concatenate([lpos[:lhs], rpos_aligned[rhs:]])
        # violations mask
        mask = check_interactions(pos, bonds)
        no_clash_mask = ~mask.any()
        return pos, no_clash_mask, rmsd

    pos, no_clash_mask, rmsds = vmap(align_and_validate, in_axes=(0, 0, None, None))(rpos, lpos, r_indices, l_indices)
    mask = (rmsds < 0.06) & no_clash_mask  # 0.6 Angstrom
    return pos, mask


def get_probs(x):
    _, indices, counts = jnp.unique(x, return_counts=True, return_inverse=True, size=x.shape[0])
    probs = (1 / counts)[indices]
    return probs


def jit_chunked_vmap(f, args, chunk_size):
    lengths = jax.tree.flatten(jax.tree.map(lambda x: jnp.shape(x)[0], args))[0]
    assert len(set(lengths)) == 1, f"Not all inputs are same length: {lengths}"

    def _body(_, args):
        return (None, f(*args))

    _, outputs = jax.lax.scan(
        _body,
        init=None,
        xs=jax.tree.map(lambda x: x.reshape((-1, chunk_size) + x.shape[1:]), args),
    )
    return jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), outputs)


def join_fragments(data, l, r, key, n=10000):
    overlap = 2
    nres = len(l) + len(r) - overlap
    batch_size = lambda nres, target=10: target / (1e-5 * (nres**2))  #  target in GB
    chunk_size = int(batch_size(nres))
    n = get_n_chunks(n, chunk_size) * chunk_size

    (
        ((lpos, l_bb_indices, lprobs), (rpos, r_bb_indices, rprobs), bonds),
        sequence,
        bb_indices,
    ) = pre_join_fragments(data, l, r)

    redo = True
    attempts = 0
    while redo and attempts < 5:
        lkey, rkey = random.split(key)

        random_indices = tuple(
            random.choice(key, jnp.arange(len(pos)), shape=(n,), p=probs)
            for key, pos, probs in ((lkey, lpos, lprobs), (rkey, rpos, rprobs))
        )

        context = ((lpos, l_bb_indices), (rpos, r_bb_indices), bonds)
        context = jax.tree.map(jnp.array, context)

        f = partial(
            _join_fragments,
            context=context,
            static_context=tuple(map(int, (l_bb_indices[-overlap][3] + 1, r_bb_indices[0][3] + 1))),
        )
        args = (random_indices,)
        (pos_vmap, mask) = jit_chunked_vmap(f, args, chunk_size)

        if np.count_nonzero(mask) == 0:
            attempts += 1
            logger.warning(f"mask is all zero, retrying (attempt: {attempts})")
        else:
            redo = False
    if attempts == 5:
        logger.warning("giving up, setting random 1 in mask")
        mask = np.zeros_like(mask)
        mask[np.random.randint(len(mask))] = 1

    sources, pos = jax.tree.map(lambda x: x[mask], (random_indices, pos_vmap))
    return pos, sequence, bb_indices, sources


def build_ensemble(data, fragments, n_joins_to_attempt=500000):
    n_fragments = len(fragments)
    keys = random.split(random.key(0), n_fragments)
    key_index = 0

    # Split into heirarchical (power of two) subsets
    bits = np.unpackbits(np.array(n_fragments, dtype=np.dtype("uint32").newbyteorder(">"))[None].view("uint8"))[
        -16:
    ].astype(bool)
    segments = (2 ** np.arange(16)[::-1])[bits]
    fragment_subsets = np.split(
        np.array(fragments), segments[::-1][:-1].cumsum(),
    )  # the [::-1] means you do smaller joins

    final_fragments = []
    for fragment_subset in fragment_subsets:
        i = 0
        joined_fragments = [fragment_subset]
        for _ in range(int(np.floor(np.log2(len(joined_fragments[0]))))):
            joined_fragments.append([])
            for l, r in zip(joined_fragments[i][::2], joined_fragments[i][1::2], strict=False):
                (pos, seq, bb_indices, sources) = join_fragments(data, l, r, keys[key_index], n=n_joins_to_attempt)
                data[seq] = {
                    "coords": pos,
                    "bb_indices": bb_indices,
                    "source": sources,
                    "probs": jax.tree.map(jnp.multiply, *map(get_probs, sources)),
                }
                joined_fragments[i + 1].append(seq)
                key_index += 1
            i += 1

            # Memory cleanup
            if i > 1:
                for s in joined_fragments[i - 2]:
                    jax.tree.map(
                        lambda x: x.delete() if isinstance(x, jax.Array) else None,
                        data[s],
                    )
        final_fragments.append(joined_fragments[-1][-1])

    # Linearly combine the power of 2 subsets
    l = final_fragments[0]
    for r in final_fragments[1:]:
        (pos, seq, bb_indices, sources) = join_fragments(data, l, r, keys[key_index], n=n_joins_to_attempt)
        data[seq] = {
            "coords": pos,
            "bb_indices": bb_indices,
            "source": sources,
            "probs": jax.tree.map(jnp.multiply, *map(get_probs, sources)),
        }
        key_index += 1
        l = seq
    return seq


def sort_trajectory(trajectory):
    trajectory.superpose(trajectory, frame=0)

    # Calculate RMSD for all pairs relative to each frame
    rmsd_matrix = np.zeros((len(trajectory), len(trajectory)))
    for i in range(len(trajectory)):
        rmsd_matrix[i] = md.rmsd(trajectory, trajectory, frame=i)

    sorted_frames = [0]  # Start with the first frame
    visited = set(sorted_frames)

    # Sort frames by minimizing RMSD to the previous frame
    for _ in range(1, trajectory.n_frames):
        last_frame = sorted_frames[-1]
        distances = rmsd_matrix[last_frame]

        # Find the closest unvisited frame
        closest_frame = np.argmin([dist if i not in visited else np.inf for i, dist in enumerate(distances)])
        sorted_frames.append(closest_frame)
        visited.add(closest_frame)

    # Return the sorted trajectory
    return trajectory[sorted_frames]


def main(
    sequence,
    outpath,
    fragments_folder,
    joins_to_attempt_per_pairing,
    max_structures_in_ensemble,
    rmsd_sort=False,
):
    overlap, seq_len = 2, 6
    fragments = generate_fragments(sequence, overlap=overlap, seq_len=seq_len)

    backend = jax.lib.xla_bridge.get_backend()
    # Memory cleanup
    for buf in backend.live_buffers():
        buf.delete()

    if not exists(outpath):
        start = time()
        data = {}
        for fragment in fragments:
            t = md.load_hdf5(f"{fragments_folder}/{fragment}.h5")  # as f:
            d = t.top.to_dataframe()[0]
            bb_indexs = np.flatnonzero(d.name.apply(lambda s: s in ("N", "CA", "C", "O")).values).reshape(-1, 4)
            data[fragment] = {
                "coords": t.xyz,
                "bb_indices": bb_indexs,  # - coords.shape[-2]
                "probs": np.ones(t.xyz.shape[0]),
            }
        seq = build_ensemble(data, fragments, joins_to_attempt_per_pairing)

        t = md.Trajectory(
            data[seq]["coords"][:max_structures_in_ensemble],
            md.Topology.from_dataframe(build_mdtraj_top(seq)),
        )
        if rmsd_sort:
            logger.info("sorting trajectory according to RMSD matrix...")
            t = sort_trajectory(t)
        t = infer_and_insert_hydrogens(t)
        end = time()
        t.save(outpath)
        logger.info(f"{end - start:.0f} seconds for {len(seq)} residues")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merges fragment ensembles together")
    parser.add_argument("--sequence", required=True)
    parser.add_argument(
        "--fragments_folder",
        help="Path to the ensemble fragment .h5 files",
        required=True,
    )
    parser.add_argument("--outpath", help="Path to write ensemble to", required=True)

    parser.add_argument("--joins_to_attempt_per_pairing", type=int, default=500000)
    parser.add_argument("--max_structures_in_ensemble", type=int, default=500000)
    parser.add_argument("--rmsd_sort", action="store_true", default=False)
    args = parser.parse_args()

    main(
        args.sequence,
        args.outpath,
        args.fragments_folder,
        args.joins_to_attempt_per_pairing,
        args.max_structures_in_ensemble,
        args.rmsd_sort,
    )
