"""Small, auditable security primitives used by the research benchmark.

The secure-aggregation class models the pairwise-mask invariant used by
practical secure aggregation with X25519-derived pairwise secrets. Every pair
of clients contributes opposite masks, so the server can recover only the
sum. It is still a benchmark simulator, not a production protocol; production
deployment needs private keys to remain client-side, authenticated transport,
threshold secret sharing, and dropout recovery.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey


def _private_key(round_seed: int, client_id: int) -> X25519PrivateKey:
    # Deterministic private bytes make benchmark reruns reproducible. In a
    # deployment, replace this with fresh key material from the OS CSPRNG.
    digest = hashlib.sha256(f"x25519:{round_seed}:{client_id}".encode("utf-8")).digest()
    return X25519PrivateKey.from_private_bytes(digest)


@dataclass(frozen=True)
class SecureAggregationReport:
    participants: int
    vector_dim: int
    payload_bytes: int
    key_exchange_bytes: int
    total_bytes: int


class PairwiseSecureAggregator:
    """X25519 pairwise-mask secure-sum simulator for one synchronous round.

    The in-memory benchmark stores client keys centrally only to emulate all
    participants in one process. A deployment must keep private keys on the
    clients and add authenticated transport plus dropout recovery.
    """

    def __init__(self, participant_ids: list[int], vector_dim: int, round_seed: int):
        if len(set(participant_ids)) != len(participant_ids):
            raise ValueError("participant_ids must be unique")
        if vector_dim < 1:
            raise ValueError("vector_dim must be positive")
        self.participant_ids = tuple(sorted(int(value) for value in participant_ids))
        self.vector_dim = int(vector_dim)
        self.round_seed = int(round_seed)
        self._private_keys = {
            client_id: _private_key(self.round_seed, client_id)
            for client_id in self.participant_ids
        }
        self.public_keys = {
            client_id: self._private_keys[client_id].public_key().public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw,
            )
            for client_id in self.participant_ids
        }

    def _pair_rng(self, client_id: int, peer: int) -> np.random.Generator:
        left, right = sorted((int(client_id), int(peer)))
        peer_key = X25519PublicKey.from_public_bytes(self.public_keys[peer])
        shared_secret = self._private_keys[client_id].exchange(peer_key)
        digest = hmac.new(
            shared_secret,
            f"mask:{self.round_seed}:{left}:{right}".encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))

    def mask(self, client_id: int, vector: np.ndarray) -> np.ndarray:
        """Return a client payload whose pairwise masks cancel in the sum."""
        client_id = int(client_id)
        if client_id not in self.participant_ids:
            raise ValueError("unknown participant")
        value = np.asarray(vector, dtype=np.float64)
        if value.shape != (self.vector_dim,):
            raise ValueError("vector has the wrong dimension")
        masked = value.copy()
        for peer in self.participant_ids:
            if peer == client_id:
                continue
            left, _ = sorted((client_id, peer))
            rng = self._pair_rng(client_id, peer)
            pair_mask = rng.normal(0.0, 1.0, self.vector_dim)
            masked += pair_mask if client_id == left else -pair_mask
        return masked

    def aggregate(self, masked_vectors: dict[int, np.ndarray]) -> np.ndarray:
        """Recover the sum without inspecting any individual plaintext."""
        if set(masked_vectors) != set(self.participant_ids):
            raise ValueError("this simulator requires all participants to finish")
        return np.sum(
            [np.asarray(masked_vectors[client_id], dtype=np.float64)
             for client_id in self.participant_ids],
            axis=0,
        )

    def report(self) -> SecureAggregationReport:
        participants = len(self.participant_ids)
        payload_bytes = participants * self.vector_dim * 4
        # One 32-byte X25519 public key broadcast per participant.
        key_exchange_bytes = participants * 32
        return SecureAggregationReport(
            participants=participants,
            vector_dim=self.vector_dim,
            payload_bytes=payload_bytes,
            key_exchange_bytes=key_exchange_bytes,
            total_bytes=payload_bytes + key_exchange_bytes,
        )


def _mask_with_private_key(
    client_id: int,
    private_key: X25519PrivateKey,
    public_keys: dict[int, bytes],
    vector: np.ndarray,
    round_seed: int,
) -> np.ndarray:
    """Apply pairwise masks using only one client's private key."""
    value = np.asarray(vector, dtype=np.float64)
    masked = value.copy()
    for peer, public_bytes in sorted(public_keys.items()):
        if int(peer) == int(client_id):
            continue
        left, _ = sorted((int(client_id), int(peer)))
        shared_secret = private_key.exchange(X25519PublicKey.from_public_bytes(public_bytes))
        digest = hmac.new(
            shared_secret,
            f"mask:{int(round_seed)}:{left}:{int(peer) if left == int(client_id) else int(client_id)}".encode("utf-8"),
            hashlib.sha256,
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))
        pair_mask = rng.normal(0.0, 1.0, value.size)
        masked += pair_mask if int(client_id) == left else -pair_mask
    return masked


def _isolated_client_worker(
    connection,
    client_id: int,
    vector: np.ndarray,
    round_seed: int,
) -> None:
    """Worker holding one private key and one plaintext client update."""
    private_key = _private_key(round_seed, client_id)
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    connection.send((int(client_id), public_key))
    public_keys = connection.recv()
    masked = _mask_with_private_key(
        client_id, private_key, public_keys, np.asarray(vector), round_seed
    )
    connection.send(masked)
    connection.close()


def run_process_isolated_secure_round(
    vectors: dict[int, np.ndarray],
    round_seed: int,
) -> tuple[np.ndarray, SecureAggregationReport]:
    """Run one secure-sum round with one OS process per simulated client.

    This is still a local protocol harness, not a network deployment.  Its
    purpose is narrower and testable: private X25519 keys exist only inside
    client workers, while the coordinating process receives public keys and
    masked payloads before summing them.  Production still needs an
    authenticated transport, key rotation, threshold recovery, and a threat
    model covering compromised clients and servers.
    """
    if not vectors:
        raise ValueError("vectors must not be empty")
    participant_ids = sorted(int(client_id) for client_id in vectors)
    arrays = {client_id: np.asarray(vectors[client_id], dtype=np.float64) for client_id in participant_ids}
    vector_shape = arrays[participant_ids[0]].shape
    if len(vector_shape) != 1 or vector_shape[0] < 1:
        raise ValueError("vectors must be one-dimensional and non-empty")
    if any(array.shape != vector_shape for array in arrays.values()):
        raise ValueError("all vectors must have the same dimension")

    context = mp.get_context("spawn")
    workers = []
    connections = {}
    def receive(connection):
        if not connection.poll(60.0):
            raise TimeoutError("isolated secure-aggregation client timed out")
        return connection.recv()
    try:
        for client_id in participant_ids:
            parent, child = context.Pipe()
            process = context.Process(
                target=_isolated_client_worker,
                args=(child, client_id, arrays[client_id], int(round_seed)),
            )
            process.start()
            child.close()
            workers.append(process)
            connections[client_id] = parent
        public_keys = dict(receive(connections[client_id]) for client_id in participant_ids)
        for client_id in participant_ids:
            connections[client_id].send(public_keys)
        masked = {client_id: receive(connections[client_id]) for client_id in participant_ids}
        summed = np.sum([masked[client_id] for client_id in participant_ids], axis=0)
    finally:
        for connection in connections.values():
            connection.close()
        for process in workers:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
    report = SecureAggregationReport(
        participants=len(participant_ids),
        vector_dim=vector_shape[0],
        payload_bytes=len(participant_ids) * vector_shape[0] * 4,
        key_exchange_bytes=len(participant_ids) * 32,
        total_bytes=len(participant_ids) * vector_shape[0] * 4 + len(participant_ids) * 32,
    )
    return np.asarray(summed, dtype=np.float64), report


def clip_and_noise_vector(
    vector: np.ndarray,
    clip_norm: float,
    noise_multiplier: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Clip one client update and add Gaussian noise.

    The returned norm is useful for auditing.  This is a conservative
    client-level Gaussian mechanism; it deliberately does not claim privacy
    amplification from sampling or secure aggregation.
    """
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive")
    if noise_multiplier < 0:
        raise ValueError("noise_multiplier must be non-negative")
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    scale = min(1.0, clip_norm / max(norm, 1e-12))
    clipped = value * scale
    if noise_multiplier > 0:
        clipped = clipped + rng.normal(0.0, noise_multiplier * clip_norm, size=value.shape)
    return clipped.astype(np.float64), float(norm * scale)


def conservative_gaussian_epsilon(
    noise_multiplier: float,
    delta: float,
    steps: int,
) -> float:
    """Return a deliberately conservative sequential Gaussian bound."""
    if noise_multiplier <= 0:
        return math.inf
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if steps < 1:
        return 0.0
    per_step = math.sqrt(2.0 * math.log(1.25 / delta)) / noise_multiplier
    return float(steps * per_step)
