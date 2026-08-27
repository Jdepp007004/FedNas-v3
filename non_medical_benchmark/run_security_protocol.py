"""One-round process-isolated secure-aggregation protocol check."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .security import run_process_isolated_secure_round


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    vectors = {
        client_id: np.random.default_rng(args.seed + client_id).normal(size=args.dimension)
        for client_id in range(args.clients)
    }
    summed, report = run_process_isolated_secure_round(vectors, args.seed)
    expected = np.sum(list(vectors.values()), axis=0)
    print(json.dumps({
        "clients": args.clients,
        "dimension": args.dimension,
        "masks_cancel": bool(np.allclose(summed, expected, atol=1e-10)),
        "max_abs_error": float(np.max(np.abs(summed - expected))),
        "payload_bytes": report.payload_bytes,
        "key_exchange_bytes": report.key_exchange_bytes,
        "total_bytes": report.total_bytes,
        "private_keys_server_side": False,
    }, indent=2))


if __name__ == "__main__":
    main()
