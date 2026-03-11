#!/usr/bin/env python3
"""bloom_join.py — Bloom filter join for distributed query optimization.

Uses Bloom filters to reduce data transfer in distributed joins by
pre-filtering rows that can't possibly match.

One file. Zero deps. Does one thing well.
"""

import hashlib
import math
import struct
import sys
from typing import Any


class BloomFilter:
    """Space-efficient probabilistic set membership."""

    def __init__(self, expected: int, fp_rate: float = 0.01):
        self.size = max(1, int(-expected * math.log(fp_rate) / (math.log(2) ** 2)))
        self.num_hashes = max(1, int(self.size / expected * math.log(2)))
        self.bits = bytearray((self.size + 7) // 8)
        self.count = 0

    def _hashes(self, item: str):
        h = hashlib.md5(item.encode()).digest()
        h1, h2 = struct.unpack('<QQ', h)
        for i in range(self.num_hashes):
            yield (h1 + i * h2) % self.size

    def add(self, item: str):
        for pos in self._hashes(item):
            self.bits[pos // 8] |= 1 << (pos % 8)
        self.count += 1

    def __contains__(self, item: str) -> bool:
        return all(self.bits[pos // 8] & (1 << (pos % 8)) for pos in self._hashes(item))


def bloom_join(left: list[dict], right: list[dict], key: str) -> list[dict]:
    """Join two relations using Bloom filter pre-filtering.

    1. Build Bloom filter from smaller relation's join keys
    2. Filter larger relation using Bloom filter (eliminates non-matching rows)
    3. Hash join the filtered result
    """
    # Determine build vs probe side
    if len(left) <= len(right):
        build, probe, build_key, probe_key = left, right, key, key
        swap = False
    else:
        build, probe, build_key, probe_key = right, left, key, key
        swap = True

    # Phase 1: Build Bloom filter from build side
    bf = BloomFilter(len(build), 0.01)
    for row in build:
        bf.add(str(row.get(build_key, '')))

    # Phase 2: Filter probe side
    filtered = [row for row in probe if str(row.get(probe_key, '')) in bf]

    # Phase 3: Hash join on filtered data
    index: dict[str, list[dict]] = {}
    for row in build:
        k = str(row.get(build_key, ''))
        index.setdefault(k, []).append(row)

    results = []
    for row in filtered:
        k = str(row.get(probe_key, ''))
        for match in index.get(k, []):
            merged = {**match, **row} if not swap else {**row, **match}
            results.append(merged)

    return results


def semi_join(local: list[dict], remote: list[dict], key: str) -> list[dict]:
    """Semi-join: return local rows that have matches in remote.

    Simulates distributed semi-join where only the Bloom filter
    is shipped across the network, not the full relation.
    """
    bf = BloomFilter(len(remote), 0.01)
    for row in remote:
        bf.add(str(row.get(key, '')))
    return [row for row in local if str(row.get(key, '')) in bf]


def demo():
    """Demonstrate Bloom join vs naive nested loop."""
    # Simulate two distributed tables
    orders = [{'order_id': i, 'customer_id': i % 100, 'amount': i * 10.5}
              for i in range(10000)]
    customers = [{'customer_id': i, 'name': f'Customer_{i}', 'city': f'City_{i % 50}'}
                 for i in range(100)]

    # Bloom join
    result = bloom_join(customers, orders, 'customer_id')
    print(f"Bloom Join: {len(customers)} customers × {len(orders)} orders")
    print(f"  Result rows: {len(result)}")
    print(f"  Sample: {result[0]}")

    # Semi-join (distributed optimization)
    matching = semi_join(orders, customers, 'customer_id')
    print(f"\nSemi-Join: {len(matching)} orders match customers")

    # Show Bloom filter stats
    bf = BloomFilter(len(customers))
    for c in customers:
        bf.add(str(c['customer_id']))
    print(f"\nBloom Filter stats:")
    print(f"  Items: {bf.count}")
    print(f"  Bits: {bf.size}")
    print(f"  Hashes: {bf.num_hashes}")
    print(f"  Size: {len(bf.bits)} bytes (vs {sum(len(str(c['customer_id'])) for c in customers)} bytes raw keys)")

    # False positive test
    fps = sum(1 for i in range(100, 200) if str(i) in bf)
    print(f"  False positives (100 non-members): {fps} ({fps}%)")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Quick self-test
        left = [{'id': 1, 'x': 'a'}, {'id': 2, 'x': 'b'}, {'id': 3, 'x': 'c'}]
        right = [{'id': 2, 'y': 'p'}, {'id': 3, 'y': 'q'}, {'id': 4, 'y': 'r'}]
        result = bloom_join(left, right, 'id')
        assert len(result) == 2, f"Expected 2 joins, got {len(result)}"
        ids = {r['id'] for r in result}
        assert ids == {2, 3}, f"Expected {{2, 3}}, got {ids}"

        semi = semi_join(left, right, 'id')
        assert len(semi) == 2
        print("All tests passed ✓")
    else:
        demo()
