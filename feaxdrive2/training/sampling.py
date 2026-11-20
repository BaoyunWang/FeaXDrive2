"""Source-derived role-deck sampling; no replacement within a global batch."""
import random
from typing import Iterable
ROLES = ("collision", "near_collision", "hard_safe", "normal")
STEPS = 4000

def global_batch_quotas(step):
    return dict(zip(ROLES, (8,4,8,12) if step % 5 == 4 else (8,5,8,11)))

def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value

def _pool_indices(indices: Iterable[int], name: str, quota: int) -> list[int]:
    try:
        values = list(indices)
    except TypeError as exc:
        raise ValueError(f'{name} must be an iterable of scene indices') from exc
    if any((type(index) is not int or index < 0 for index in values)):
        raise ValueError(f'{name} indices must be nonnegative integers')
    if len(set(values)) != len(values):
        raise ValueError(f'{name} indices must be unique')
    if len(values) < quota:
        raise ValueError(f'{name} needs at least {quota} indices for a unique batch')
    return sorted(values)

class _CoveragePool:

    def __init__(self, indices: list[int], seed: int, name: str) -> None:
        self.indices = indices
        self.rng = random.Random(f'E-Body-coverage-v1:{name}:{seed}')
        self.order: list[int] = []
        self.cursor = 0
        self.draws = 0
        self.seen: set[int] = set()

    def take(self, count: int) -> list[int]:
        batch = self.order[self.cursor:self.cursor + count]
        self.cursor += len(batch)
        missing = count - len(batch)
        if missing:
            self.order = self.indices.copy()
            self.rng.shuffle(self.order)
            if batch:
                excluded = set(batch)
                prefix = [index for index in self.order if index not in excluded][:missing]
                selected = set(prefix)
                self.order = prefix + [index for index in self.order if index not in selected]
            batch.extend(self.order[:missing])
            self.cursor = missing
        self.draws += count
        self.seen.update(batch)
        return batch

    def report(self) -> dict[str, int]:
        return {'pool_size': len(self.indices), 'draws': self.draws, 'unique': len(self.seen), 'repeats': self.draws - len(self.seen)}

class FourRoleSampler:
    """Global B32, exact 25/15/25/35 over five updates, equal-sized rank shards."""

    def __init__(self, pools, seed=0, steps=STEPS):
        if set(pools) != set(ROLES) or type(seed) is not int or seed < 0:
            raise ValueError('Expected four disjoint pools and nonnegative seed')
        self.steps = _positive_int(steps, 'steps')
        maximum = dict(zip(ROLES, (8, 5, 8, 12)))
        values = {k: _pool_indices(pools[k], k, maximum[k]) for k in ROLES}
        if len(set().union(*(set(v) for v in values.values()))) != sum(map(len, values.values())):
            raise ValueError('Four roles overlap')
        self.pools = {k: _CoveragePool(v, seed, 'collision-gt-' + k) for k, v in values.items()}
        self.emitted = 0

    def next_batch(self):
        if self.emitted == self.steps:
            raise StopIteration
        batch = [i for role, quota in global_batch_quotas(self.emitted).items() for i in self.pools[role].take(quota)]
        if self.emitted % 2:
            batch = [v for i in range(0, 32, 2) for v in (batch[i + 1], batch[i])]
        self.emitted += 1
        return batch
