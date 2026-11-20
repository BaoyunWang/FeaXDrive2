from collections import Counter
import pytest
from feaxdrive2.training.sampling import FourRoleSampler, ROLES, global_batch_quotas


def pools():
    return {role: list(range(i*100, i*100+40)) for i, role in enumerate(ROLES)}


def test_role_quotas_no_duplicates_and_determinism():
    source = pools()
    role_of = {index: role for role, values in source.items() for index in values}
    first, second = FourRoleSampler(source, steps=5), FourRoleSampler(source, steps=5)
    totals = Counter()
    for step in range(5):
        batch = first.next_batch()
        assert batch == second.next_batch()
        assert len(batch) == len(set(batch)) == 32
        assert Counter(role_of[i] for i in batch) == global_batch_quotas(step)
        totals.update(role_of[i] for i in batch)
    assert tuple(totals[role] for role in ROLES) == (40, 24, 40, 56)
    with pytest.raises(StopIteration):
        first.next_batch()


def test_overlapping_roles_are_rejected():
    source = pools()
    source[ROLES[1]][0] = source[ROLES[0]][0]
    with pytest.raises(ValueError, match="overlap"):
        FourRoleSampler(source)
