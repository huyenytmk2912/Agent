from atlas.reasoning_data import generate_examples


def test_curriculum_is_reproducible_and_well_formed():
    a = generate_examples(8, seed=11)
    b = generate_examples(8, seed=11)
    assert a == b
    assert len(a) == 8
    assert all(item.target.startswith("Question:") for item in a)
    assert all("compute:" in item.target and "verify:" in item.target for item in a)
