from kgcl_retro.chemistry.edits import generate_reaction_edits  # Explanation: imports packaged ground-truth edit extraction.


def test_generate_reaction_edits_adds_termination():  # Explanation: verifies edit extraction always appends the stop action.
    reaction = "[CH3:1][OH:2]>>[CH3:1][OH:2]"  # Explanation: defines a mapped identity reaction with no structural edits.
    data = generate_reaction_edits(reaction, kekulize=False)  # Explanation: extracts the edit sequence from the sample reaction.

    assert data is not None  # Explanation: checks that valid mapped molecules produce reaction data.
    assert data.edits[-1] == "Terminate"  # Explanation: checks that the final edit is the termination action.
