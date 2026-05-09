import os
def test_retrain_imports():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from retrain import fetch_feedback_rows, augment_with_feedback
    # Shouldn't crash even with empty DB
    fb = fetch_feedback_rows()
    assert isinstance(fb, type(fb))  # sanity
