"""
Grading script: takes the current open-log (from Project) and a dict of
{prediction_id: actual_winner} for games that have finished, moves graded
entries into history, and returns the still-open remainder.

Usage pattern (run each day):
  1. project_read predictions-open.json -> data/predictions-open.json
  2. project_read predictions-history.json -> data/predictions-history.json
  3. gather actual results for games whose date has passed (WebFetch)
  4. call grade_open(results_dict) here
  5. project_write both files back
"""
import json, sys, os
sys.path.insert(0, "/home/claude/sportspredict")
from engine import grade_prediction, summarize

DATA = "/home/claude/sportspredict/data"

def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)

def save(name, obj):
    with open(os.path.join(DATA, name), "w") as f:
        json.dump(obj, f, indent=2)

def grade_open(actual_results):
    """actual_results: {prediction_id: actual_winner_team_name}"""
    open_log = load("predictions-open.json")
    history = load("predictions-history.json")

    still_open = []
    newly_graded = []
    for pred in open_log:
        if pred["id"] in actual_results:
            g = grade_prediction(pred, actual_results[pred["id"]])
            pred.update(g)
            newly_graded.append(pred)
        else:
            still_open.append(pred)

    history["graded"].extend(newly_graded)
    save("predictions-open.json", still_open)
    save("predictions-history.json", history)

    print(f"graded {len(newly_graded)} predictions, {len(still_open)} remain open")
    if newly_graded:
        summary = summarize(history["graded"])
        print(json.dumps(summary, indent=2))
    return newly_graded, still_open, history

if __name__ == "__main__":
    # ---- self-test with synthetic data ----
    import shutil
    shutil.copy(os.path.join(DATA, "predictions-open.json"), "/tmp/open_backup.json") if os.path.exists(os.path.join(DATA, "predictions-open.json")) else None

    test_open = [
        {"id":"t1","league":"nfl","home":"A","away":"B","home_win_prob":0.7,"pick":"A","pick_prob":0.7,"confidence":"Moderate"},
        {"id":"t2","league":"nfl","home":"C","away":"D","home_win_prob":0.3,"pick":"D","pick_prob":0.7,"confidence":"Moderate"},
        {"id":"t3","league":"mlb","home":"E","away":"F","home_win_prob":0.55,"pick":"E","pick_prob":0.55,"confidence":"Lean"},
    ]
    save("predictions-open.json", test_open)
    save("predictions-history.json", {"graded": [], "rollups": {}})

    newly_graded, still_open, history = grade_open({"t1": "A", "t2": "C", "t3": "F"})
    assert len(newly_graded) == 3
    assert still_open == []
    correct_flags = {g["id"]: g["correct"] for g in newly_graded}
    assert correct_flags == {"t1": True, "t2": False, "t3": False}, correct_flags
    print("SELF-TEST PASSED")
