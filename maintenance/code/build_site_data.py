import json, sys, os
sys.path.insert(0, "/home/claude/sportspredict")
from engine import summarize, summarize_horse

DATA = "/home/claude/sportspredict/data"

def load(name, default=None):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)

predictions = load("predictions.json")
horses = load("horse_predictions.json", {"races": []})
history = load("predictions-history.json", {"graded": [], "rollups": {}})

team_sport_graded = [g for g in history["graded"] if g.get("league") != "horse_racing"]
horse_graded = [g for g in history["graded"] if g.get("league") == "horse_racing"]

accuracy = {
    "team_sports": summarize(team_sport_graded),
    "horse_racing": summarize_horse(horse_graded),
    "tracking_since": "2026-08-29",
    # trimmed raw graded logs so the page can run a live bankroll simulator
    # (any stake/currency) client-side without a server round trip
    "graded_log": [
        {"date": g["date"], "league": g["league"], "matchup": f"{g['away']} @ {g['home']}",
         "pick": g["pick"], "pick_prob": g["pick_prob"], "correct": g["correct"]}
        for g in team_sport_graded
    ],
    "graded_log_horse": [
        {"date": g["date"], "race": g.get("race", g.get("id","")),
         "pick": g["pick"], "pick_prob": g["pick_prob"], "correct": g["correct"]}
        for g in horse_graded
    ],
}

site = {
    "generated_at": predictions["generated_at"],
    "model_version": predictions["model_version"],
    "leagues": predictions["leagues"],
    "horse_racing": horses,
    "accuracy": accuracy,
}

with open(os.path.join(DATA, "site_data.json"), "w") as f:
    json.dump(site, f, indent=2)

print("site_data.json built:", len(json.dumps(site)), "bytes")
