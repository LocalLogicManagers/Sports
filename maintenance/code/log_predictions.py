"""Convert predictions.json into open-log entries ready to merge into the
persistent Project store (claude/predictions-open.json)."""
import json, re, os

DATA = "/home/claude/sportspredict/data"

def slug(name):
    return re.sub(r"[^a-z0-9]+", "", name.lower())

with open(os.path.join(DATA, "predictions.json")) as f:
    preds = json.load(f)

entries = []
for league, lg in preds["leagues"].items():
    for g in lg["games"]:
        eid = f"{league}-{g['date']}-{slug(g['away'])}-{slug(g['home'])}"
        entries.append({
            "id": eid,
            "league": league,
            "sport_label": lg["label"],
            "date": g["date"],
            "away": g["away"],
            "home": g["home"],
            "home_win_prob": g["home_win_prob"],
            "pick": g["pick"],
            "pick_prob": g["pick_prob"],
            "confidence": g["confidence"],
            "note": g.get("note"),
            "model_version": preds["model_version"],
            "logged_at": preds["generated_at"],
            "graded": False,
        })

with open(os.path.join(DATA, "new_open_entries.json"), "w") as f:
    json.dump(entries, f, indent=2)

print(f"{len(entries)} new open predictions ready to log")
for e in entries:
    print(" ", e["id"])
