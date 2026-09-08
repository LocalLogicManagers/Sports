import csv, math, json, os, sys, datetime
sys.path.insert(0, "/home/claude/sportspredict")
from engine import win_prob, shrink, make_pick, nfl_team_rating_v2, nfl_regime_change_sigma, attach_market

DATA = "/home/claude/sportspredict/data"
TODAY = datetime.date.today().isoformat()

def read_csv(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))

with open(os.path.join(DATA, "model_params.json")) as f:
    PARAMS = json.load(f)

def P(league):
    return PARAMS["leagues"][league]

def blend_weight_for(league):
    """Fitted market-blend weight for a league (0-1), or None if it hasn't
    been backtested yet -- see model_params.json's market_blend_weight."""
    return PARAMS.get("market_blend_weight", {}).get(league)

def build_game(league, away, home, date, rh, ra, note=None, sigma_override=None, model_version=None):
    p = P(league)
    sigma = sigma_override if sigma_override is not None else p["sigma"]
    home_prob = win_prob(rh, ra, p["home_adv"], sigma)
    g = {"away": away, "home": home, "date": date, "home_win_prob": round(home_prob, 3)}
    if note:
        g["note"] = note
    if model_version is not None:
        g["model_version"] = model_version
    g.update(make_pick(away, home, home_prob))
    return g

# NOTE on design: league blocks below fall into two shapes.
#   (a) "live" leagues (MLB/EPL/MLS/PLL fixtures) read their ratings/schedule
#       from data files (CSVs) that a refresh step regenerates from real
#       current sources each run -- this file itself never hardcodes a game
#       date or matchup for these.
#   (b) "point-in-time" leagues (NFL/NCAAF Week 1, PLL's regular-season
#       ratings, NBA/NCAAB/UCL off-season placeholders) are genuinely static
#       until their season progresses -- real, already-scheduled/final data,
#       not stand-ins for a live fetch -- so they stay as literals here on
#       purpose. Revisit NFL/NCAAF once Week 1 has been played.

results = {"generated_at": TODAY, "model_version": PARAMS["version"], "leagues": {}}

# ---------------- NFL (v2: win-total-baseline ratings + situational adjustments) ----------------
# v2 replaces v1's point-diff-carryover ratings, which backtested at Brier
# 0.263 on real 2024 games (worse than a coin flip) because they were never
# anchored to anything with real predictive content -- see "Team rating
# methodology v2" in sports-predictor-app.md for the full diagnosis. Baseline
# = each team's real 2026 sportsbook season win total run through
# nfl_win_total_baseline() (inside nfl_team_rating_v2); layered with injury/
# turnover adjustments where we have specific sourced data, and a sigma bump
# for any team in its first season under a new head coach. Every NFL game
# built here is tagged model_version 2 -- already-logged v1 picks (e.g. the
# locked Week 1 slate) are a separate, untouched record; this only affects
# what gets computed/logged from here on.

NFL_SEASON_WIN_TOTALS = {  # 2026 season win totals, sourced via WebSearch/WebFetch (FOX Sports/DraftKings-derived), 2026-09-07
    "Arizona Cardinals": 3.5, "Atlanta Falcons": 6.5, "Baltimore Ravens": 11.5,
    "Buffalo Bills": 10.5, "Carolina Panthers": 7.5, "Chicago Bears": 9.5,
    "Cincinnati Bengals": 10.5, "Cleveland Browns": 5.5, "Dallas Cowboys": 9.5,
    "Denver Broncos": 9.5, "Detroit Lions": 10.5, "Green Bay Packers": 9.5,
    "Houston Texans": 9.5, "Indianapolis Colts": 7.5, "Jacksonville Jaguars": 8.5,
    "Kansas City Chiefs": 10.5, "Las Vegas Raiders": 5.5, "Los Angeles Chargers": 9.5,
    "Los Angeles Rams": 11.5, "Miami Dolphins": 4.5, "Minnesota Vikings": 8.5,
    "New England Patriots": 10.5, "New Orleans Saints": 7.5, "New York Giants": 7.5,
    "New York Jets": 5.5, "Philadelphia Eagles": 10.5, "Pittsburgh Steelers": 8.5,
    "San Francisco 49ers": 9.5, "Seattle Seahawks": 10.5, "Tampa Bay Buccaneers": 8.5,
    "Tennessee Titans": 6.5, "Washington Commanders": 7.5,
}

NFL_NEW_HC_TEAMS = {  # first season under a new head coach, 2026 -- drives the regime-change sigma bump
    "Buffalo Bills", "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens",
    "Cleveland Browns", "Las Vegas Raiders", "Miami Dolphins", "New York Giants",
    "Pittsburgh Steelers", "Tennessee Titans",
}

# Injury/turnover adjustments where we have specific, sourced data (currently
# just the Ravens -- see ravens-data.json). Teams not listed here get 0 for
# both, an honest default: the win-total baseline already does nearly all the
# work per the Ravens diagnostic. Extend opportunistically as real injury/
# transaction news gets sourced for other teams.
NFL_INJURY_TURNOVER = {
    "Baltimore Ravens": {
        "injuries": [
            {"pos": "C", "severity": "out"},              # Danny Pinter, out extended time
            {"pos": "DT", "severity": "positive_return"},  # Nnamdi Madubuike, back to full-contact practice
        ],
        "departures": ["TE", "FB", "OT", "P", "S", "CB", "OLB", "WR"],  # Likely, Ricard, Faalele, Stout, Washington, Alexander, Oweh, C.Johnson
        "acquisitions": ["OLB", "S", "CB", "LB", "ILB"],                 # D.Jones, Gilman, White, Hendrickson, Barrett
    },
    # 2026-09-08: extended beyond the Ravens for the first time, sourced via WebSearch/WebFetch
    # (Bleacher Report injury roundup + NFL.com's Week 1 report). Injuries only -- no sourced
    # roster-turnover (departures/acquisitions) data for these teams yet, an honest default per the
    # same convention as every other team. Only covers teams playing in the currently-tracked Week 1
    # slate; extend opportunistically as news comes in.
    "Carolina Panthers": {
        "injuries": [
            {"pos": "OT", "severity": "out"},  # Ikem Ekwonu, ruptured patellar tendon, out most/all season
            {"pos": "OT", "severity": "out"},  # Taylor Moton, blood clot in lung, out extended time
        ],
    },
    "Detroit Lions": {
        "injuries": [
            {"pos": "S", "severity": "out"},          # Brian Branch, ruptured Achilles, targeting Dec return
            {"pos": "S", "severity": "questionable"},  # Kerby Joseph, nagging knee, recovery timeline unclear
        ],
    },
    "Green Bay Packers": {
        "injuries": [
            {"pos": "EDGE", "severity": "out"},  # Micah Parsons, torn ACL, targeting a playoff return
        ],
    },
    "Seattle Seahawks": {
        "injuries": [
            {"pos": "RB", "severity": "out"},  # Zach Charbonnet, torn ACL, timing unfavorable for Week 1
        ],
    },
    "San Francisco 49ers": {
        "injuries": [
            {"pos": "TE", "severity": "questionable"},  # George Kittle, Achilles, Week 1 return unlikely per reports but not ruled out
        ],
    },
}

nfl_ratings = {}
for team, win_total in NFL_SEASON_WIN_TOTALS.items():
    extra = NFL_INJURY_TURNOVER.get(team, {})
    nfl_ratings[team] = round(nfl_team_rating_v2(
        win_total,
        injuries=extra.get("injuries"),
        departures=extra.get("departures"),
        acquisitions=extra.get("acquisitions"),
    ), 3)

# 2026-09-08: was a hardcoded Week-1-only literal list; now reads from nfl_slate.csv
# (away,home,date -- same convention as mlb_slate.csv/epl_fixtures.csv/etc.) so adding Week 2+
# is a data update, not a code change. Currently seeded with the same 16 Week 1 games as before
# (byte-identical behavior) -- replace/append rows in nfl_slate.csv once Week 1 is played and a
# real Week 2 slate is sourced. log_predictions.py's append-only/no-duplicate logic still governs
# whether new rows here actually produce new logged picks.
nfl_week1 = [(r["away"], r["home"], r["date"]) for r in read_csv("nfl_slate.csv")]
# Real current market lines for the open Week 1 slate (FanDuel moneylines, sourced
# via WebSearch/WebFetch, 2026-08-31) -- keyed by (away, home). A game not in this
# dict just doesn't get a market_odds/value_bet/blended_prob block attached.
NFL_MARKET_ODDS = {
    ("New England Patriots", "Seattle Seahawks"): {"home_ml": "+164", "away_ml": "-196", "spread": "NE -3.5", "book": "FanDuel"},
    ("San Francisco 49ers", "Los Angeles Rams"): {"home_ml": "+168", "away_ml": "-200", "spread": "SF -3.5", "book": "FanDuel"},
    ("Chicago Bears", "Carolina Panthers"): {"home_ml": "+132", "away_ml": "-156", "spread": "CHI -2.5", "book": "FanDuel"},
    ("Tampa Bay Buccaneers", "Cincinnati Bengals"): {"home_ml": "+166", "away_ml": "-198", "spread": "TB -3.5", "book": "FanDuel"},
    ("New Orleans Saints", "Detroit Lions"): {"home_ml": "-360", "away_ml": "+290", "spread": "DET -7.0", "book": "FanDuel"},
    ("Buffalo Bills", "Houston Texans"): {"home_ml": "-102", "away_ml": "-116", "spread": "BUF -1.5", "book": "FanDuel"},
    ("Baltimore Ravens", "Indianapolis Colts"): {"home_ml": "+164", "away_ml": "-196", "spread": "BAL -3.5", "book": "FanDuel"},
    ("Cleveland Browns", "Jacksonville Jaguars"): {"home_ml": "-420", "away_ml": "+330", "spread": "JAX -7.5", "book": "FanDuel"},
    ("Atlanta Falcons", "Pittsburgh Steelers"): {"home_ml": "-168", "away_ml": "+142", "spread": "PIT -3.0", "book": "FanDuel"},
    ("New York Jets", "Tennessee Titans"): {"home_ml": "-138", "away_ml": "+118", "spread": "TEN -2.5", "book": "FanDuel"},
    ("Arizona Cardinals", "Los Angeles Chargers"): {"home_ml": "-650", "away_ml": "+480", "spread": "LAC -10.5", "book": "FanDuel"},
    ("Miami Dolphins", "Las Vegas Raiders"): {"home_ml": "+162", "away_ml": "-194", "spread": "MIA -3.5", "book": "FanDuel"},
    ("Green Bay Packers", "Minnesota Vikings"): {"home_ml": "-118", "away_ml": "+100", "spread": "MIN -1.5", "book": "FanDuel"},
    ("Washington Commanders", "Philadelphia Eagles"): {"home_ml": "-240", "away_ml": "+198", "spread": "PHI -5.5", "book": "FanDuel"},
    ("Dallas Cowboys", "New York Giants"): {"home_ml": "+122", "away_ml": "-144", "spread": "DAL -2.5", "book": "FanDuel"},
    ("Denver Broncos", "Kansas City Chiefs"): {"home_ml": "-144", "away_ml": "+122", "spread": "KC -2.5", "book": "FanDuel"},
}

nfl_games = []
for a, h, d in nfl_week1:
    g = build_game(
        "nfl", a, h, d, nfl_ratings.get(h, 0), nfl_ratings.get(a, 0),
        sigma_override=nfl_regime_change_sigma(
            P("nfl")["sigma"],
            home_new_coach=h in NFL_NEW_HC_TEAMS,
            away_new_coach=a in NFL_NEW_HC_TEAMS,
        ),
        model_version=2,
    )
    market = NFL_MARKET_ODDS.get((a, h))
    if market:
        g = attach_market(g, market, blend_weight=blend_weight_for("nfl"))
    nfl_games.append(g)
results["leagues"]["nfl"] = {
    "label": "NFL",
    "status": "v2 ratings (win-total baseline + injury/turnover/regime-change adjustments) — Week 1 not yet played",
    "model_version": 2,
    "ratings": nfl_ratings, "games": nfl_games,
}

# ---------------- MLB (live: fresh standings + slate each refresh) ----------------
mlb_rows = read_csv("mlb_2026_standings.csv")
mlb_ratings = {}
for r in mlb_rows:
    w, l = int(r["w"]), int(r["l"])
    gp = w + l
    raw = int(r["run_diff"]) / gp
    mlb_ratings[r["team"]] = round(shrink(raw, gp, P("mlb")["shrink_k"]), 3)

mlb_slate = [(r["away"], r["home"], r["date"]) for r in read_csv("mlb_slate.csv")]
mlb_games = [build_game("mlb", a, h, d, mlb_ratings.get(h, 0), mlb_ratings.get(a, 0)) for a, h, d in mlb_slate]
results["leagues"]["mlb"] = {"label": "MLB", "status": "in season", "ratings": mlb_ratings, "games": mlb_games}

# ---------------- Premier League (live) ----------------
epl_rows = read_csv("epl_2026_27_standings.csv")
epl_ratings = {}
for r in epl_rows:
    gp = int(r["gp"])
    raw = int(r["gd"]) / gp if gp else 0
    epl_ratings[r["team"]] = round(shrink(raw, gp, P("epl")["shrink_k"]), 3)

# Real current market lines for specific EPL games where we have a clean 3-way
# consensus price (Covers.com) -- keyed by (away, home). Most EPL games won't be
# in this dict; it's populated opportunistically, not every fixture.
EPL_MARKET_ODDS = {
    ("Arsenal", "Aston Villa"): {"home_win_prob": 0.13, "draw_prob": 0.23, "away_win_prob": 0.65, "book": "Covers.com consensus"},
}

epl_fixtures = [(r["away"], r["home"], r["date"]) for r in read_csv("epl_fixtures.csv")]
epl_games = []
for a, h, d in epl_fixtures:
    g = build_game("epl", a, h, d, epl_ratings.get(h, 0), epl_ratings.get(a, 0))
    market = EPL_MARKET_ODDS.get((a, h))
    if market:
        g = attach_market(g, market, blend_weight=blend_weight_for("epl"))
    epl_games.append(g)
results["leagues"]["epl"] = {
    "label": "Premier League",
    "status": "in season — early-season ratings are heavily shrunk toward league average",
    "ratings": epl_ratings, "games": epl_games,
}

# ---------------- MLS (live) ----------------
mls_rows = read_csv("mls_2026_standings.csv")
mls_ratings = {}
for r in mls_rows:
    gp = int(r["gp"])
    raw = int(r["gd"]) / gp if gp else 0
    mls_ratings[r["team"]] = round(shrink(raw, gp, P("mls")["shrink_k"]), 3)

mls_fixtures = [(r["away"], r["home"], r["date"]) for r in read_csv("mls_fixtures.csv")]
mls_games = [build_game("mls", a, h, d, mls_ratings.get(h, 0), mls_ratings.get(a, 0)) for a, h, d in mls_fixtures]
results["leagues"]["mls"] = {"label": "MLS", "status": "in season", "ratings": mls_ratings, "games": mls_games}

# ---------------- PLL Lacrosse (regular-season ratings are final/static; fixtures are live) ----------------
pll_ratings = {
    "Philadelphia Waterdogs": 2.5, "Boston Cannons": 0.833, "Maryland Whipsnakes": -0.833,
    "New York Atlas": -2.5, "Utah Archers": 0.833, "California Redwoods": 0.0,
    "Denver Outlaws": 0.0, "Carolina Chaos": -0.833,
}
pll_fixtures = [(r["away"], r["home"], r["date"], r["note"]) for r in read_csv("pll_fixtures.csv")]
pll_games = [build_game("pll", a, h, d, pll_ratings.get(h, 0), pll_ratings.get(a, 0), note) for a, h, d, note in pll_fixtures]
results["leagues"]["pll"] = {
    "label": "PLL Lacrosse",
    "status": "playoffs (semifinals) — neutral site, no home advantage applied",
    "ratings": pll_ratings, "games": pll_games,
}

# ---------------- NCAA Football (point-in-time: real Week 1 schedule, AP Top 25 preseason poll) ----------------
ap25 = ["Ohio State","Oregon","Georgia","Notre Dame","Texas","Indiana","Miami","Texas A&M",
        "Ole Miss","Oklahoma","LSU","Texas Tech","Alabama","BYU","USC","Michigan","Washington",
        "Penn State","SMU","Tennessee","Utah","Iowa","Houston","Louisville","Missouri"]
ncaaf_ratings = {team: round(max(0, 26 - (i + 1)) * 2.4, 3) for i, team in enumerate(ap25)}

# Week 1 (2026-09-05) has been played and graded (all 4 favorites covered — see
# predictions-history.json). The Week-2+ file-driven pipeline (CSV-based
# standings/fixtures like MLB/EPL/MLS) hasn't been built yet -- flagged for
# Zak rather than guessing at a new hardcoded slate with no real ranking
# update methodology behind it. Leaving games empty (rather than re-showing
# the finished Week 1 slate as if it were still upcoming) until that's built.
ncaaf_games = []
results["leagues"]["ncaaf"] = {
    "label": "NCAA Football",
    "status": "Week 1 complete (all graded) — Week 2+ needs a file-driven pipeline (not yet built); scoped to AP Top 25 preseason poll",
    "ratings": ncaaf_ratings, "games": ncaaf_games,
}

# ---------------- NBA (offseason, ratings only, static until 2026-27 tips off) ----------------
nba_ratings = {"Oklahoma City Thunder": 5.61, "San Antonio Spurs": 5.122, "Detroit Pistons": 4.634, "Denver Nuggets": 3.171, "Los Angeles Lakers": 2.927, "Boston Celtics": 3.659, "New York Knicks": 2.927, "Houston Rockets": 2.683, "Cleveland Cavaliers": 2.683, "Minnesota Timberwolves": 1.951, "Toronto Raptors": 1.22, "Atlanta Hawks": 1.22, "Portland Trail Blazers": 0.244, "Los Angeles Clippers": 0.244, "Philadelphia 76ers": 0.976, "Orlando Magic": 0.976, "Phoenix Suns": 0.976, "Charlotte Hornets": 0.732, "Miami Heat": 0.488, "Golden State Warriors": -0.976, "Milwaukee Bucks": -2.195, "Chicago Bulls": -2.439, "New Orleans Pelicans": -3.659, "Dallas Mavericks": -3.659, "Memphis Grizzlies": -3.902, "Sacramento Kings": -4.634, "Utah Jazz": -4.634, "Indiana Pacers": -5.366, "Washington Wizards": -5.854, "Brooklyn Nets": -5.122}

results["leagues"]["nba"] = {
    "label": "NBA",
    "status": "offseason — 2026-27 season tips off in October; ratings below are final 2025-26 power ratings (unregressed)",
    "ratings": nba_ratings, "games": [],
}

# ---------------- NCAA Basketball / Champions League (placeholders) ----------------
results["leagues"]["ncaab"] = {
    "label": "NCAA Basketball", "status": "offseason — season tips off in November 2026",
    "ratings": {}, "games": [],
}
results["leagues"]["ucl"] = {
    "label": "Champions League", "status": "league phase has not started — kicks off mid-September 2026",
    "ratings": {}, "games": [],
}

with open(os.path.join(DATA, "predictions.json"), "w") as f:
    json.dump(results, f, indent=2)

for lk, lv in results["leagues"].items():
    print(lk, lv["label"], "games:", len(lv["games"]), "teams rated:", len(lv["ratings"]))
    for g in lv["games"]:
        print("  ", g["away"], "@", g["home"], "->", g["home_win_prob"], "pick:", g["pick"], g["confidence"])
