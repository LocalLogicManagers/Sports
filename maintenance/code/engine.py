"""
Core prediction engine: rating math, pick/confidence logic, grading, and
self-calibration. Designed to be reusable both interactively and from the
daily scheduled task (which starts a fresh session each time, so all state
that must survive between runs lives in the Project docs, not here).
"""
import math

# ---------- probability model ----------

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def win_prob(rating_home, rating_away, home_adv, sigma, cap=0.97):
    """Capped at [1-cap, cap] -- no real matchup is a true lock, and letting the
    model claim >97% (e.g. a big rating gap against a replacement-level unranked
    opponent) produces absurd implied odds (-9900) that misrepresent confidence."""
    margin = (rating_home - rating_away) + home_adv
    p = norm_cdf(margin / sigma)
    return min(cap, max(1 - cap, p))

def shrink(raw, games, k):
    if games <= 0:
        return 0.0
    return raw * (games / (games + k))

# ---------- pick / confidence ----------

CONFIDENCE_BANDS = [
    (0.80, "Strong"),
    (0.65, "Moderate"),
    (0.55, "Lean"),
    (0.0,  "Toss-up"),
]

def confidence_tier(prob):
    """prob is the win probability of whichever side is favored (>=0.5)."""
    for threshold, label in CONFIDENCE_BANDS:
        if prob >= threshold:
            return label
    return "Toss-up"

def make_pick(away, home, home_win_prob):
    if home_win_prob >= 0.5:
        pick, pick_prob = home, home_win_prob
    else:
        pick, pick_prob = away, 1 - home_win_prob
    return {
        "pick": pick,
        "pick_prob": round(pick_prob, 3),
        "confidence": confidence_tier(pick_prob),
    }

# ---------- simulated sportsbook odds & wager math ----------
# We don't have a live odds feed for team sports (only horse racing has real
# market odds, via morning line). To let users size hypothetical wagers on our
# own picks, we simulate what a typical sportsbook line would look like for a
# given win probability: apply a standard vig/overround, then convert to
# American odds. This is clearly NOT a real market line -- it's a modeled
# approximation so "what would $50 on this pick have paid" is answerable.

DEFAULT_VIG = 0.045  # ~4.5% overround, typical for a mainstream US sportsbook moneyline

def fair_prob_to_vig_prob(prob, vig=DEFAULT_VIG):
    """Nudge a fair (no-vig) probability toward 0.5 just enough that this side's
    vig'd probability, plus a mirrored opposite side, sums to 1+vig. Simple
    symmetric-overround approximation."""
    p = min(max(prob, 0.01), 0.99)
    return min(0.99, p + vig * p * (1 - p) * 2)

def prob_to_american_odds(prob, vig=DEFAULT_VIG):
    """Win probability -> simulated American odds string, e.g. '-150' or '+130'."""
    p = fair_prob_to_vig_prob(prob, vig)
    if p >= 0.5:
        american = -100 * p / (1 - p)
    else:
        american = 100 * (1 - p) / p
    return f"{'+' if american > 0 else ''}{round(american)}"

def american_to_decimal(american_odds):
    a = float(str(american_odds).replace("+", ""))
    return 1 + (a / 100 if a > 0 else 100 / abs(a))

def wager_outcome(stake, american_odds, won):
    """Returns {'stake', 'profit', 'payout'} for a single wager. profit is
    negative (loses the stake) when won is False."""
    decimal_odds = american_to_decimal(american_odds)
    if won:
        payout = stake * decimal_odds
        return {"stake": stake, "payout": round(payout, 2), "profit": round(payout - stake, 2)}
    return {"stake": stake, "payout": 0.0, "profit": round(-stake, 2)}

def parlay_decimal_odds(american_odds_list):
    d = 1.0
    for o in american_odds_list:
        d *= american_to_decimal(o)
    return d

# ---------- grading ----------

def grade_prediction(pred, actual_winner):
    """Given a logged prediction dict and the actual winning team name,
    return grading fields to merge in."""
    correct = (pred["pick"] == actual_winner)
    # Brier score against the pick's implied probability for the actual outcome
    p_home = pred["home_win_prob"]
    outcome_home = 1.0 if actual_winner == pred["home"] else 0.0
    brier = (p_home - outcome_home) ** 2
    return {
        "graded": True,
        "actual_winner": actual_winner,
        "correct": correct,
        "brier": round(brier, 4),
    }

# ---------- accuracy summary ----------

def summarize(graded_predictions):
    """graded_predictions: list of dicts with at least league, correct, brier,
    home_win_prob, pick_prob, confidence, home, away, pick.
    Returns a nested summary dict."""
    if not graded_predictions:
        return {"n": 0}

    n = len(graded_predictions)
    correct = sum(1 for p in graded_predictions if p["correct"])
    brier_avg = sum(p["brier"] for p in graded_predictions) / n

    by_league = {}
    for p in graded_predictions:
        lg = p["league"]
        s = by_league.setdefault(lg, {"n": 0, "correct": 0, "brier_sum": 0.0})
        s["n"] += 1
        s["correct"] += 1 if p["correct"] else 0
        s["brier_sum"] += p["brier"]
    for lg, s in by_league.items():
        s["accuracy"] = round(s["correct"] / s["n"], 3)
        s["brier"] = round(s["brier_sum"] / s["n"], 4)
        del s["brier_sum"]

    by_confidence = {}
    for p in graded_predictions:
        c = p["confidence"]
        s = by_confidence.setdefault(c, {"n": 0, "correct": 0, "prob_sum": 0.0})
        s["n"] += 1
        s["correct"] += 1 if p["correct"] else 0
        s["prob_sum"] += p["pick_prob"]
    for c, s in by_confidence.items():
        s["accuracy"] = round(s["correct"] / s["n"], 3)
        s["avg_predicted_prob"] = round(s["prob_sum"] / s["n"], 3)
        del s["prob_sum"]

    home_correct = sum(1 for p in graded_predictions if p["pick"] == p["home"] and p["correct"])
    home_picks = sum(1 for p in graded_predictions if p["pick"] == p["home"])
    away_picks = n - home_picks
    away_correct = correct - home_correct

    fav_correct = sum(1 for p in graded_predictions if p["pick_prob"] >= 0.5 and p["correct"])
    # picks are always the favored side by construction, so "favorite accuracy" == overall accuracy;
    # instead track "large favorite" (>=65%) vs "close call" (<65%) as the meaningful split
    big_fav = [p for p in graded_predictions if p["pick_prob"] >= 0.65]
    close_call = [p for p in graded_predictions if p["pick_prob"] < 0.65]

    return {
        "n": n,
        "accuracy": round(correct / n, 3),
        "brier": round(brier_avg, 4),
        "by_league": by_league,
        "by_confidence": by_confidence,
        "home_vs_away_pick": {
            "home_picks": home_picks,
            "home_pick_accuracy": round(home_correct / home_picks, 3) if home_picks else None,
            "away_picks": away_picks,
            "away_pick_accuracy": round(away_correct / away_picks, 3) if away_picks else None,
        },
        "big_favorite_vs_close": {
            "big_favorite_n": len(big_fav),
            "big_favorite_accuracy": round(sum(1 for p in big_fav if p["correct"]) / len(big_fav), 3) if big_fav else None,
            "close_call_n": len(close_call),
            "close_call_accuracy": round(sum(1 for p in close_call if p["correct"]) / len(close_call), 3) if close_call else None,
        },
    }

# ---------- self-calibration ----------

# ---------- horse racing (market-odds model) ----------

def implied_prob_from_ml(odds_str):
    """'5-2' (fractional morning line, 5-to-2) -> raw implied probability (with overround)."""
    a, b = odds_str.split("-")
    decimal_odds = float(a) / float(b) + 1
    return 1 / decimal_odds

def field_probabilities(horses):
    """horses: list of {"horse":..., "ml_odds": "5-2", ...}. Returns same list with
    raw_prob and norm_prob (overround removed) added."""
    raw = {h["horse"]: implied_prob_from_ml(h["ml_odds"]) for h in horses}
    total = sum(raw.values())
    out = []
    for h in horses:
        r = raw[h["horse"]]
        out.append({**h, "raw_prob": round(r, 4), "norm_prob": round(r / total, 4)})
    return out, round(total, 4)

HORSE_CONFIDENCE_BANDS = [
    (3.0, "Strong favorite"),
    (2.0, "Moderate favorite"),
    (1.3, "Lean favorite"),
    (0.0, "Wide-open field"),
]

def horse_confidence(top_prob, field_size):
    fair_share = 1 / field_size
    ratio = top_prob / fair_share if fair_share else 0
    for threshold, label in HORSE_CONFIDENCE_BANDS:
        if ratio >= threshold:
            return label, round(ratio, 2)
    return "Wide-open field", round(ratio, 2)

def race_pick(horses):
    """horses: list with norm_prob already computed. Returns the pick dict."""
    top = max(horses, key=lambda h: h["norm_prob"])
    label, ratio = horse_confidence(top["norm_prob"], len(horses))
    return {
        "pick": top["horse"],
        "pick_prob": top["norm_prob"],
        "confidence": label,
        "fair_share_ratio": ratio,
    }

def summarize_horse(graded_races):
    """graded_races: list of dicts with pick, correct, pick_prob, confidence."""
    if not graded_races:
        return {"n": 0}
    n = len(graded_races)
    correct = sum(1 for r in graded_races if r["correct"])
    by_confidence = {}
    for r in graded_races:
        c = r["confidence"]
        s = by_confidence.setdefault(c, {"n": 0, "correct": 0})
        s["n"] += 1
        s["correct"] += 1 if r["correct"] else 0
    for c, s in by_confidence.items():
        s["accuracy"] = round(s["correct"] / s["n"], 3)
    return {"n": n, "accuracy": round(correct / n, 3), "by_confidence": by_confidence}


def fit_home_adv_sigma(graded_predictions, ratings_lookup, current_home_adv, current_sigma,
                        min_n=25, max_step_adv=0.5, max_step_sigma=1.0):
    """
    Lightweight recalibration: nudge home_adv and sigma toward values that reduce
    average Brier score on recent graded games, via coordinate-wise grid search
    in a bounded neighborhood of the current values. Returns (new_home_adv, new_sigma, meta).

    ratings_lookup(pred) -> (rating_home, rating_away) as used at prediction time.
    Requires min_n graded games for this league or no change is made (avoids
    overfitting to a tiny sample).
    """
    if len(graded_predictions) < min_n:
        return current_home_adv, current_sigma, {"changed": False, "reason": f"n<{min_n}"}

    def avg_brier(home_adv, sigma):
        total = 0.0
        for p in graded_predictions:
            rh, ra = ratings_lookup(p)
            prob = win_prob(rh, ra, home_adv, sigma)
            outcome = 1.0 if p["actual_winner"] == p["home"] else 0.0
            total += (prob - outcome) ** 2
        return total / len(graded_predictions)

    best_adv, best_sigma = current_home_adv, current_sigma
    best_score = avg_brier(current_home_adv, current_sigma)

    adv_grid = [current_home_adv + s for s in (-max_step_adv, -max_step_adv/2, 0, max_step_adv/2, max_step_adv)]
    sigma_grid = [current_sigma + s for s in (-max_step_sigma, -max_step_sigma/2, 0, max_step_sigma/2, max_step_sigma)]
    sigma_grid = [s for s in sigma_grid if s > 0.5]

    for adv in adv_grid:
        for sigma in sigma_grid:
            score = avg_brier(adv, sigma)
            if score < best_score:
                best_score = score
                best_adv, best_sigma = adv, sigma

    changed = (round(best_adv, 3) != round(current_home_adv, 3)) or (round(best_sigma, 3) != round(current_sigma, 3))
    return round(best_adv, 3), round(best_sigma, 3), {
        "changed": changed,
        "n": len(graded_predictions),
        "old_brier": round(avg_brier(current_home_adv, current_sigma), 4),
        "new_brier": round(best_score, 4),
    }


# ---------- NFL team rating v2 (win-total baseline + situational adjustments) ----------
# Replaces v1's point-diff-carryover NFL ratings, which backtested at Brier
# 0.263 on real 2024 games -- worse than a coin flip -- because they were never
# anchored to anything with real predictive content. See "Team rating
# methodology v2" in sports-predictor-app.md for the full diagnosis. NFL only
# so far; other leagues still use the plain shrink()-based carryover rating.

NFL_POSITION_WEIGHT = {
    "QB": 7.0,
    "CB": 2.5, "EDGE": 2.5, "OLB": 2.5,
    "WR": 2.0,
    "S": 1.5, "OT": 1.5, "C": 1.5, "DT": 1.5, "RB": 1.5,
    "TE": 1.0, "G": 1.0, "ILB": 1.0,
    "K": 0.3, "P": 0.2, "LS": 0.1,
}
NFL_DEFAULT_POSITION_WEIGHT = 1.0  # fallback for a position not in the table above

def nfl_win_total_baseline(season_win_total, avg_win_total=8.5, pts_per_win=2.5):
    """v2 baseline team rating from the sportsbook's full-season win total (not
    this game's own line -- avoids just copying the matchup's own market
    price). avg_win_total=8.5 = 'average team' in a 17-game season;
    pts_per_win=2.5 is an approximate NFL points-per-marginal-win conversion
    (commonly cited range ~2.2-2.7; revisit if a better-sourced constant turns
    up)."""
    return (season_win_total - avg_win_total) * pts_per_win

def nfl_injury_adjustment(injuries):
    """injuries: list of {"pos": "QB", "severity": "out"|"questionable"|"positive_return"}.
    Returns one additive points adjustment (negative = worse off). "out" =
    out extended time/season-ending (-1.0x weight), "questionable" =
    week-to-week (-0.4x weight), "positive_return" = notable return from
    injury (+0.3x weight)."""
    total = 0.0
    for inj in injuries:
        weight = NFL_POSITION_WEIGHT.get(inj["pos"], NFL_DEFAULT_POSITION_WEIGHT)
        sev = inj["severity"]
        if sev == "out":
            total -= weight * 1.0
        elif sev == "questionable":
            total -= weight * 0.4
        elif sev == "positive_return":
            total += weight * 0.3
        else:
            raise ValueError(f"unknown injury severity: {sev!r}")
    return total

def nfl_turnover_adjustment(departures, acquisitions, cap=3.0):
    """departures/acquisitions: list of position strings for real roster moves
    only (trades, cuts, notable free-agent signings -- not unproven draft
    picks). Net turnover, additive, capped at +/-cap points."""
    dep_weight = sum(NFL_POSITION_WEIGHT.get(p, NFL_DEFAULT_POSITION_WEIGHT) for p in departures)
    acq_weight = sum(NFL_POSITION_WEIGHT.get(p, NFL_DEFAULT_POSITION_WEIGHT) for p in acquisitions)
    adj = -(dep_weight - acq_weight) * 0.15
    return max(-cap, min(cap, adj))

def nfl_team_rating_v2(season_win_total, injuries=None, departures=None, acquisitions=None,
                        avg_win_total=8.5, pts_per_win=2.5):
    """Full v2 NFL team rating: win-total baseline + injury adjustment + net
    roster-turnover adjustment. injuries/departures/acquisitions each default
    to none (a team with no specific sourced situational data just gets the
    baseline, which does nearly all the work per the Ravens diagnostic)."""
    rating = nfl_win_total_baseline(season_win_total, avg_win_total, pts_per_win)
    rating += nfl_injury_adjustment(injuries or [])
    rating += nfl_turnover_adjustment(departures or [], acquisitions or [])
    return rating

def nfl_regime_change_sigma(base_sigma, home_new_coach=False, away_new_coach=False, multiplier=1.15):
    """Inflate sigma for a game involving a team in its first season under a
    new head coach -- more variance, less predictability. Uses the higher of
    the two teams' multipliers if both qualify (a documented simplification,
    not a rigorous variance decomposition)."""
    factor = multiplier if (home_new_coach or away_new_coach) else 1.0
    return base_sigma * factor


# ---------- market blending (real market vs. model) ----------
# Compares/combines a model probability with a REAL market probability (not
# the simulated wager-calculator odds above). Used once a real market line is
# available for a game (see NFL_MARKET_ODDS etc. in compute.py).

def devig_two_way(odds_a, odds_b):
    """American moneyline odds for both sides of a two-way market -> fair
    (no-vig) probabilities for each side, normalized so they sum to 1.
    odds_a/odds_b: American odds as strings or numbers, e.g. '-196', '+164' --
    order-independent (works whichever side is listed first)."""
    def implied(odds):
        a = float(str(odds).replace("+", ""))
        return -a / (-a + 100) if a < 0 else 100 / (a + 100)
    p_a, p_b = implied(odds_a), implied(odds_b)
    total = p_a + p_b
    return p_a / total, p_b / total

def market_value_read(model_prob, market_prob, large_disagreement_pts=0.15):
    """Compare a model probability to a real market probability (same side,
    e.g. both home-team win prob). Returns a plain-language 'read' plus the
    edge in probability points -- the same 'agree / lean / large disagreement'
    framing used on the Ravens page's value-bet block, generalized so it's not
    Ravens-specific. Thresholds: <0.02 = agree, >=large_disagreement_pts (default
    0.15) = large disagreement, otherwise a directional 'leans' read."""
    edge = round(model_prob - market_prob, 3)
    large = abs(edge) >= large_disagreement_pts
    if large:
        read = "Large disagreement with the market — more likely our simple model is missing context than a real edge over the book's pricing."
    elif abs(edge) < 0.02:
        read = "Model roughly agrees with the market."
    elif edge > 0:
        read = "Model leans slightly more toward the home side than the market does."
    else:
        read = "Model leans slightly more toward the away side than the market does."
    return {
        "market_implied_prob": round(market_prob, 3),
        "model_prob": round(model_prob, 3),
        "edge_pts": edge,
        "read": read,
        "large_disagreement": large,
    }

def blend_prob(model_prob, market_prob, market_weight):
    """Simple weighted average of a model probability and a real market
    probability (both same side, e.g. home-team win prob). market_weight is
    clamped to [0,1] -- 1.0 means 'trust the market entirely', 0.0 'ignore the
    market entirely'. Weight comes from model_params.json's
    market_blend_weight, fit by backtesting (see the runbook)."""
    w = min(1.0, max(0.0, market_weight))
    return w * market_prob + (1 - w) * model_prob

def attach_market(game, market_entry, blend_weight=None):
    """Merge real market data into a built game dict (non-mutating -- returns
    a new dict, game itself is untouched). market_entry shapes handled:
      - two-way moneyline: {'home_ml':..., 'away_ml':..., 'spread':..., 'book':...}
        -> devigged market_home_prob, a value_bet block, and (if blend_weight
        is given) a blended_prob block.
      - three-way (soccer): {'home_win_prob':..., 'draw_prob':..., 'away_win_prob':...,
        'book':...} -> market_home_prob = home_win_prob directly, value_bet
        flagged with a note that this is a 3-way-vs-binary approximation since
        our model doesn't price a draw. No blended_prob (not backtested for
        3-way leagues).
      - spread-only, no moneyline: {'spread':..., 'book':...} -> market_odds
        attached, no value_bet/blended_prob (no probability to devig).
    """
    out = dict(game)
    out["market_odds"] = dict(market_entry)

    if "home_ml" in market_entry and "away_ml" in market_entry:
        market_home_prob, _ = devig_two_way(market_entry["home_ml"], market_entry["away_ml"])
        out["value_bet"] = market_value_read(game["home_win_prob"], market_home_prob)
        if blend_weight is not None:
            blended_home = blend_prob(game["home_win_prob"], market_home_prob, blend_weight)
            pick = game["home"] if blended_home >= 0.5 else game["away"]
            pick_prob = blended_home if blended_home >= 0.5 else 1 - blended_home
            out["blended_prob"] = {
                "home_win_prob": round(blended_home, 3),
                "market_weight": blend_weight,
                "pick": pick,
                "pick_prob": round(pick_prob, 3),
            }
    elif "home_win_prob" in market_entry:
        market_home_prob = market_entry["home_win_prob"]
        vb = market_value_read(game["home_win_prob"], market_home_prob)
        vb["note"] = "Market prob is a 3-way (home/draw/away) line; our model does not yet price a draw outcome, so this comparison is approximate."
        out["value_bet"] = vb
        # no blended_prob here -- not backtested for 3-way/soccer leagues yet

    return out
