"""Per-user persona text + per-item descriptions for NLI scoring.

Personas include BOTH liked and disliked dimensions, mined from training interactions:
  - liked   : categories/keywords from items with y == 1
  - disliked: categories/keywords from items with y == 0

The premise/hypothesis design used in NLI scoring (see src.train.precompute_persona_nli):
    premise    = item description (factual)
    hypothesis = "A person who <persona clauses> would enjoy this item."

This ordering matches MNLI-style training data (factual premise, opinion hypothesis)
and gives the NLI model real entailment/contradiction axes to discriminate on.
"""
from __future__ import annotations

from collections import Counter
from typing import List

from src.data.cache import CacheBundle


def _top_k_exclusive(pos: Counter, neg: Counter, k: int) -> tuple[list[str], list[str]]:
    """Return (top-k pos items, top-k neg items) with any overlap removed from the LESS
    frequent side to avoid contradictory persona claims ("likes X AND dislikes X").
    """
    top_p = [x for x, _ in pos.most_common(k * 3)]
    top_n = [x for x, _ in neg.most_common(k * 3)]
    ps = set(top_p[:k])
    ns = set(top_n[:k])
    overlap = ps & ns
    # Drop overlapping items from the side where they're less frequent
    for item in overlap:
        if pos[item] >= neg[item]:
            top_n = [x for x in top_n if x != item]
        else:
            top_p = [x for x in top_p if x != item]
    return top_p[:k], top_n[:k]


def build_user_personas(
    splits: dict,
    cache: CacheBundle,
    top_cats: int = 3,
    top_kws: int = 5,
) -> List[str]:
    """Return a list of persona text strings, one per user index.

    Personas integrate likes AND dislikes so NLI has a contradiction axis.
    """
    n_users = splits["n_users"]
    personas: List[str] = []
    cats_per_item = cache.item_categories
    items_raw = cache.items_raw

    for u in range(n_users):
        pos_cat: Counter[str] = Counter()
        pos_kw: Counter[str] = Counter()
        neg_cat: Counter[str] = Counter()
        neg_kw: Counter[str] = Counter()

        for (iid, y, _) in splits["train"][u]:
            c = cats_per_item[iid]
            it = items_raw[iid]
            if y == 1:
                if c:
                    pos_cat[c] += 1
                for k in it.get("pos", []):
                    pos_kw[k] += 1
            else:
                if c:
                    neg_cat[c] += 1
                # Key insight: a user who rated an item LOW often disliked the item's
                # positive attributes (or simply the genre). Using the item's POS keywords
                # gives a "actively dissatisfied with" signal.
                for k in it.get("pos", []):
                    neg_kw[k] += 1

        like_cats, dislike_cats = _top_k_exclusive(pos_cat, neg_cat, top_cats)
        like_kws, dislike_kws = _top_k_exclusive(pos_kw, neg_kw, top_kws)

        clauses: list[str] = []
        if like_cats:
            clauses.append(f"prefers {', '.join(like_cats)}")
        if like_kws:
            clauses.append(f"actively seeks items that are {', '.join(like_kws)}")
        if dislike_cats:
            clauses.append(f"avoids {', '.join(dislike_cats)}")
        if dislike_kws:
            clauses.append(f"is dissatisfied when items are {', '.join(dislike_kws)}")

        if not clauses:
            personas.append("A general consumer with no strong preferences.")
        else:
            personas.append("A user who " + "; ".join(clauses) + ".")
    return personas


def build_item_descriptions(cache: CacheBundle) -> List[str]:
    """Short textual item description; serves as the NLI premise (factual)."""
    items = cache.items_raw
    cats = cache.item_categories
    titles = cache.item_titles
    descs: List[str] = []
    for i, it in enumerate(items):
        pos_kws = it.get("pos", [])[:5]
        name = titles[i] if titles[i] else "This item"
        cat = cats[i] if cats[i] else "general"
        if pos_kws:
            kw_str = ", ".join(pos_kws)
            d = f"{name} is a {cat} item characterized by being {kw_str}."
        else:
            d = f"{name} is a {cat} item."
        descs.append(d)
    return descs


def build_user_preference_hypotheses(personas: List[str]) -> List[str]:
    """Given a persona text (used as a premise describing the user), produce a
    hypothesis claim of the form 'This user would enjoy this item.' so we can
    flip the NLI direction at scoring time.

    NOT USED in the current design — we instead swap premise/hypothesis at the
    call site so that the item description is the premise (see precompute_persona_nli).
    Retained for experimentation.
    """
    return ["This user would enjoy this item." for _ in personas]
