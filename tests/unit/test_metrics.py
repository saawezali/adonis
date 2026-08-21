"""Pure metric math: per-category P/R/F1, micro/macro, detection, faithfulness."""

from __future__ import annotations

from adonis.eval.metrics import (
    citation_faithfulness,
    detection_rates,
    macro_f1,
    micro_metrics,
    per_category_metrics,
)


def test_per_category_counts():
    # truth, prediction (None = never judged)
    predictions = [
        ("genuine_contradiction", "genuine_contradiction"),  # tp gc
        ("genuine_contradiction", "different_scope"),  # fn gc, fp ds
        ("genuine_contradiction", None),  # fn gc (missed)
        ("true_negative_unrelated", "genuine_contradiction"),  # fp gc; fn tn (was flagged)
        ("different_scope", "different_scope"),  # tp ds
        ("true_negative_near_dup", "not_conflicting"),  # tn tp (not flagged)
    ]
    results = per_category_metrics(predictions)
    by = {r.category: r for r in results}

    gc = by["genuine_contradiction"]
    assert (gc.tp, gc.fp, gc.fn) == (1, 1, 2)
    assert gc.precision == 0.5 and gc.recall == 1 / 3
    assert gc.f1 == 2 * 0.5 * (1 / 3) / (0.5 + 1 / 3)

    ds = by["different_scope"]
    assert (ds.tp, ds.fp, ds.fn) == (1, 1, 0)
    assert ds.precision == 0.5 and ds.recall == 1.0

    nd = by["true_negative_near_dup"]
    assert (nd.tp, nd.fp, nd.fn) == (1, 0, 0)
    assert nd.recall == 1.0 and nd.precision == 1.0

    un = by["true_negative_unrelated"]
    assert (un.tp, un.fp, un.fn) == (0, 0, 1)  # wrongly flagged
    assert un.recall == 0.0

    # Not_conflicting predictions must not produce counts anywhere else.
    assert all(r.tp == 0 and r.fp == 0 and r.fn == 0 for r in results if r.category in (
        "superseded_by_time", "ambiguous"))


def test_zero_precision_when_no_prediction():
    results = per_category_metrics([("ambiguous", None)])
    by = {r.category: r for r in results}
    assert by["ambiguous"].fn == 1
    assert by["ambiguous"].precision is None
    assert by["ambiguous"].recall == 0.0
    assert by["ambiguous"].f1 is None


def test_micro_and_macro():
    predictions = [
        ("genuine_contradiction", "genuine_contradiction"),
        ("genuine_contradiction", "different_scope"),
        ("different_scope", "different_scope"),
    ]
    results = per_category_metrics(predictions)
    micro = micro_metrics(results)
    # tp=2, fp=1, fn=1 -> p=2/3, r=2/3, f1=2/3
    assert micro["precision"] == micro["recall"] == micro["f1"] == 2 / 3

    # Macro excludes categories without any F1.
    assert macro_f1(results) is not None
    assert macro_f1(per_category_metrics([])) is None


def test_detection_rates():
    predictions = [
        ("genuine_contradiction", "genuine_contradiction"),  # hit
        ("genuine_contradiction", None),  # missed contradiction
        ("true_negative_unrelated", "genuine_contradiction"),  # false positive flag
        ("true_negative_near_dup", "not_conflicting"),  # correctly un-flagged
    ]
    recall, precision = detection_rates(predictions)
    assert recall == 0.5  # 1 of 2 labeled contradictions flagged
    assert precision == 0.5  # 1 of 2 flagged pairs is a real contradiction


def test_detection_rates_empty():
    assert detection_rates([]) == (None, None)


def test_citation_faithfulness():
    assert citation_faithfulness([1, 1, 0, 1]) == 0.75
    assert citation_faithfulness([]) is None
    assert citation_faithfulness([True, True]) == 1.0