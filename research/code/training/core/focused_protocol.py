"""Fixed confirmation scope. Legacy fit horizons stay unchanged for reuse."""

from coco_baselines.storage import identity
from scale_plan import METHODS

CORE_SHA = "8a29854a50c94420686f9b904f7eaecd88f9324bcf2b016c1ce6ad4057aa0b9c"
STUDY_RUN = "bd3eb87f9fb5ffc0a41963b4e8154c881e32b96cb44b8c40e8730665a72fff32"
COMPONENT_PREFIX = "a2f215ed99c2"
GROUPS = 4096
WIDTH = 12
TOPK = 32
CAP = 0.60
WARM = 7640
ADDITIONAL = 3000
TOTAL = WARM + ADDITIONAL
SCREEN = [21, 22, 23]
SEEDS = [21, 22, 23, 24, 25, 26, 27]
# Missing non-Vanilla families run first, then Vanilla tuning. Every family
# retains its released and separately labelled paper-specified implementation.
BASELINE_ORDER = [
    "released_grassmannian",
    "released_group_lasso",
    "paper_grassmannian",
    "paper_group_lasso",
    "paper_vanilla",
    "released_vanilla",
]
ARMS = ["agf_g", *METHODS]


def recipe():
    return dict(
        groups=GROUPS,
        width=WIDTH,
        active_groups=TOPK,
        cap=CAP,
        warm_updates=WARM,
        additional_updates=ADDITIONAL,
        total_updates=TOTAL,
        seeds=SEEDS,
        screen_seeds=SCREEN,
        arms=ARMS,
        configuration_selection=False,
        additional_component_training=False,
        exchanges=False,
        pcgrad=False,
        rank_or_mdl_sweep=False,
        baseline_selection="Three-seed development MSE; Group Lasso activity tolerance 20%; retain and flag mismatches.",
        fixed_recipe_scope="Current paper recipe; not a claim of optimality. Selected after earlier exploratory work; do not use final holdout for further tuning.",
        seed_scope="Seeds21-23 are previously inspected development starts;24-27 are additional starts.",
    )


def expected_pairs():
    return {(s, k) for s in SEEDS for k in ARMS}


def validate_row(row):
    if (
        row["groups"],
        row["rank"],
        row["active_groups"],
        row["additional_updates"],
        row["total_training_updates"],
    ) != (GROUPS, WIDTH, TOPK, ADDITIONAL, TOTAL):
        raise ValueError("Endpoint is not the fixed4096/12D/Top32/10640-update recipe")
    if row["kind"] not in ARMS or row["seed"] not in SEEDS:
        raise ValueError("Unexpected model/seed")
    if row["kind"] == "agf_g" and row["cap"] != CAP:
        raise ValueError("GGF cap must remain0.60")
    if identity(row["targets"]) != row["targets_sha"]:
        raise ValueError("Readout identity changed")
    return row
