from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 40304030

N_GROUPS = 100
OBSERVATION_YEAR = 2026

OUTPUT_DIR = Path("data")
PUBLIC_FILE = OUTPUT_DIR / "kpop_groups.csv"
ANSWER_KEY_FILE = OUTPUT_DIR / "kpop_groups_answer_key.csv"


# =============================================================================
# NAMING
# =============================================================================

FIRM_NAMES = {
    "A": "Stellar Entertainment",
    "B": "Nova Wave Labs",
    "C": "Apex Factory Media",
    "D": "Pure Tone Collective",
    "E": "Horizon Global Music",
    "F": "Pulse Digital Studios",
    "G": "Ember Rising Records",
}

BOY_GROUP_NAMES = [
    "ARCLIGHT", "BLACKTOP", "CINDERBLOCK", "DUSKFALL", "EMBERLINE",
    "FAULTLINE", "GRIDLOCK", "HAVOC", "IRONCLAW", "JETSTREAM",
    "KILLSWITCH", "LOCKDOWN", "MAGMA", "NITRO", "OBELISK",
    "PAVEMENT", "QUARRY", "RAZORLINE", "STEELDRUM", "THUNDERHEAD",
    "UNDERTONE", "VANDAL", "WARPATH", "XENITH", "YEARZERO",
    "ZEROHOUR", "ANVIL", "BULWARK", "CROSSWIRE", "DEADBOLT",

    "EASTGATE", "FIREWALL", "GRAVITON", "HARDWIRE", "IRONSIDE",
    "JACKKNIFE", "KNUCKLE", "LOADBEAR", "MACHINERY", "NIGHTWATCH",
    "OVERTAKE", "PITSTOP", "QUICKSAND", "REDLINE", "SLIPSTREAM",
    "TREADMARK", "UPROOT", "VOLLEY", "WIREFRAME", "XEROX",

    "YARDSTICK", "ZODIAC", "AFTERBURN", "BLACKOUT", "COUNTERPUNCH",
    "DOWNFORCE", "ENGINEROOM", "FLASHPOINT", "GROUNDZERO", "HEADLOCK",
    "IRONWORK", "JUGGERNAUT", "KEROSENE", "LONGSHOT", "MOTORCADE",
    "NIGHTFALL", "OUTBURST", "POWERLINE", "QUICKDRAW", "RIPTIDE",

    "SHORELINE", "TIDEBREAK", "UNDERTOW", "VOLTAGE", "WATERLINE",
    "XENOLITH", "YELLOWLINE", "ZIGZAG", "ALLOY", "BEDROCK",
    "CARBIDE", "DRILLBIT", "ELEMENT", "FOUNDRY", "GRANITE",
    "HOTWIRE", "IGNITION", "JUNCTION", "KEYSTONE", "LODESTONE",

    "MOONSHOT", "NORTHGATE", "OUTCROP", "PINNACLE", "QUICKSILVER",
    "ROCKSLIDE", "SANDBLAST", "TOUCHSTONE", "UPLINK", "VERTIGO",
    "WILDFIRE", "XENON", "YIELD", "ZENITH", "AFTERSHOCK",
    "BLACKSMITH", "CROSSCUT", "DEEPWATER", "EDGELINE", "FIRESTORM",

    "GOLDRUSH", "HEADWIND", "IRONHORSE", "JETBLACK", "KILOWATT",
    "LANDSLIDE", "MAGNETITE", "NIGHTSOIL", "OILSLICK", "POWERGRID",
    "QUARTZITE", "RIVERBED", "SALTFLAT", "TIMBERLINE", "UNDERGROUND",
    "VANGUARD", "WINDCHILL", "XRAY", "YEARBOOK", "ZEPPELIN",
]

GIRL_GROUP_NAMES = [
    "PETAL", "BLOOM", "SILK", "RIBBON", "LACE",
    "SATIN", "CHIFFON", "TULLE", "SEQUIN", "BROOCH",
    "LOCKET", "TRINKET", "CHARM", "AMULET", "TALISMAN",
    "ORACLE", "SIBYL", "MYTH", "LEGEND", "LORE",
    "SONNET", "HAIKU", "LIMERICK", "BALLAD", "LYRIC",

    "WHISPER", "MURMUR", "HUSH", "SIGH", "BREATH",
    "SPARKLE", "TWINKLE", "SHIMMER", "GLEAM", "FLICKER",
    "CANDLE", "LANTERN", "BEACON", "TORCH", "HEARTH",
    "MEADOW", "GLADE", "GROVE", "ORCHARD", "GARDEN",
    "RIVER", "BROOK", "CREEK", "LAGOON", "HARBOR",

    "SEASHELL", "DRIFTWOOD", "PEBBLE", "SANDGRAIN", "SEAFOAM",
    "CLOUDLET", "MIST", "FOG", "DEW", "FROST",
    "SNOWFLAKE", "ICICLE", "HAILSTONE", "RAINDROP", "PUDDLE",
    "RAINBOW", "PRISM LIGHT", "SUNFLARE", "MOONBEAM", "STARGLOW",
    "COMET", "NEBULA", "GALAXY", "COSMOS", "ASTRAL",

    "VESPER", "AURORA", "ZEPHYR", "SOLSTICE", "EQUINOX",
    "HALCYON", "ELEGY", "ODE", "HYMN", "CAROL",
    "LULLABY", "SERENADE", "NOCTURNE", "PRELUDE", "INTERLUDE",
    "FUGUE", "ARIETTA", "CANZONE", "MADRIGAL", "RONDO",

    "PASTEL", "CRAYON", "CHALK", "INKWELL", "PARCHMENT",
    "STATIONERY", "WAX SEAL", "RIBBON BOW", "BUTTERFLY", "DRAGONFLY",
    "LADYBUG", "FIREFLY", "HUMMINGBIRD", "SWALLOW", "SPARROW",
    "DOVE", "FAWN", "KITTEN", "BUNNY", "LAMB",

    "TOFFEE", "CARAMEL", "HONEYCOMB", "VANILLA", "COCOA",
    "CINNAMON", "NUTMEG", "SAFFRON", "PAPRIKA", "GINGER",
    "LAVENDER", "ROSEMARY", "SAGE", "THYME", "BASIL",
    "MINT LEAF", "LEMONGRASS", "CHAMOMILE", "HIBISCUS", "ORCHID",

    "TIARA", "CROWN", "DIADEM", "SCEPTER", "THRONE",
    "CASTLE", "PALACE", "MANOR", "COTTAGE", "VILLA",
    "BALCONY", "TERRACE", "PATIO", "GAZEBO", "FOUNTAIN",
    "LANTERN FEST", "CARNIVAL", "FIREWORKS", "CONFETTI", "PARADE",
]


# =============================================================================
# FIRM CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class FirmConfig:
    tier: int
    resource_base: float
    consistency_bias: float
    debut_share: float


FIRMS: Dict[str, FirmConfig] = {
    "A": FirmConfig(5, 0.90, 0.80, 0.15),
    "B": FirmConfig(3, 0.60, 0.40, 0.15),
    "C": FirmConfig(2, 0.30, 0.20, 0.18),
    "D": FirmConfig(3, 0.50, 0.70, 0.10),
    "E": FirmConfig(4, 0.70, 0.50, 0.15),
    "F": FirmConfig(3, 0.50, 0.30, 0.15),
    "G": FirmConfig(2, 0.40, 0.20, 0.12),
}


# =============================================================================
# ARCHETYPES
#
# Latent dimensions:
#
# 0 = commercial_strength
# 1 = digital_momentum
# 2 = fandom_strength
# 3 = instability
# 4 = prestige
# =============================================================================

ARCHETYPES = {
    1: {
        "mean": np.array([2.0, 0.0, 0.0, -1.0, 0.5]),
        "std": np.array([0.9, 0.6, 0.5, 0.5, 0.7]),
    },
    2: {
        "mean": np.array([0.5, 2.0, 0.0, 0.0, 0.8]),
        "std": np.array([0.8, 0.9, 0.7, 0.7, 0.8]),
    },
    3: {
        "mean": np.array([0.0, 0.0, 2.0, -0.5, 0.0]),
        "std": np.array([0.5, 0.5, 0.7, 0.5, 0.5]),
    },
    4: {
        "mean": np.array([-0.5, 0.0, -0.5, 2.0, -0.5]),
        "std": np.array([0.8, 0.7, 0.7, 0.9, 0.7]),
    },
    5: {
        "mean": np.array([0.8, 0.5, 0.0, 0.0, 2.0]),
        "std": np.array([0.7, 0.7, 0.6, 0.6, 0.8]),
    },
}


FIRM_ARCHETYPE_PROBS = {
    "A": [0.50, 0.10, 0.05, 0.05, 0.30],
    "B": [0.05, 0.30, 0.10, 0.10, 0.45],
    "C": [0.00, 0.10, 0.05, 0.70, 0.15],
    "D": [0.05, 0.05, 0.60, 0.05, 0.25],
    "E": [0.10, 0.25, 0.15, 0.10, 0.40],
    "F": [0.00, 0.40, 0.05, 0.20, 0.35],
    "G": [0.00, 0.30, 0.05, 0.30, 0.35],
}


FIRM_GENDER_PROB = {
    "A": 0.50,
    "B": 0.50,
    "C": 0.60,
    "D": 0.40,
    "E": 0.50,
    "F": 0.60,
    "G": 0.50,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def clip_probability(x: float) -> float:
    return float(np.clip(x, 0.01, 0.99))


def positive_noise(
    rng: np.random.Generator,
    mean: float,
    sigma: float,
) -> float:
    """
    Log-normal multiplicative noise.
    """
    return float(np.exp(rng.normal(mean, sigma)))


def sample_archetype(
    archetype: int,
    rng: np.random.Generator,
) -> np.ndarray:
    params = ARCHETYPES[archetype]

    return rng.normal(
        loc=params["mean"],
        scale=params["std"],
    )


# =============================================================================
# FIRM / ARCHETYPE GENERATION
# =============================================================================

def sample_firms(
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:

    firms = list(FIRMS.keys())

    # Normalize configured population shares.
    probabilities = np.array(
        [FIRMS[f].debut_share for f in firms],
        dtype=float,
    )

    probabilities /= probabilities.sum()

    return rng.choice(
        firms,
        size=n,
        p=probabilities,
    )


def sample_archetypes(
    firms: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:

    archetypes = np.empty(len(firms), dtype=int)

    for i, firm in enumerate(firms):
        archetypes[i] = rng.choice(
            np.arange(1, 6),
            p=FIRM_ARCHETYPE_PROBS[firm],
        )

    return archetypes


def sample_genders(
    firms: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:

    return np.array([
        rng.binomial(1, FIRM_GENDER_PROB[firm])
        for firm in firms
    ])


# =============================================================================
# NAMES
# =============================================================================

def assign_names(
    genders: np.ndarray,
    rng: np.random.Generator,
) -> List[str]:

    boys = BOY_GROUP_NAMES.copy()
    girls = GIRL_GROUP_NAMES.copy()

    rng.shuffle(boys)
    rng.shuffle(girls)

    boy_counter = 0
    girl_counter = 0

    names = []

    for gender in genders:

        if gender == 0:
            if boy_counter >= len(boys):
                raise ValueError(
                    "Not enough unique boy-group names."
                )

            names.append(boys[boy_counter])
            boy_counter += 1

        else:
            if girl_counter >= len(girls):
                raise ValueError(
                    "Not enough unique girl-group names."
                )

            names.append(girls[girl_counter])
            girl_counter += 1

    return names


# =============================================================================
# GROUP-LEVEL DEMOGRAPHICS
# =============================================================================

def generate_demographics(
    firm: str,
    gender: int,
    latent: np.ndarray,
    rng: np.random.Generator,
):

    config = FIRMS[firm]

    # More commercially successful groups tend to have longer preparation.
    training_months = np.clip(
        24
        + 8 * latent[0]
        + 5 * latent[2]
        - 5 * latent[3]
        + rng.normal(0, 5),
        6,
        84,
    )

    # Group size is correlated weakly with archetype and firm.
    member_lambda = np.clip(
        5.5
        + 0.5 * latent[1]
        + 0.3 * latent[2]
        + 0.2 * (config.tier - 3),
        3.5,
        8.5,
    )

    member_count = int(
        np.clip(
            rng.poisson(member_lambda),
            3,
            12,
        )
    )

    debut_age = np.clip(
        20
        + 1.2 * latent[3]
        - 0.8 * latent[4]
        + rng.normal(0, 1.3),
        16,
        28,
    )

    return member_count, debut_age, training_months


# =============================================================================
# FEATURE GENERATION
# =============================================================================

def latent_to_features(
    latent: np.ndarray,
    firm: str,
    gender: int,
    debut_year: int,
    years_active: int,
    rng: np.random.Generator,
) -> Dict:

    cfg = FIRMS[firm]

    commercial = latent[0]
    digital = latent[1]
    fandom = latent[2]
    instability = latent[3]
    prestige = latent[4]

    # -------------------------------------------------------------------------
    # Commercial performance
    # -------------------------------------------------------------------------

    streaming = (
        positive_noise(
            rng,
            mean=3.2
            + 0.55 * commercial
            + 0.40 * digital
            + 0.15 * cfg.resource_base,
            sigma=0.45,
        )
        * (1 + 0.025 * years_active)
    )

    album_sales = positive_noise(
        rng,
        mean=2.0
        + 0.55 * commercial
        + 0.20 * prestige
        + 0.20 * cfg.resource_base,
        sigma=0.35,
    )

    mv_views = positive_noise(
        rng,
        mean=4.0
        + 0.55 * digital
        + 0.25 * commercial,
        sigma=0.45,
    )

    # Higher latent commercial strength => better chart rank.
    chart_peak = np.clip(
        95
        - 25 * commercial
        - 10 * digital
        - 8 * prestige
        + rng.normal(0, 15),
        1,
        200,
    )

    # -------------------------------------------------------------------------
    # Industry recognition
    # -------------------------------------------------------------------------

    award_nominations = np.clip(
        2
        + 2.2 * commercial
        + 1.5 * prestige
        + 1.2 * cfg.tier
        + rng.normal(0, 2.5),
        0,
        30,
    )

    music_show_wins = np.clip(
        1
        + 1.8 * commercial
        + 1.0 * digital
        + rng.normal(0, 1.5),
        0,
        25,
    )

    # -------------------------------------------------------------------------
    # Artistic / performance characteristics
    # -------------------------------------------------------------------------

    vocal_stability = np.clip(
        sigmoid(
            0.5
            + 0.35 * commercial
            + 0.20 * prestige
            - 0.30 * instability
            + rng.normal(0, 0.3)
        ),
        0,
        1,
    )

    choreography = np.clip(
        50
        + 16 * digital
        + 8 * commercial
        + rng.normal(0, 10),
        0,
        100,
    )

    concept_consistency = np.clip(
        cfg.consistency_bias
        + 0.12 * prestige
        + 0.10 * commercial
        - 0.15 * instability
        + rng.normal(0, 0.08),
        0,
        1,
    )

    # -------------------------------------------------------------------------
    # Fandom / digital ecosystem
    # -------------------------------------------------------------------------

    social_growth = np.clip(
        5
        + 4 * digital
        + 2 * fandom
        + rng.normal(0, 2),
        -10,
        40,
    )

    fan_cafe = positive_noise(
        rng,
        mean=3.2
        + 0.65 * fandom
        + 0.20 * commercial,
        sigma=0.45,
    )

    fan_labor = positive_noise(
        rng,
        mean=1.8
        + 0.70 * fandom
        + 0.15 * digital,
        sigma=0.40,
    )

    merch_index = np.clip(
        40
        + 18 * fandom
        + 10 * commercial
        + rng.normal(0, 10),
        0,
        100,
    )

    viral_moments = np.clip(
        rng.poisson(
            np.clip(
                2
                + 2 * digital
                + 0.8 * fandom
                + 0.5 * prestige,
                0.2,
                15,
            )
        ),
        0,
        30,
    )

    # -------------------------------------------------------------------------
    # Business / company
    # -------------------------------------------------------------------------

    brand_deals = np.clip(
        1
        + 1.5 * commercial
        + 0.8 * prestige
        + 3 * cfg.tier / 5
        + rng.normal(0, 1),
        0,
        20,
    )

    resource_allocation = np.clip(
        cfg.resource_base
        + 0.08 * commercial
        - 0.08 * instability
        + rng.normal(0, 0.05),
        0,
        1,
    )

    # -------------------------------------------------------------------------
    # International / sentiment
    # -------------------------------------------------------------------------

    intl_ratio = np.clip(
        0.25
        + 0.15 * digital
        + 0.18 * fandom
        + 0.08 * (cfg.tier >= 4)
        + rng.normal(0, 0.08),
        0,
        1,
    )

    sentiment = np.clip(
        0.15
        + 0.20 * commercial
        + 0.15 * fandom
        - 0.25 * instability
        + rng.normal(0, 0.15),
        -1,
        1,
    )

    # -------------------------------------------------------------------------
    # Organizational stability
    # -------------------------------------------------------------------------

    turnover_probability = sigmoid(
        -2.2
        + 1.0 * instability
        - 0.4 * commercial
        + 0.5 * (cfg.tier <= 2)
    )

    turnover_rate = np.clip(
        turnover_probability
        + rng.normal(0, 0.04),
        0,
        0.60,
    )

    conflict = np.clip(
        sigmoid(
            -1.5
            + 1.2 * instability
            - 0.4 * commercial
            + 0.4 * (cfg.tier <= 2)
            + rng.normal(0, 0.25)
        ),
        0,
        1,
    )

    renewal_probability = clip_probability(
        sigmoid(
            1.0
            + 0.7 * commercial
            + 0.4 * prestige
            - 1.0 * instability
        )
    )

    contract_renewed = int(
        rng.random() < renewal_probability
    )

    comeback_freq = np.clip(
        1.5
        + 0.6 * commercial
        - 0.5 * instability
        + 0.25 * cfg.resource_base
        + rng.normal(0, 0.35),
        0.3,
        5,
    )

    # -------------------------------------------------------------------------
    # Demographics
    # -------------------------------------------------------------------------

    member_count, debut_age, training_months = generate_demographics(
        firm,
        gender,
        latent,
        rng,
    )

    # -------------------------------------------------------------------------
    # Return PUBLIC features only.
    # -------------------------------------------------------------------------

    return {
        "streaming_log": np.log1p(streaming),
        "chart_peak_inv_log": np.log1p(201 - chart_peak),
        "comeback_freq": comeback_freq,
        "mv_views_log": np.log1p(mv_views),
        "album_sales_log": np.log1p(album_sales),

        "award_nominations": award_nominations,
        "music_show_wins": music_show_wins,

        "vocal_stability": vocal_stability,
        "choreo_complexity": choreography,
        "concept_consistency": concept_consistency,

        "social_growth_pct": social_growth,
        "fan_cafe_log": np.log1p(fan_cafe),
        "merch_index": merch_index,
        "brand_deals": brand_deals,

        "intl_fan_ratio": intl_ratio,
        "sentiment_polarity": sentiment,
        "viral_moments": viral_moments,
        "fan_labor_log": np.log1p(fan_labor),

        "member_count": member_count,
        "debut_age_avg": debut_age,
        "training_months_avg": training_months,

        "turnover_rate": turnover_rate,
        "contract_renewed": contract_renewed,
        "internal_conflict": conflict,

        "company_tier": cfg.tier,
        "debut_year": debut_year,
        "years_active": years_active,

        "gender": gender,

        "resource_allocation": resource_allocation,
        "market_saturation": np.clip(
            0.50
            + 0.15 * instability
            - 0.08 * cfg.tier
            + rng.normal(0, 0.08),
            0,
            1,
        ),
    }

# =============================================================================
# DATASET INTRODUCTION
# =============================================================================

INTRO_FILE = OUTPUT_DIR / "data_introduction.md"


def generate_data_introduction(
    public_df: pd.DataFrame,
    answer_df: pd.DataFrame,
) -> None:
    """
    Generate a student-friendly overview of the synthetic dataset.

    The introduction describes:
    - dataset size
    - parent companies
    - groups belonging to each company
    - boy/girl group breakdown
    - public variables
    - hidden variables
    - parameter meanings
    - logarithmic variables
    """

    lines = []

    lines.append("# K-pop Group Synthetic Dataset\n")

    lines.append(
        "## Overview\n"
    )

    lines.append(
        f"This dataset contains **{len(public_df)} fictional K-pop groups** "
        f"generated from a simulated data-generation process. "
        "Each group is assigned to a fictional parent company, given a "
        "gender category, debut year, demographic characteristics, "
        "commercial performance, fandom characteristics, organizational "
        "characteristics, and other simulated attributes. "
        "The dataset is designed for learning data analysis, statistics, "
        "machine learning, anomaly detection, and exploratory data analysis.\n"
    )

    lines.append(
        "The data is **synthetic**. The companies, groups, measurements, "
        "and relationships were generated by code and should not be "
        "interpreted as real-world information about actual K-pop groups.\n"
    )

    # -------------------------------------------------------------------------
    # Dataset size
    # -------------------------------------------------------------------------

    lines.append("## Dataset Size\n")

    lines.append(
        f"- Number of groups: **{len(public_df)}**"
    )

    lines.append(
        f"- Number of public variables: **{len(public_df.columns)}**"
    )

    lines.append(
        f"- Number of parent companies: **{public_df['firm_name'].nunique()}**"
    )

    boy_count = int((public_df["gender"] == 0).sum())
    girl_count = int((public_df["gender"] == 1).sum())

    lines.append(f"- Boy groups: **{boy_count}**")
    lines.append(f"- Girl groups: **{girl_count}**")
    lines.append(
        f"- Intentionally generated anomalies: approximately "
        f"**{len(answer_df[answer_df['is_anomaly'] == 1])}**"
    )

    # -------------------------------------------------------------------------
    # Parent companies and groups
    # -------------------------------------------------------------------------

    lines.append("\n## Parent Companies and Groups\n")

    lines.append(
        "The dataset contains the following fictional parent companies. "
        "Groups are listed underneath their company and separated into "
        "boy groups and girl groups.\n"
    )

    for firm_name, company_df in public_df.groupby(
        "firm_name",
        sort=True,
    ):

        lines.append(f"### {firm_name}\n")

        company_boys = company_df[
            company_df["gender"] == 0
        ]["group_name"].tolist()

        company_girls = company_df[
            company_df["gender"] == 1
        ]["group_name"].tolist()

        lines.append(
            f"**Total groups:** {len(company_df)}  \n"
            f"**Boy groups:** {len(company_boys)}  \n"
            f"**Girl groups:** {len(company_girls)}\n"
        )

        lines.append("\n**Boy groups:**")

        if company_boys:
            for group in company_boys:
                lines.append(f"- {group}")
        else:
            lines.append("- None")

        lines.append("\n**Girl groups:**")

        if company_girls:
            for group in company_girls:
                lines.append(f"- {group}")
        else:
            lines.append("- None")

        lines.append("")

    # -------------------------------------------------------------------------
    # Company configuration
    # -------------------------------------------------------------------------

    lines.append("## Parent Company Parameters\n")

    lines.append(
        "| Company | Tier | Resource base | Consistency bias | Debut share |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|"
    )

    firm_lookup = {
        name: code
        for code, name in FIRM_NAMES.items()
    }

    for firm_name in sorted(public_df["firm_name"].unique()):

        firm_code = firm_lookup[firm_name]
        config = FIRMS[firm_code]

        lines.append(
            f"| {firm_name} | "
            f"{config.tier} | "
            f"{config.resource_base:.2f} | "
            f"{config.consistency_bias:.2f} | "
            f"{config.debut_share:.2f} |"
        )

    lines.append("")

    lines.append(
        "### What these company parameters mean\n"
    )

    lines.append(
        "- **Tier:** A simulated company status/resource category. "
        "Higher values generally represent greater company resources.\n"
    )

    lines.append(
        "- **Resource base:** Baseline level of resources available to "
        "groups from the company. Range in this dataset: "
        "**0.30–0.90**.\n"
    )

    lines.append(
        "- **Consistency bias:** Baseline tendency toward maintaining a "
        "consistent concept. Range: **0.20–0.80**.\n"
    )

    lines.append(
        "- **Debut share:** Relative probability that a generated group "
        "comes from that company. The values are normalized before "
        "sampling, so they represent relative weights rather than "
        "exact percentages.\n"
    )

    # -------------------------------------------------------------------------
    # Public columns
    # -------------------------------------------------------------------------

    lines.append("## Public Variables\n")

    column_descriptions = {
        "group_name":
            "Fictional group name.",

        "firm_name":
            "Fictional parent company.",

        "streaming_log":
            "Log-transformed simulated streaming performance.",

        "chart_peak_inv_log":
            "Log-transformed inverse chart position. Higher values "
            "generally indicate a better simulated peak.",

        "comeback_freq":
            "Average simulated number of comebacks per year.",

        "mv_views_log":
            "Log-transformed simulated music-video views.",

        "album_sales_log":
            "Log-transformed simulated album sales.",

        "award_nominations":
            "Simulated number of award nominations.",

        "music_show_wins":
            "Simulated number of music-show wins.",

        "vocal_stability":
            "Simulated vocal stability score between 0 and 1.",

        "choreo_complexity":
            "Simulated choreography complexity score from 0 to 100.",

        "concept_consistency":
            "Simulated consistency of the group's concept, from 0 to 1.",

        "social_growth_pct":
            "Simulated social-media growth percentage.",

        "fan_cafe_log":
            "Log-transformed simulated fan-cafe activity.",

        "merch_index":
            "Simulated merchandise demand index from 0 to 100.",

        "brand_deals":
            "Simulated number of brand deals.",

        "intl_fan_ratio":
            "Estimated simulated proportion of international fans, "
            "between 0 and 1.",

        "sentiment_polarity":
            "Simulated sentiment score from -1 to 1.",

        "viral_moments":
            "Simulated count of viral moments.",

        "fan_labor_log":
            "Log-transformed simulated fan-labor/activity measure.",

        "member_count":
            "Number of group members.",

        "debut_age_avg":
            "Average simulated age at debut.",

        "training_months_avg":
            "Average simulated training duration in months.",

        "turnover_rate":
            "Simulated member turnover rate, from 0 to 0.60.",

        "contract_renewed":
            "Whether the simulated contract was renewed: 0 = no, "
            "1 = yes.",

        "internal_conflict":
            "Simulated internal conflict score from 0 to 1.",

        "company_tier":
            "Parent company's simulated tier.",

        "debut_year":
            "Simulated debut year, between 2012 and 2026.",

        "years_active":
            "Number of years active at the 2026 observation point.",

        "gender":
            "Gender category used during generation: 0 = boy group, "
            "1 = girl group.",

        "resource_allocation":
            "Simulated fraction of company resources allocated to the group.",

        "market_saturation":
            "Simulated market saturation score from 0 to 1.",
    }

    lines.append(
        "| Variable | Description |"
    )
    lines.append(
        "|---|---|"
    )

    for column in public_df.columns:

        description = column_descriptions.get(
            column,
            "Generated dataset variable."
        )

        lines.append(
            f"| `{column}` | {description} |"
        )

    # -------------------------------------------------------------------------
    # Logged variables
    # -------------------------------------------------------------------------

    lines.append("\n## Why Are Some Variables Logged?\n")

    lines.append(
        "Several variables are stored using `log1p(x)`, which means "
        "`log(1 + x)`. These include:\n"
    )

    logged_columns = [
        "streaming_log",
        "mv_views_log",
        "album_sales_log",
        "fan_cafe_log",
        "fan_labor_log",
        "chart_peak_inv_log",
    ]

    for column in logged_columns:
        if column in public_df.columns:
            lines.append(f"- `{column}`")

    lines.append("")

    lines.append(
        "These variables represent quantities that can have very large "
        "differences between groups. For example, one group might have "
        "thousands of streams while another has millions. A logarithmic "
        "transformation compresses these large values and makes the "
        "distribution easier to analyze.\n"
    )

    lines.append(
        "For example, if the original value is 1,000,000, the stored "
        "value is approximately `log(1 + 1,000,000) = 13.82`. "
        "Therefore, a column ending in `_log` should not be interpreted "
        "as the original measurement itself.\n"
    )

    # -------------------------------------------------------------------------
    # Hidden answer key
    # -------------------------------------------------------------------------

    lines.append("## Hidden Answer Key\n")

    lines.append(
        "The file `kpop_groups_answer_key.csv` contains information that "
        "is intentionally hidden from students. It records the ground "
        "truth used when generating the dataset.\n"
    )

    lines.append(
        "It contains:\n"
    )

    lines.append("- `firm_code` — internal company code.")
    lines.append("- `true_archetype` — the hidden archetype assigned to the group.")
    lines.append("- `is_anomaly` — whether the row was intentionally modified.")
    lines.append("- `latent_commercial` — hidden commercial strength.")
    lines.append("- `latent_digital` — hidden digital momentum.")
    lines.append("- `latent_fandom` — hidden fandom strength.")
    lines.append("- `latent_instability` — hidden instability.")
    lines.append("- `latent_prestige` — hidden prestige.")

    lines.append("\n## Hidden Archetypes\n")

    lines.append(
        "| Archetype | Main latent characteristic |"
    )
    lines.append(
        "|---:|---|"
    )

    archetype_descriptions = {
        1: "Commercial strength",
        2: "Digital momentum",
        3: "Fandom strength",
        4: "Instability",
        5: "Prestige",
    }

    for archetype, description in archetype_descriptions.items():
        lines.append(
            f"| {archetype} | {description} |"
        )

    # -------------------------------------------------------------------------
    # Generation process
    # -------------------------------------------------------------------------

    lines.append("\n## Data Generation Process\n")

    lines.append(
        "1. A random number generator is initialized using the configured seed.\n"
        "2. Each group is assigned a parent company.\n"
        "3. Each company influences which hidden archetypes are more likely.\n"
        "4. A hidden archetype generates five latent characteristics.\n"
        "5. Gender is generated using company-specific probabilities.\n"
        "6. A unique fictional group name is assigned.\n"
        "7. Debut year and years active are generated.\n"
        "8. The latent characteristics and company characteristics are used "
        "to generate the observable variables.\n"
        "9. Approximately 5% of groups receive deliberately unusual "
        "modifications.\n"
        "10. The observable variables are written to the public CSV.\n"
        "11. The hidden ground truth is written to the answer-key CSV.\n"
    )

    # -------------------------------------------------------------------------
    # Educational note
    # -------------------------------------------------------------------------

    lines.append("## Suggested Student Workflow\n")

    lines.append(
        "Students should normally begin with `kpop_groups.csv` and use "
        "this document to understand the structure of the dataset. "
        "They should not use the answer key when attempting to discover "
        "patterns, classify groups, or detect anomalies. The answer key "
        "can be used afterward to evaluate their findings against the "
        "known ground truth.\n"
    )

    INTRO_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"\nDataset introduction:")
    print(INTRO_FILE)

# =============================================================================
# ANOMALIES
# =============================================================================

def inject_anomalies(
    df: pd.DataFrame,
    anomaly_indices: np.ndarray,
) -> None:

    anomaly_types = [
        "legacy_viral_contradiction",
        "factory_fandom_misallocation",
        "rising_conflict_failure",
        "niche_mainstream_breakout",
    ]

    for i, idx in enumerate(anomaly_indices):

        anomaly_type = anomaly_types[i % len(anomaly_types)]

        if anomaly_type == "legacy_viral_contradiction":

            df.loc[idx, "streaming_log"] = np.log1p(5_000_000)
            df.loc[idx, "viral_moments"] = 15
            df.loc[idx, "award_nominations"] = 0
            df.loc[idx, "concept_consistency"] = 0.10

        elif anomaly_type == "factory_fandom_misallocation":

            df.loc[idx, "fan_labor_log"] = np.log1p(5_000)
            df.loc[idx, "intl_fan_ratio"] = 0.80
            df.loc[idx, "comeback_freq"] = 3.5
            df.loc[idx, "resource_allocation"] = 0.20

        elif anomaly_type == "rising_conflict_failure":

            df.loc[idx, "internal_conflict"] = 0.90
            df.loc[idx, "streaming_log"] = np.log1p(100_000)
            df.loc[idx, "turnover_rate"] = 0.40

        elif anomaly_type == "niche_mainstream_breakout":

            df.loc[idx, "chart_peak_inv_log"] = np.log1p(200)
            df.loc[idx, "streaming_log"] = np.log1p(8_000_000)
            df.loc[idx, "award_nominations"] = 12


# =============================================================================
# MAIN GENERATOR
# =============================================================================

def generate_dataset(
    n_groups: int = N_GROUPS,
    seed: int = SEED,
):

    rng = np.random.default_rng(seed)

    # -------------------------------------------------------------------------
    # Sample group-level latent structure
    # -------------------------------------------------------------------------

    firms = sample_firms(n_groups, rng)
    archetypes = sample_archetypes(firms, rng)
    genders = sample_genders(firms, rng)
    names = assign_names(genders, rng)

    # -------------------------------------------------------------------------
    # Choose anomalies independently of the model features.
    #
    # This makes the anomaly label an experimental ground truth rather than
    # something that can be inferred from row number.
    # -------------------------------------------------------------------------

    n_anomalies = max(4, round(n_groups * 0.05))

    anomaly_indices = rng.choice(
        n_groups,
        size=n_anomalies,
        replace=False,
    )

    anomaly_indices = np.sort(anomaly_indices)

    anomaly_set = set(anomaly_indices)

    # -------------------------------------------------------------------------
    # Generate observations
    # -------------------------------------------------------------------------

    public_rows = []
    answer_rows = []

    for i in range(n_groups):

        firm = firms[i]
        archetype = int(archetypes[i])
        gender = int(genders[i])

        latent = sample_archetype(
            archetype,
            rng,
        )

        debut_year = int(
            rng.integers(
                2012,
                OBSERVATION_YEAR + 1,
            )
        )

        years_active = OBSERVATION_YEAR - debut_year

        features = latent_to_features(
            latent=latent,
            firm=firm,
            gender=gender,
            debut_year=debut_year,
            years_active=years_active,
            rng=rng,
        )

        features["group_name"] = names[i]
        features["firm_name"] = FIRM_NAMES[firm]

        public_rows.append(features)

        # Ground truth goes into a separate file.
        answer_rows.append({
            "group_name": names[i],
            "firm_code": firm,
            "true_archetype": archetype,
            "is_anomaly": int(i in anomaly_set),
            "latent_commercial": latent[0],
            "latent_digital": latent[1],
            "latent_fandom": latent[2],
            "latent_instability": latent[3],
            "latent_prestige": latent[4],
        })

    public_df = pd.DataFrame(public_rows)
    answer_df = pd.DataFrame(answer_rows)

    # -------------------------------------------------------------------------
    # Inject anomalies into PUBLIC data.
    # -------------------------------------------------------------------------

    inject_anomalies(
        public_df,
        anomaly_indices,
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    public_df.to_csv(
        PUBLIC_FILE,
        index=False,
    )

    answer_df.to_csv(
        ANSWER_KEY_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    print("=" * 70)
    print("DATASET GENERATED")
    print("=" * 70)

    print(f"Rows:              {len(public_df):,}")
    print(f"Columns:           {len(public_df.columns)}")
    print(f"Anomalies:         {len(anomaly_indices)}")
    print(f"Boy groups:        {(genders == 0).sum()}")
    print(f"Girl groups:       {(genders == 1).sum()}")

    print("\nFirm distribution:")
    print(
        pd.Series(firms)
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nArchetype distribution:")
    print(
        pd.Series(archetypes)
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nPublic dataset:")
    print(PUBLIC_FILE)

    print("\nHidden answer key:")
    print(ANSWER_KEY_FILE)

    generate_data_introduction(
        public_df,
        answer_df,
    )

    print("\nDataset introduction:")
    print(INTRO_FILE)

    print("=" * 70)

    return public_df, answer_df


# =============================================================================
# ENTRY POINT
# =============================================================================

# uv run -m src.data_generation.group_data_gen
if __name__ == "__main__":
    generate_dataset()
