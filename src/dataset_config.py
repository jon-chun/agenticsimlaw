"""
Dataset configuration for AgenticSimLaw debate experiments.

Provides frozen dataclass configs for NLSY97 and COMPAS datasets,
including feature descriptions, feature importance rankings, and
dataset-specific metadata needed by the debate runner.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class DatasetConfig:
    """Immutable dataset configuration for debate experiments."""
    name: str
    display_name: str
    target_col: str
    vignettes_csv: str  # filename in data/
    feature_descriptions: Dict[str, str]
    feature_importance: Dict[str, Dict[int, Tuple[str, float]]]
    key_statistics: Dict[str, str]  # generic key -> dataset column name
    key_statistics_labels: Dict[str, str]  # generic key -> display label
    default_topn: int
    task_description: str


# ---------------------------------------------------------------------------
# NLSY97 Configuration
# ---------------------------------------------------------------------------

NLSY97_CONFIG = DatasetConfig(
    name="nlsy97",
    display_name="NLSY97",
    target_col="y_arrestedafter2002",
    vignettes_csv="sample_vignettes.csv",
    feature_descriptions={
        "college02": "highest degree",
        "parentrelations": "parent/guardian relationship at age 12",
        "married02": "married/cohabitation status",
        "urbanrural": "resident locale",
        "faminjail": "other adult family member in jail previous 5 years",
        "experience02": "total job in last year",
        "homelessness": "homeless for 2+ days in past 5 years",
        "askgod": "asks God for help",
        "sex": "sex",
        "foodstamp": "used food stamps in last year",
        "hhsize97": "househouse size as teen 5 years ago",
        "cocaine": "used cocine in past 4 years",
        "depression": "depression in last month",
        "convictedby2002": "convitions in past 5 years",
        "marijuana_anydrug": "used any drug except cocaine in past 4 years",
        "victim": "violent crime victim in past 5 years",
        "num_weight": "weight in pounds",
        "numberofarrestsby2002": "arrests om last 5 years",
        "race": "race",
        "height_total_inches": "height in inches",
        "godhasnothingtodo": "does not believe in God",
        "age": "age",
    },
    feature_importance={
        "lofo": {
            1:  ("college02", 0.31640315019623294),
            2:  ("parentrelations", 0.16622644930514224),
            3:  ("married02", 0.11189210302320653),
            4:  ("urbanrural", 0.05336253643252919),
            5:  ("faminjail", 0.04671290618585846),
            6:  ("experience02", 0.04207588239393834),
            7:  ("homelessness", 0.035446056533800376),
            8:  ("askgod", 0.03447195485147537),
            9:  ("sex", 0.03091658583763357),
            10: ("foodstamp", 0.030738349658660787),
            11: ("hhsize97", 0.026845605533599947),
            12: ("cocaine", 0.026769658482863914),
            13: ("depression", 0.025503788853804978),
            14: ("convictedby2002", 0.02486771044166117),
            15: ("marijuana_anydrug", 0.023526804395596644),
            16: ("victim", 0.015977393538086013),
            17: ("num_weight", 0.00958843087967067),
            18: ("numberofarrestsby2002", 0.007359434891422724),
            19: ("race", 0.003932293141977266),
            20: ("height_total_inches", -0.00027495773651182797),
            21: ("godhasnothingtodo", -0.0065914250649457325),
            22: ("age", -0.025750711775703577),
        },
        "mi": {
            1:  ("parentrelations", 0.301762698017622),
            2:  ("college02", 0.18789660651759504),
            3:  ("urbanrural", 0.14292073749486656),
            4:  ("married02", 0.09982998470758618),
            5:  ("race", 0.05885801833562226),
            6:  ("homelessness", 0.048696491083685664),
            7:  ("numberofarrestsby2002", 0.04093888559977543),
            8:  ("experience02", 0.03401562894511002),
            9:  ("depression", 0.03162582452453523),
            10: ("num_weight", 0.02419565717364073),
            11: ("cocaine", 0.015569030075649056),
            12: ("godhasnothingtodo", 0.010171414368253027),
            13: ("height_total_inches", 0.003519023156058786),
            14: ("hhsize97", 0.0),
            15: ("victim", 0.0),
            16: ("foodstamp", 0.0),
            17: ("faminjail", 0.0),
            18: ("askgod", 0.0),
            19: ("age", 0.0),
            20: ("convictedby2002", 0.0),
            21: ("sex", 0.0),
            22: ("marijuana_anydrug", 0.0),
        },
        "permutation": {
            1:  ("college02", 0.37055335968379366),
            2:  ("race", 0.145256916996047),
            3:  ("parentrelations", 0.10573122529644213),
            4:  ("urbanrural", 0.10375494071146242),
            5:  ("marijuana_anydrug", 0.06916996047430828),
            6:  ("married02", 0.05632411067193684),
            7:  ("homelessness", 0.055335968379446626),
            8:  ("num_weight", 0.04644268774703547),
            9:  ("sex", 0.03656126482213434),
            10: ("height_total_inches", 0.03162055335968384),
            11: ("depression", 0.018774703557312526),
            12: ("godhasnothingtodo", 0.016798418972332082),
            13: ("askgod", 0.01581027667984198),
            14: ("faminjail", 0.0),
            15: ("victim", 0.0),
            16: ("cocaine", -0.0009881422924901005),
            17: ("hhsize97", -0.0009881422924901005),
            18: ("age", -0.008893280632410905),
            19: ("experience02", -0.011857707509881209),
            20: ("foodstamp", -0.013833992094861408),
            21: ("numberofarrestsby2002", -0.01679841897233171),
            22: ("convictedby2002", -0.018774703557311912),
        },
        "shap": {
            1:  ("college02", 0.12818749593249737),
            2:  ("num_weight", 0.09398594486986911),
            3:  ("experience02", 0.07169957177042126),
            4:  ("parentrelations", 0.06795094011897652),
            5:  ("height_total_inches", 0.0650105333619277),
            6:  ("sex", 0.05518290673234916),
            7:  ("married02", 0.054588915220504935),
            8:  ("cocaine", 0.05444696069993243),
            9:  ("urbanrural", 0.048476596322157534),
            10: ("race", 0.046977370493464066),
            11: ("depression", 0.043489218240072465),
            12: ("numberofarrestsby2002", 0.042277480568486035),
            13: ("marijuana_anydrug", 0.041332989486089516),
            14: ("hhsize97", 0.03736089442679975),
            15: ("age", 0.030774694228099607),
            16: ("homelessness", 0.02667086662463848),
            17: ("convictedby2002", 0.01909877728432129),
            18: ("godhasnothingtodo", 0.017936632976723068),
            19: ("faminjail", 0.0176590656137768),
            20: ("askgod", 0.013718871750016366),
            21: ("foodstamp", 0.012804887237972462),
            22: ("victim", 0.01036838604090408),
        },
        "xgboost": {
            1:  ("parentrelations", 0.1775072697414016),
            2:  ("college02", 0.12858321978104661),
            3:  ("married02", 0.0927474750283767),
            4:  ("race", 0.08598612588201723),
            5:  ("depression", 0.07281242304655802),
            6:  ("marijuana_anydrug", 0.06447120800748955),
            7:  ("homelessness", 0.041046826370552764),
            8:  ("sex", 0.03798521873653104),
            9:  ("urbanrural", 0.03602550732472669),
            10: ("faminjail", 0.030673138477233682),
            11: ("numberofarrestsby2002", 0.02156229053099052),
            12: ("godhasnothingtodo", 0.021470827198982863),
            13: ("experience02", 0.021319518937965878),
            14: ("hhsize97", 0.021064791025127018),
            15: ("cocaine", 0.01972381683540168),
            16: ("victim", 0.019693561888721157),
            17: ("foodstamp", 0.019549452751568004),
            18: ("num_weight", 0.019063104902773784),
            19: ("height_total_inches", 0.01860986743581746),
            20: ("age", 0.017947127082520036),
            21: ("askgod", 0.016353119987126886),
            22: ("convictedby2002", 0.01580410902707082),
        },
    },
    key_statistics={"age": "age", "prior_offenses": "numberofarrestsby2002"},
    key_statistics_labels={"age": "Age", "prior_offenses": "Prior Arrests"},
    default_topn=10,
    task_description="rearrested within 3 years",
)


_COMPAS_LOFO_FALLBACK = [
    "priors_count", "race", "decile_score", "sex", "age",
    "c_charge_degree", "juv_other_count", "juv_misd_count", "juv_fel_count",
]


def _load_compas_feature_importance() -> Dict[str, Dict[int, Tuple[str, float]]]:
    """Load COMPAS LOFO ranking from JSON and convert to ver25-compatible format."""
    json_path = Path(__file__).resolve().parent.parent / "data" / "compas_topn_feature_by_algo.json"
    try:
        with open(json_path) as f:
            raw = json.load(f)
    except FileNotFoundError:
        import logging
        logging.getLogger(__name__).warning(
            f"COMPAS feature importance JSON not found at {json_path}; "
            "using hardcoded LOFO fallback"
        )
        raw = {"lofo": _COMPAS_LOFO_FALLBACK}

    result = {}
    for algo, ranking in raw.items():
        # ranking is a list of feature names ordered by importance
        result[algo] = {
            rank + 1: (feat, 0.0)  # score not available from list format
            for rank, feat in enumerate(ranking)
        }
    return result


COMPAS_CONFIG = DatasetConfig(
    name="compas",
    display_name="COMPAS",
    target_col="two_year_recid",
    vignettes_csv="compas_vignettes.csv",
    feature_descriptions={
        "age": "age at COMPAS screening",
        "sex": "sex",
        "race": "race",
        "juv_fel_count": "juvenile felony count",
        "juv_misd_count": "juvenile misdemeanor count",
        "juv_other_count": "juvenile other offense count",
        "priors_count": "number of prior adult offenses",
        "c_charge_degree": "current charge degree",
        "decile_score": "COMPAS recidivism risk score (1-10)",
    },
    feature_importance=_load_compas_feature_importance(),
    key_statistics={"age": "age", "prior_offenses": "priors_count"},
    key_statistics_labels={"age": "Age", "prior_offenses": "Prior Offenses"},
    default_topn=9,
    task_description="recidivate within 2 years",
)


_COMPAS_NODECILE_LOFO_FALLBACK = [
    "priors_count", "race", "sex", "age",
    "c_charge_degree", "juv_other_count", "juv_misd_count", "juv_fel_count",
]


def _load_compas_nodecile_feature_importance() -> Dict[str, Dict[int, Tuple[str, float]]]:
    """Load COMPAS LOFO ranking excluding decile_score."""
    fi = _load_compas_feature_importance()
    result = {}
    for algo, ranking in fi.items():
        filtered = {
            new_rank: (feat, score)
            for new_rank, (_, (feat, score)) in enumerate(
                ((r, v) for r, v in sorted(ranking.items()) if v[0] != "decile_score"),
                start=1,
            )
        }
        result[algo] = filtered if filtered else {
            r + 1: (f, 0.0) for r, f in enumerate(_COMPAS_NODECILE_LOFO_FALLBACK)
        }
    return result


COMPAS_NODECILE_CONFIG = DatasetConfig(
    name="compas_nodecile",
    display_name="COMPAS (no decile)",
    target_col="two_year_recid",
    vignettes_csv="compas_vignettes.csv",
    feature_descriptions={
        "age": "age at COMPAS screening",
        "sex": "sex",
        "race": "race",
        "juv_fel_count": "juvenile felony count",
        "juv_misd_count": "juvenile misdemeanor count",
        "juv_other_count": "juvenile other offense count",
        "priors_count": "number of prior adult offenses",
        "c_charge_degree": "current charge degree",
    },
    feature_importance=_load_compas_nodecile_feature_importance(),
    key_statistics={"age": "age", "prior_offenses": "priors_count"},
    key_statistics_labels={"age": "Age", "prior_offenses": "Prior Offenses"},
    default_topn=8,
    task_description="recidivate within 2 years",
)


def get_dataset_config(name: str) -> DatasetConfig:
    """Return dataset config by name."""
    configs = {
        "nlsy97": NLSY97_CONFIG,
        "compas": COMPAS_CONFIG,
        "compas_nodecile": COMPAS_NODECILE_CONFIG,
    }
    if name not in configs:
        raise ValueError(f"Unknown dataset: {name!r}. Choose from: {list(configs.keys())}")
    return configs[name]
