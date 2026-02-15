###############################################################################
# Imports
###############################################################################
from ollama import chat  # <-- Import the chat function from ollama
import asyncio
import json
import logging
import pandas as pd
import enum
import os
import re
import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field, ValidationError
import numpy

TOPN_FEATURES_CT = 10
FEATURE_ALGO_LS = ['lofo','mi','permutation','shap','xgboost']
FEATURE_ALGO_NAME = FEATURE_ALGO_LS[0]
NTOP_SUMMARY_COL = 'ntop_text_summary'
ALL_SUMMARY_COL = 'all_text_summary'

TARGET_COL = 'y_arrestedafter2002'
FLAG_INCL_ID_COL = False

# Ensure FEATURE_DESCIPTION_DT is correctly defined as a dictionary
FEATURE_DESCIPTION_DT = {
    "college02": "highest degree",
    "parentrelations":"parent/guardian relationship at age 12",
    "married02":"married/cohabitation status",
    "urbanrural":"resident locale",
    "faminjail":"other adult family member in jail previous 5 years",
    "experience02":"total job in last year",
    "homelessness":"homeless for 2+ days in past 5 years",
    "askgod":"asks God for help",
    "sex":"sex",
    "foodstamp":"used food stamps in last year",
    "hhsize97":"househouse size as teen 5 years ago",
    "cocaine":"used cocine in past 4 years",
    "depression":"depression in last month",
    "convictedby2002":"convitions in past 5 years",
    "marijuana_anydrug":"used any drug except cocaine in past 4 years",
    "victim":"violent crime victim in past 5 years",
    "num_weight":"weight in pounds",
    "numberofarrestsby2002":"arrests om last 5 years",
    "race":"race",
    "height_total_inches":"height in inches",
    "godhasnothingtodo":"does not believe in God",
    "age":"age",
}

# Manually edit values from Colab notebook Feature importance outputs
FEATURE_IMPORTANCE_DT = {
    "lofo": {
        1:  ("college02",0.31640315019623294),
        2:  ("parentrelations",0.16622644930514224),
        3:  ("married02",0.11189210302320653),
        4:  ("urbanrural",0.05336253643252919),
        5:  ("faminjail",0.04671290618585846),
        6:  ("experience02",0.04207588239393834),
        7:  ("homelessness",0.035446056533800376),
        8:  ("askgod",0.03447195485147537),
        9:  ("sex",0.03091658583763357),
        10: ("foodstamp",0.030738349658660787),
        11: ("hhsize97",0.026845605533599947),
        12: ("cocaine",0.026769658482863914),
        13: ("depression",0.025503788853804978),
        14: ("convictedby2002",0.02486771044166117),
        15: ("marijuana_anydrug",0.023526804395596644),
        16: ("victim",0.015977393538086013),
        17: ("num_weight",0.00958843087967067),
        18: ("numberofarrestsby2002",0.007359434891422724),
        19: ("race",0.003932293141977266),
        20: ("height_total_inches",-0.00027495773651182797),
        21: ("godhasnothingtodo",-0.0065914250649457325),
        22: ("age",-0.025750711775703577),
    },
    "mi": {
        1:  ("parentrelations",0.301762698017622),
        2:  ("college02",0.18789660651759504),
        3:  ("urbanrural",0.14292073749486656),
        4:  ("married02",0.09982998470758618),
        5:  ("race",0.05885801833562226),
        6:  ("homelessness",0.048696491083685664),
        7:  ("numberofarrestsby2002",0.04093888559977543),
        8:  ("experience02",0.03401562894511002),
        9:  ("depression",0.03162582452453523),
        10: ("num_weight",0.02419565717364073),
        11: ("cocaine",0.015569030075649056),
        12: ("godhasnothingtodo",0.010171414368253027),
        13: ("height_total_inches",0.003519023156058786),
        14: ("hhsize97",0.0),
        15: ("victim",0.0),
        16: ("foodstamp",0.0),
        17: ("faminjail",0.0),
        18: ("askgod",0.0),
        19: ("age",0.0),
        20: ("convictedby2002",0.0),
        21: ("sex",0.0),
        22: ("marijuana_anydrug",0.0),
    },
    "permutation": {
        1:  ("college02",0.37055335968379366),
        2:  ("race",0.145256916996047),
        3:  ("parentrelations",0.10573122529644213),
        4:  ("urbanrural",0.10375494071146242),
        5:  ("marijuana_anydrug",0.06916996047430828),
        6:  ("married02",0.05632411067193684),
        7:  ("homelessness",0.055335968379446626),
        8:  ("num_weight",0.04644268774703547),
        9:  ("sex",0.03656126482213434),
        10: ("height_total_inches",0.03162055335968384),
        11: ("depression",0.018774703557312526),
        12: ("godhasnothingtodo",0.016798418972332082),
        13: ("askgod",0.01581027667984198),
        14: ("faminjail",0.0),
        15: ("victim",0.0),
        16: ("cocaine",-0.0009881422924901005),
        17: ("hhsize97",-0.0009881422924901005),
        18: ("age",-0.008893280632410905),
        19: ("experience02",-0.011857707509881209),
        20: ("foodstamp",-0.013833992094861408),
        21: ("numberofarrestsby2002",-0.01679841897233171),
        22: ("convictedby2002",-0.018774703557311912),
    },
    "shap": {
        1:  ("college02",0.12818749593249737),
        2:  ("num_weight",0.09398594486986911),
        3:  ("experience02",0.07169957177042126),
        4:  ("parentrelations",0.06795094011897652),
        5:  ("height_total_inches",0.0650105333619277),
        6:  ("sex",0.05518290673234916),
        7:  ("married02",0.054588915220504935),
        8:  ("cocaine",0.05444696069993243),
        9:  ("urbanrural",0.048476596322157534),
        10: ("race",0.046977370493464066),
        11: ("depression",0.043489218240072465),
        12: ("numberofarrestsby2002",0.042277480568486035),
        13: ("marijuana_anydrug",0.041332989486089516),
        14: ("hhsize97",0.03736089442679975),
        15: ("age",0.030774694228099607),
        16: ("homelessness",0.02667086662463848),
        17: ("convictedby2002",0.01909877728432129),
        18: ("godhasnothingtodo",0.017936632976723068),
        19: ("faminjail",0.0176590656137768),
        20: ("askgod",0.013718871750016366),
        21: ("foodstamp",0.012804887237972462),
        22: ("victim",0.01036838604090408),
    },
    "xgboost": {
        1:  ("parentrelations",0.1775072697414016),
        2:  ("college02",0.12858321978104661),
        3:  ("married02",0.0927474750283767),
        4:  ("race",0.08598612588201723),
        5:  ("depression",0.07281242304655802),
        6:  ("marijuana_anydrug",0.06447120800748955),
        7:  ("homelessness",0.041046826370552764),
        8:  ("sex",0.03798521873653104),
        9:  ("urbanrural",0.03602550732472669),
        10: ("faminjail",0.030673138477233682),
        11: ("numberofarrestsby2002",0.02156229053099052),
        12: ("godhasnothingtodo",0.021470827198982863),
        13: ("experience02",0.021319518937965878),
        14: ("hhsize97",0.021064791025127018),
        15: ("cocaine",0.01972381683540168),
        16: ("victim",0.019693561888721157),
        17: ("foodstamp",0.019549452751568004),
        18: ("num_weight",0.019063104902773784),
        19: ("height_total_inches",0.01860986743581746),
        20: ("age",0.017947127082520036),
        21: ("askgod",0.016353119987126886),
        22: ("convictedby2002",0.01580410902707082),
    }
}


###############################################################################
# 1. LogLevel enum for custom logging
###############################################################################
class LogLevel(enum.Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


###############################################################################
# 2. Global logger and model config
###############################################################################
logger = logging.getLogger(__name__)

CASE_CT = 30
REPEAT_CT = 5  # how many times to repeat each (model+row)

MAX_API_TIME_SEC = 900

INPUT_VIGNETTES_CSV = "vignettes_final_clean.csv" 
DATASET_TYPE="final" # "vignettes_final_clean"
# INPUT_VIGNETTES_CSV = "vignettes_renamed_clean.csv"
# DATASET_TYPE="vignettes_renamed_clean"  # "top14_mi_features"

OUTPUT_SUBDIR = "transcripts_final_20250128"
DATETIME_RUNSTART = datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_OUTPUT_FILENAME = f"log_transcript_{DATASET_TYPE}_{DATETIME_RUNSTART}.txt"
LOG_OUTPUT_PATH = os.path.join("..", OUTPUT_SUBDIR, DATASET_TYPE, LOG_OUTPUT_FILENAME)
LOG_OUTPUT_FILENAME = os.path.join("..", OUTPUT_SUBDIR,  DATASET_TYPE, LOG_OUTPUT_FILENAME)

OLLAMA_MODEL_NAME = "llama3.1:8b-instruct-fp16"  # Adjust to your actual model
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024

RAND_SELECTION_SEED = 42

# These ensembles are synched with /src/config_ollama_models_[all||oss|reasoning|size].yaml files
OLLAMA_ENSEMBLE_SIZE_LS = [
    "llama3.2:1b-instruct-q4_K_M",  # 1400
    "llama3.2:1b-instruct-fp16",    # 538
    "llama3.2:3b-instruct-q4_K_M",
    "llama3.2:3b-instruct-fp16",
    "llama3.1:8b-instruct-q4_K_M",
    "llama3.1:8b-instruct-fp16",
    "llama3.1:70b-instruct-q4_K_M",
    "llama3.3:70b-instruct-q4_K_M",   
    "qwen2.5:0.5b-instruct-q4_K_M",
    "qwen2.5:1.5b-instruct-q4_K_M",
    "qwen2.5:3b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "qwen2.5:14b-instruct-q4_K_M",
    "qwen2.5:32b-instruct-q4_K_M", 
]

OLLAMA_ENSEMBLE_OSS_LS = [
    "command-r:35b-08-2024-q4_K_M",
    "gemma2:9b-instruct-q4_K_M",
    "granite3.1-dense:8b-instruct-q4_K_M",
    "granite3.1-moe:3b-instruct-q4_K_M",
    "mistral:7b-instruct-q4_K_M",
    "phi4:14b-q4_K_M",
]
# Add these from previous SIZE ensemble
#     "qwen2.5:7b-instruct-q4_K_M",
#     "llama3.1:8b-instruct-q4_K_M",

OLLAMA_ENSEMBLE_REASONING_LS = [
    "dolphin3:8b-llama3.1-q4_K_M",
    "exaone3.5:7.8b-instruct-q4_K_M",
    "glm4:9b-chat-q4_K_M",
    "marco-o1:7b-q4_K_M",
    "olmo2:7b-1124-instruct-q4_K_M",
    "falcon3:7b-instruct-q4_K_M",        # 36
    "hermes3:8b-llama3.1-q4_K_M",        # 146 (runpod 03:07)
    "internlm2:7b-chat-1m-v2.5-q4_K_M",
    "nemotron-mini:4b-instruct-q4_K_M",  # 0
    "smallthinker:3b-preview-q4_K_M",    # 104
    "smollm2:1.7b-instruct-q4_K_M",
    "tulu3:8b-q4_K_M",
    "opencoder:8b-instruct-q4_K_M",      # 8
    "qwen2.5:32b-instruct-q4_K_M",
    "yi:9b-v1.5-q4_K_M",
]
# Add these from previous SIZE ensemble
#     "qwen2.5:7b-instruct-q4_K_M",
#     "llama3.1:8b-instruct-q4_K_M",
# Add these from previous REASONING ensemble
#     "command-r:35b-08-2024-q4_K_M",
#     "falcon3:7b-instruct-q4_K_M",
#     "gemma2:9b-instruct-q4_K_M",
#     "granite3.1-dense:8b-instruct-q4_K_M",
#     "llama3.1:8b-instruct-q4_K_M",
#     "marco-o1:7b-q4_K_M",
#     "phi4:14b-q4_K_M",
#     "qwen2.5:7b-instruct-q4_K_M",
#     "tulu3:8b-q4_K_M",
# Add the LARGE REASONING models
#     "tulu3:8b-q4_K_M",
#     "qwq:32b-preview-q4_K_M",
#     "qwen2.5:72b-instruct-q4_K_M",
#     "reflection:70b-q4_K_M",
#     "athene-v2:72b-q4_K_M",

# Assign 1/4 ensembles to the Master list

OLLAMA_MODEL_LS = [
    "gemma2:9b-instruct-q4_K_M",
    "exaone3.5:7.8b-instruct-q4_K_M",
]
'''


OLLAMA_ENSEMBLE_ALL_LS = [
    "aya-expanse:8b-q4_K_M",
    "aya-expanse:8b-fp16",
    "aya-expanse:32b-q4_K_M",
    "command-r:35b-08-2024-q4_K_M",
    "dolphin3:8b-llama3.1-q4_K_M",
    "dolphin3:8b-llama3.1-fp16",
    "dolphin-llama3.1:8b-v2.9.4-Q4_K_M",
    "dolphin-mistral-nemo:12b-v2.9.3-Q8_0",
    "dolphin-2.9.2-qwen2-7b:Q4_K_M",
    "dolphin-llama3.1:8b-v2.9.4-F16",
    "dolphin-mistral-nemo:12b-v2.9.3-Q4_K_M",
    "dolphin-2.9.2-qwen2-7b:fp16",
    "exaone3.5:32b-instruct-q4_K_M",
    "exaone3.5:7.8b-instruct-q4_K_M",
    "exaone3.5:7.8b-instruct-fp16",
    "exaone3.5:2.4b-instruct-q4_K_M",
    "exaone3.5:2.4b-instruct-fp16",
    "falcon3:3b-instruct-fp16",
    "falcon3:7b-instruct-q4_K_M",
    "falcon3:7b-instruct-fp16",
    "falcon3:10b-instruct-q4_K_M",
    "falcon3:10b-instruct-fp16",
    "gemma2:2b-instruct-fp16",
    "gemma2:9b-instruct-q4_K_M",
    "gemma2:9b-instruct-fp16",
    "gemma2:27b-instruct-q4_K_M",
    "glm4:9b-chat-q4_K_M",
    "granite3.1-dense:2b-instruct-fp16",
    "granite3.1-dense:8b-instruct-q4_K_M",
    "granite3.1-dense:8b-instruct-fp16",
    "hermes3:8b-llama3.1-q4_0",
    "hermes3:8b-llama3.1-fp16",
    "llama3.2:1b-instruct-q4_K_M",
    "llama3.2:1b-instruct-fp16",
    "llama3.2:3b-instruct-q4_K_M",
    "llama3.2:3b-instruct-fp16",
    "llama3.1:8b-instruct-q4_K_M",
    "llama3.1:8b-instruct-fp16",
    "llama3.1:70b-instruct-q4_K_M",
    "llama3.3:70b-instruct-q4_K_M",
    "marco-o1:7b-q4_K_M",
    "marco-o1:7b-fp16",
    "mistral:7b-instruct-q4_K_M",
    "mistral:7b-instruct-fp16",
    "mistral-nemo:12b-instruct-2407-q4_K_M",
    "mistral-small:22b-instruct-2409-q4_K_M",
    "mixtral:8x7b-instruct-v0.1-q4_K_M",
    "nemotron-mini:4b-instruct-q4_K_M",
    "nemotron-mini:4b-instruct-fp16",
    "olmo2:7b-1124-instruct-q4_K_M",
    "olmo2:7b-1124-instruct-fp16",
    "olmo2:13b-1124-instruct-q4_K_M",
    "olmo2:13b-1124-instruct-fp16",
    "phi4:14b-q4_K_M",
    "phi4:14b-fp16",
    "qwen2.5:0.5b-instruct-q4_K_M",
    "qwen2.5:1.5b-instruct-q4_K_M",
    "qwen2.5:3b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-fp16",
    "qwen2.5:14b-instruct-q4_K_M",
    "qwen2.5:32b-instruct-q4_K_M",
    "sailor2:8b-chat-q4_K_M",
    "sailor2:20b-chat-q4_K_M",
    "smollm2:1.7b-instruct-q4_K_M",
    "smollm2:1.7b-instruct-fp16",
    "solar-pro:22b-preview-instruct-q4_K_M",
    "tulu3:8b-q4_K_M",
    "tulu3:8b-fp16",
    "tulu3:70b-q4_K_M",
    "athene-v2:72b-q4_K_M",
    "qwen2.5:72b-instruct-q4_K_M",
    "reflection:70b-q4_K_M",
    "smallthinker:3b-preview-q4_K_M",
    "smallthinker:3b-preview-fp16",
    "qwq:32b-preview-q4_K_M",
    "yi:9b-v1.5-q4_K_M",
]
''';


###############################################################################
# 3. Data models
###############################################################################
class JudgeOpinion(BaseModel):
    """Track judge's evolving opinion"""
    prediction: str  # YES or NO
    confidence: int = Field(ge=0, le=100)  # 0-100
    timestamp: datetime = Field(default_factory=datetime.now)
    after_speaker: str  # Which speaker this opinion follows


###############################################################################
# 3B. MetaData class to store Ollama metadata
###############################################################################


class MetaData(BaseModel):
    """Model to represent metadata from an API response."""
    model: Optional[str] = Field(None, description="Model name")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    done: Optional[bool] = Field(None, description="Completion status")
    total_duration_ns: Optional[float] = Field(None, description="Total duration in ns")
    load_duration_ns: Optional[float] = Field(None, description="Load duration in ns")
    prompt_eval_count: Optional[int] = Field(None, ge=0, description="Prompt token count")
    prompt_eval_duration_ns: Optional[float] = Field(None, ge=0, description="Prompt eval time in ns")
    eval_count: Optional[int] = Field(None, ge=0, description="Completion token count")
    eval_duration_ns: Optional[float] = Field(None, ge=0, description="Completion time in ns")
    
    # Derived (seconds) fields
    total_duration_sec: Optional[float] = None
    load_duration_sec: Optional[float] = None
    prompt_eval_duration_sec: Optional[float] = None
    eval_duration_sec: Optional[float] = None

    # Additional local duration from Python measurement
    python_api_duration_sec: Optional[float] = None

'''
class DebateResponse(BaseModel):
    """Structure for debate responses"""
    content: str
    reasoning: Optional[List[str]] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=100)
    critique: Optional[str] = None

    # <-- ADDED: store the predicted YES/NO (especially for final judge)
    prediction: Optional[str] = "UNKNOWN"

    @classmethod
    def parse_response(cls, response_text: str) -> 'DebateResponse':
        """Parse raw response text into structured format, including 'prediction' if present."""
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
                
                # Clean reasoning list if present
                if 'reasoning' in data and isinstance(data['reasoning'], list):
                    data['reasoning'] = [
                        str(item) if isinstance(item, str)
                        else str(item.get('factor', str(item)))
                        for item in data['reasoning']
                    ]
                
                # Try to ensure a valid confidence
                raw_conf = data.get("confidence", 75.0)
                try:
                    conf_val = float(raw_conf)
                    conf_val = min(max(conf_val, 0.0), 100.0)  # clamp 0..100
                except (ValueError, TypeError):
                    conf_val = 75.0

                # Extract a final "prediction" if present:
                raw_pred = str(data.get('prediction', '')).upper()
                if raw_pred not in ["YES", "NO"]:
                    # fallback check if raw_pred includes 'YES'
                    if "YES" in raw_pred:
                        raw_pred = "YES"
                    elif "NO" in raw_pred:
                        raw_pred = "NO"
                    else:
                        raw_pred = "UNKNOWN"

                return cls(
                    content=data.get('content', response_text),
                    reasoning=data.get('reasoning', []),
                    confidence=conf_val,
                    critique=data.get('critique'),
                    prediction=raw_pred
                )
            
            # Fallback to a basic structure if no JSON blocks found
            return cls(
                content=response_text,
                confidence=75.0,
                reasoning=["Failed to parse structured response"]
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Error parsing response: {str(e)}")
            # Return partial object with note that we couldn't parse
            return cls(
                content=response_text,
                confidence=75.0,
                reasoning=["Failed to parse structured response"]
            )
'''


class DebateResponse(BaseModel):
    """Structure for debate responses"""
    content: str
    reasoning: Optional[List[str]] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=100)
    critique: Optional[str] = None  # Updated logic handles lists in parse_response
    prediction: Optional[str] = "UNKNOWN"

    @classmethod
    def parse_response(cls, response_text: str) -> 'DebateResponse':
        """Parse raw response text into structured format, including 'prediction' if present."""
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)

                # Normalize reasoning list if present
                if 'reasoning' in data and isinstance(data['reasoning'], list):
                    data['reasoning'] = [
                        str(item) if isinstance(item, str)
                        else str(item.get('factor', str(item)))
                        for item in data['reasoning']
                    ]

                # Normalize confidence value
                raw_conf = data.get("confidence", 75.0)
                try:
                    conf_val = float(raw_conf)
                    conf_val = min(max(conf_val, 0.0), 100.0)  # Clamp to 0..100
                except (ValueError, TypeError):
                    conf_val = 75.0

                # Normalize critique (convert list to string if necessary)
                raw_critique = data.get("critique", None)
                if isinstance(raw_critique, list):
                    raw_critique = " ".join(raw_critique)  # Join list into a single string

                # Extract prediction
                raw_pred = str(data.get('prediction', '')).upper()
                if raw_pred not in ["YES", "NO"]:
                    if "YES" in raw_pred:
                        raw_pred = "YES"
                    elif "NO" in raw_pred:
                        raw_pred = "NO"
                    else:
                        raw_pred = "UNKNOWN"

                return cls(
                    content=data.get('content', response_text),
                    reasoning=data.get('reasoning', []),
                    confidence=conf_val,
                    critique=raw_critique,
                    prediction=raw_pred
                )
            
            # Fallback to a basic structure if no JSON blocks found
            return cls(
                content=response_text,
                confidence=75.0,
                reasoning=["Failed to parse structured response"]
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Error parsing response: {str(e)}")
            # Return partial object with note that we couldn't parse
            return cls(
                content=response_text,
                confidence=75.0,
                reasoning=["Failed to parse structured response"]
            )

class CourtAgent(BaseModel):
    """Enhanced agent with memory and strategy"""
    role: str
    persona: str
    strategy_points: List[str] = Field(default_factory=list)
    memory: List[str] = Field(default_factory=list)
    
    def update_memory(self, observation: str):
        self.memory.append(observation)
        
    def add_strategy(self, point: str):
        self.strategy_points.append(point)


###############################################################################
# 4. Utility classes and functions
###############################################################################
class TranscriptWriter:
    """Handles creation of court transcript"""

    def __init__(self, output_dir: str = OUTPUT_SUBDIR): 
        # "transcripts"):
        self.transcript_entries = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def add_entry(self, timestamp: datetime, role: str, content: str, metadata: dict = None, metadata_api: dict = None):
        """Add an entry to the transcript"""
        entry = {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "metadata_api": metadata or {}
        }
        self.transcript_entries.append(entry)
        
    def write_transcript(self, case_summary: dict = None, row_identifier: str = ""):
        """
        Write complete transcript to file.
        
        row_identifier: A custom string (e.g., 'row-12_ver-1') 
                        to make filenames unique for each row/case.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Incorporate row_identifier into filename if provided
        filename = f"court_transcript_{row_identifier}_{timestamp}.txt".replace(" ", "_")
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"COURT TRANSCRIPT\n{'='*50}\n\n")
            
            if case_summary:
                f.write("CASE SUMMARY\n")
                f.write(f"Summary: {case_summary.get('summary', 'N/A')}\n")
                f.write(f"Age: {case_summary.get('age', 'N/A')}\n")
                f.write(f"Prior Arrests: {case_summary.get('numberofarrestsby2002', 'N/A')}\n\n")
            
            f.write("PROCEEDINGS\n")
            f.write("="*50 + "\n\n")
            
            for entry in self.transcript_entries:
                f.write(f"[{entry['timestamp']}] {entry['role']}\n")
                f.write("-"*50 + "\n")
                f.write(f"{entry['content']}\n")
                
                if entry['metadata']:
                    f.write("\nMetadata:\n")
                    # Print each metadata key-value
                    for k, v in entry['metadata'].items():
                        f.write(f"  {k}: {v}\n")
                f.write("\n" + "="*50 + "\n\n")

                if entry['metadata_api']:
                    f.write("\nMetadata API:\n")
                    # Print each metadata key-value
                    for k, v in entry['metadata_api'].items():
                        f.write(f"  {k}: {v}\n")
                f.write("\n" + "="*50 + "\n\n")

        logger.info(f"Transcript successfully written to {filepath.resolve()}")
        return filepath


def setup_custom_logging(level: LogLevel = LogLevel.INFO, base_dir: str = "logs") -> logging.Logger:
    """Configure custom logging with CLI and file output"""
    global logger  # Use the global logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(base_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Set the log level
    logger.setLevel(level.value)
    
    # Console handler with simple format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level.value)
    console_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    console_handler.setFormatter(console_format)
    
    # File handler with detailed format
    file_handler = logging.FileHandler(
        log_dir / f'log_debate_{timestamp}.txt',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # Always debug level for file
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Log initial setup
    logger.info(f"Logging initialized at {timestamp}")
    logger.info(f"Log file: {log_dir}/log_debate_{timestamp}.txt")
    logger.info(f"Log level: {level.name}")
    
    return logger


def log_api_interaction(logger: logging.Logger, stage: str, data: dict, meta: dict = None):
    """Helper to log API interactions consistently"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if stage == "REQUEST":
        logger.debug(f"\n{'='*50}")
        logger.debug(f"API REQUEST @ {timestamp}")
        logger.debug(f"PROMPT:\n{data.get('prompt', '')}")
        
    elif stage == "RESPONSE":
        duration = meta.get('duration', 0) if meta else 0
        logger.debug(f"\nAPI RESPONSE @ {timestamp} (took {duration:.2f}s)")
        logger.debug(f"Content:\n{data.get('content', '')}")
        if meta:
            logger.debug("Metadata:")
            for k, v in meta.items():
                logger.debug(f"  {k}: {v}")
        logger.debug(f"{'='*50}\n")


import pandas as pd

# Function to generate feature summary
def get_features_summary(df, col_new_summary, col_features_ls):
    """
    Returns top features list, and expanded DataFrame with feature descriptions.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        algo_name (str): Algorithm name to select feature importance.
        topn_ct (int): Number of top features to extract.

    Returns:
        col_features_ls (list): Top feature column names.
        expanded_df (pd.DataFrame): DataFrame with 'ntop_text_summary' column.
    """

    logger.debug(f"Creating summary column: {col_new_summary}")
    logger.debug(f"Using features: {col_features_ls}")
    logger.debug(f"DataFrame columns before: {df.columns.tolist()}")

    # Create a new column 'ntop_text_summary'
    def generate_summary(row):
        descriptions = []
        for col in col_features_ls:
            description = FEATURE_DESCIPTION_DT.get(col, f"Description not found for {col}")
            value = row.get(col, "N/A")
            descriptions.append(f"{description} is {value}")
        return ", and ".join(descriptions)
    
    # Apply the summary generation to each row
    # expanded_df = df.copy(deep=True)
    # text_summary = df.apply(generate_summary, axis=1)
    df[col_new_summary] = df.apply(generate_summary, axis=1)
    logger.info(f"new text summary col: {col_new_summary}\n  for col_features_ls: {col_features_ls}")
    
    logger.debug(f"DataFrame columns after: {df.columns.tolist()}")
    return df



def read_all_vignettes(filepath: str, col_features_ls: List[str] = [], FLAG_INCL_ID_COL: bool = True) -> pd.DataFrame:
    """
    Reads the CSV file and optionally includes the first column ('id').
    
    Parameters:
        filepath (str): Path to the CSV file.
        FLAG_INCL_ID_COL (bool): Whether to include the first column ('id') in the DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame containing the contents of the file.
    """
    logger.info(f"Reading all vignettes from: {filepath}")
    
    # Read the CSV without assuming any specific index column
    df = pd.read_csv(filepath, index_col=None)
    
    # Rename "Unnamed: 0" to "id"
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)

    # Handle case for 'id' column inclusion or exclusion
    if not FLAG_INCL_ID_COL:
        if 'id' in df.columns:  # Remove the 'id' column if present
            df = df.drop(columns=['id'])
            logger.info("Excluded 'id' column from DataFrame")
    
    logger.info(f"Loaded {len(df)} total vignettes with columns: {list(df.columns)}")

    # Get ntop_summary_text for n-top features)
    df = get_features_summary(df, NTOP_SUMMARY_COL, col_features_ls)
    df = get_features_summary(df, ALL_SUMMARY_COL, [col for col in df.columns if col not in [NTOP_SUMMARY_COL, TARGET_COL, 'id']])
    
    # Get all_sumamry_text for all features (excludes NTOP_SUMMARY_COL, TARGET_COL, and 'id' if present)
    # col_all_features = df.columns.to_list()
    # col_all_features = [col for col in col_all_features if col not in [NTOP_SUMMARY_COL, TARGET_COL, 'id']]
    # df = get_features_summary(df, ALL_SUMMARY_COL, col_features_ls=col_all_features)

    # Verify columns exist
    required_cols = [NTOP_SUMMARY_COL, ALL_SUMMARY_COL, TARGET_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def select_vignettes(df_all: pd.DataFrame, 
                     strategy: str = "random", 
                     row_ct: int = CASE_CT, 
                     arg_extra: Optional[int] = None) -> pd.DataFrame:
    if strategy == "random":
        seed = arg_extra if arg_extra is not None else RAND_SELECTION_SEED
        logger.info(f"Selecting random {row_ct} rows with seed={seed}")
        df_subset = df_all.sample(n=row_ct, random_state=seed)
    elif strategy == "first-nrows":
        logger.info(f"Selecting first {row_ct} rows.")
        df_subset = df_all.head(row_ct)
    else:
        logger.warning(f"Unknown strategy={strategy}, returning the entire df.")
        df_subset = df_all
    logger.info(f"Selected {len(df_subset)} vignettes (strategy={strategy}).")
    return df_subset

def clean_model_name(model_name: str) -> str:
    # same as in your code
    import re
    cleaned = model_name.strip().lower()
    cleaned = re.sub(r'[:.]+', '_', cleaned)
    cleaned = re.sub(r'[^\w]+', '_', cleaned)
    cleaned = re.sub(r'_+', '_', cleaned)
    cleaned = cleaned.strip('_')
    return cleaned


###############################################################################
# 5. CourtDebateManager
###############################################################################
class CourtDebateManager:
    def __init__(
        self, 
        case_facts_df: Optional[pd.DataFrame] = None, 
        row: Optional[pd.Series] = None,
        model_name: str = OLLAMA_MODEL_NAME
    ):
        # Verify required columns exist in row
        required_cols = [NTOP_SUMMARY_COL, ALL_SUMMARY_COL, 'age', 'numberofarrestsby2002', TARGET_COL]
        if row is not None:
            missing_cols = [col for col in required_cols if col not in row.index]
            if missing_cols:
                raise ValueError(f"Row missing required columns: {missing_cols}")

        self.case_facts_df = case_facts_df
        self.model_name = model_name  # store the chosen model name
        self.debate_history = []
        self.transcript = TranscriptWriter()
        self.judge_opinions: List[JudgeOpinion] = []
        
        # Initialize agents
        self.prosecutor = CourtAgent(
            role="Prosecutor",
            persona="""You are an experienced prosecutor focused on public safety. You:
- Emphasize patterns of behavior and risk factors that suggest likely reoffending
- Use statistical evidence and criminology research to support arguments
- Focus on protecting potential future victims
- Maintain professional demeanor while being assertive
- Structure arguments clearly with supporting evidence
- Consider both aggravating circumstances and risk factors"""
        )
        
        self.defender = CourtAgent(
            role="Defense Attorney",
            persona="""You are a dedicated defense attorney focused on rehabilitation. You:
- Emphasize potential for rehabilitation and positive life changes
- Highlight mitigating circumstances and environmental factors
- Focus on the individual circumstances of your client
- Advocate for alternative interventions to incarceration
- Maintain professional composure while being persuasive
- Structure arguments to address both legal and human aspects"""
        )
        
        self.judge = CourtAgent(
            role="Judge",
            persona="""You are an experienced judge focused on fair evaluation. You:
- Consider both public safety and rehabilitation potential
- Evaluate arguments based on evidence and legal precedent
- Maintain strict courtroom decorum
- Focus on objective risk assessment
- Consider both statistical and individual factors
- Make decisions based on clear reasoning"""
        )
        
        # If a specific row was provided, use it; else pick from case_facts_df randomly
        if row is not None:

            # Compare 'row.index' vs. the known columns
            row_columns_list = list(row.index)
            logger.debug(f"In CourtDebateManager.__init__, row.columns: {row_columns_list}")

            missing_cols = [col for col in required_cols if col not in row_columns_list]
            if missing_cols:
                error_msg = f"Row is missing required columns: {missing_cols}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            self.selected_case = {
                'ntop_text_summary': row[NTOP_SUMMARY_COL],  # ['short_text_summary'],
                'all_text_summary': row[ALL_SUMMARY_COL],
                'age': row['age'],
                'numberofarrestsby2002': row['numberofarrestsby2002'],
                'y_arrestedafter2002': row[TARGET_COL] # ['target']
            }
        else:
            if case_facts_df is not None:
                self.selected_case = self._select_random_case()
            else:
                raise ValueError("Must provide either row or case_facts_df")

    def _select_random_case(self) -> Dict:
        """Randomly select a case and prepare its facts"""
        random_row = self.case_facts_df.sample(n=1).iloc[0]
        return {
            'summary': random_row[NTOP_SUMMARY_COL],  # ['short_text_summary'],
            'details': random_row[ALL_SUMMARY_COL],
            'age': random_row['age'],
            'numberofarrestsby2002': random_row['numberofarrestsby2002'],
            'target': random_row[TARGET_COL] # ['target']
        }

    def _build_prompt(self, agent: CourtAgent, turn: int) -> str:
        """Build contextualized prompt for the agent"""
        COURTROOM_RULES = """
1. Arguments must be based on presented evidence and facts
2. No personal attacks or inflammatory language
3. Maintain professional courtroom decorum
4. Address specific risk factors and protective factors
5. Consider both statistical and individual factors
6. Focus on future risk prediction, not punishment
7. Respect time limits and turn-taking
8. Present clear, structured arguments
9. Cite relevant research or precedent when available
10. Acknowledge but professionally challenge opposing arguments
"""
        template = f"""
ROLE: {agent.role}
PERSONA: {agent.persona}

TASK: Evaluate and argue whether the defendant is likely to be rearrested within 3 years.

COURTROOM RULES:
{COURTROOM_RULES}

CASE FACTS:
{self.selected_case['ntop_text_summary']}
Detailed Background: {self.selected_case['all_text_summary']}

KEY STATISTICS:
- Age: {self.selected_case['age']}
- Prior Arrests: {self.selected_case['numberofarrestsby2002']}

PREVIOUS ARGUMENTS:
{chr(10).join(self.debate_history)}

YOUR STRATEGY POINTS:
{chr(10).join(agent.strategy_points)}

CURRENT TASK:
{"Present opening argument" if turn == 1 else "Respond to previous argument"}
Focus on specific factors that increase or decrease recidivism risk within 3 years.

Provide your response in JSON format:
{{
    "content": "Your main argument",
    "reasoning": ["<Reason #1>", "<Reason #2>", "<Reason #3>"...],
    "confidence": 0-100,
    "critique": "Self-reflection on argument strength",
    "prediction": "YES or NO"
}}
"""
        return template.strip()

    async def _get_agent_response(
        self,
        agent: CourtAgent,
        prompt: str,
        model_name: str = OLLAMA_MODEL_NAME,
        timeout: float = MAX_API_TIME_SEC,
        row_identifier: str = ""
    ) -> Optional[DebateResponse]:
        """
        Get response from Ollama (or another LLM) with enhanced logging.
        Returns None if the response is malformed or parsing fails significantly.
        """
        start_time = datetime.now()
        
        # Log the request
        log_api_interaction(logger, "REQUEST", {"prompt": prompt})
        
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    chat,
                    messages=[
                        {'role': 'system', 'content': agent.persona},
                        {'role': 'user', 'content': prompt}
                    ],
                    model=self.model_name,  # use the manager's model_name
                    options={
                        'temperature': DEFAULT_TEMPERATURE,
                        'max_tokens': DEFAULT_MAX_TOKENS
                    }
                ),
                timeout=timeout
            )

            duration = (datetime.now() - start_time).total_seconds()

            # Prepare metadata from Ollama response
            # If your 'response' has a .raw or .__dict__ that holds these keys, adapt accordingly
            raw_meta = getattr(response, 'raw', {})
            # Convert from ns -> sec helper
            def ns_to_s(val):
                return val / 1e9 if isinstance(val, int) else None

            ollama_meta = MetaData(
                model=getattr(response, 'model', None),
                created_at=getattr(response,'created_at'),
                done=getattr(response,'done'),
                total_duration_ns=ns_to_s(getattr(response,'total_duration')),
                load_duration_ns=ns_to_s(getattr(response,'load_duration')),
                prompt_eval_count=getattr(response,'prompt_eval_count'),
                prompt_eval_duration_ns=ns_to_s(getattr(response,'prompt_eval_duration')),
                eval_count=getattr(response,'eval_count'),
                eval_duration_ns=getattr(response,'eval_duration'),
                total_duration_sec=ns_to_s(getattr(response,'total_duration')),
                load_duration_sec=ns_to_s(getattr(response,'load_duration')),
                prompt_eval_duration_sec=ns_to_s(getattr(response,'prompt_eval_duration')),
                eval_duration_sec=ns_to_s(getattr(response,'eval_duration')),
                python_api_duration_sec=duration
            )
            
            response_text = response.message.content
            # Combine local + remote metadata
            metadata_dict = ollama_meta.model_dump()

            # Log the response
            log_api_interaction(logger, "RESPONSE", {"content": response_text}, {"duration": duration, **metadata_dict})
            
            # Attempt to parse the response
            debate_response = DebateResponse.parse_response(response_text)

            logger.debug(f"DebateResponse parse status: {debate_response}")
            logger.debug(f"Returning None? {debate_response is None}")

            # Check for parse failure
            # E.g., if parse_response returns a "Failed to parse structured response" in reasoning
            if ("Failed to parse structured response" in debate_response.reasoning
                and debate_response.prediction == "UNKNOWN"):
                
                logger.warning("Malformed response from model. Saving transcript as malformed.")
                malformed_suffix = datetime.now().strftime("malformed_%Y%m%d_%H%M%S")
                # Write out the partial transcript before returning None
                malformed_identifier = f"{row_identifier}_{agent.role}_{malformed_suffix}"
                transcript_path = self.transcript.write_transcript(
                    case_summary={"summary": "..."},
                    row_identifier=malformed_identifier
                )
                # Also write a JSON with the same suffix
                malformed_json_path = transcript_path.with_suffix(".json")
                with open(malformed_json_path, 'w', encoding='utf-8') as mf:
                    json.dump(
                        {
                            "error": "Malformed response from model",
                            "response_text": response_text,
                            "metadata": metadata_dict
                        },
                        mf,
                        indent=2
                    )
                return None

            # If parse succeeded, add to transcript
            self.transcript.add_entry(
                timestamp=datetime.now(),
                role=agent.role,
                content=debate_response.content,
                metadata={
                    'metadata_api': metadata_dict,
                    'prediction': debate_response.prediction,
                    'confidence': debate_response.confidence,
                    'reasoning': debate_response.reasoning
                },
                metadata_api=metadata_dict
            )
            
            return debate_response
            
        except asyncio.TimeoutError:
            logger.warning(f"Timeout for agent {agent.role}")
            return None
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            return None

    async def _get_silent_judge_opinion(
        self,
        speaker: str,
        latest_argument: str
    ) -> Optional[JudgeOpinion]:
        """Get judge's current opinion after each speaker."""
        prompt = f"""
ROLE: Silent Judge
TASK: Based on the current state of the debate, provide ONLY your current prediction 
on whether the defendant will reoffend within 3 years.

CASE SUMMARY:
{self.selected_case[NTOP_SUMMARY_COL]}

LATEST ARGUMENT ({speaker}):
{latest_argument}

PREVIOUS DEBATE HISTORY:
{chr(10).join(self.debate_history)}

Provide your current opinion in JSON format:
{{
    "prediction": "YES or NO",
    "confidence": 0-100
}}
Keep in mind this is just your current opinion, not final ruling.
"""
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    chat,
                    messages=[
                        {'role': 'system', 'content': self.judge.persona},
                        {'role': 'user', 'content': prompt}
                    ],
                    model=OLLAMA_MODEL_NAME,
                    options={
                        'temperature': DEFAULT_TEMPERATURE,
                        'max_tokens': DEFAULT_MAX_TOKENS
                    }
                ),
                timeout=MAX_API_TIME_SEC
            )
            
            response_text = response.message.content
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response_text[json_start:json_end])
                opinion = JudgeOpinion(
                    prediction=data['prediction'].upper(),
                    confidence=int(data['confidence']),
                    after_speaker=speaker
                )
                logger.debug(f"Silent judge opinion after {speaker}: {opinion.model_dump_json()}")
                return opinion
            else:
                logger.warning("Judge opinion not in JSON format.")
                return None

        except Exception as e:
            logger.error(f"Error getting judge opinion: {str(e)}")
            return None

    async def conduct_debate(self, rounds: int = 3, row_identifier: str = "") -> Dict:
        """Main debate loop."""
        try:
            # Initialize strategies
            self.prosecutor.strategy_points = [
                "Focus on number of prior arrests",
                "Identify risk patterns in background",
                "Emphasize public safety concerns",
                "Present statistical recidivism data"
            ]
            
            self.defender.strategy_points = [
                "Highlight rehabilitation potential",
                "Identify positive life factors",
                "Present alternatives to detention",
                "Address risk factors constructively"
            ]
            
            # Conduct debate rounds
            for round_num in range(1, rounds + 1):
                logger.info(f"Starting round {round_num}")
                
                # Prosecution turn
                pros_prompt = self._build_prompt(self.prosecutor, round_num)
                pros_response = await self._get_agent_response(
                    self.prosecutor, pros_prompt,
                    row_identifier=row_identifier
                )
                
                if pros_response:
                    self.debate_history.append(f"Prosecutor: {pros_response.content}")
                    self.prosecutor.update_memory(pros_response.content)
                    # Get judge's silent opinion
                    judge_opinion = await self._get_silent_judge_opinion("Prosecutor", pros_response.content)
                    if judge_opinion:
                        self.judge_opinions.append(judge_opinion)
                
                # Defense turn
                def_prompt = self._build_prompt(self.defender, round_num)
                def_response = await self._get_agent_response(
                    self.defender, def_prompt,
                    row_identifier=row_identifier
                )
                
                if def_response:
                    self.debate_history.append(f"Defense: {def_response.content}")
                    self.defender.update_memory(def_response.content)
                    # Get judge's silent opinion
                    judge_opinion = await self._get_silent_judge_opinion("Defense", def_response.content)
                    if judge_opinion:
                        self.judge_opinions.append(judge_opinion)
            
            # Judge's final evaluation
            judge_prompt = f"""
ROLE: Judge
PERSONA: {self.judge.persona}

CASE SUMMARY:
{self.selected_case[NTOP_SUMMARY_COL]}

DEBATE HISTORY:
{chr(10).join(self.debate_history)}

Based on all arguments presented, provide your final ruling:
1. Will the defendant reoffend within 3 years? (YES/NO)
2. Your confidence level (0-100%)
3. Detailed reasoning for your decision
4. Assessment of key arguments from both sides

Provide your response in JSON format:
{{
    "prediction": "YES or NO",
    "confidence": 0-100,
    "content": "Your detailed explanation and reasoning",
    "reasoning": ["Analysis point 1", "Analysis point 2", ...],
    "critique": "Assessment of arguments from both sides"
}}
"""
            judge_response = await self._get_agent_response(
                self.judge, judge_prompt,
                row_identifier=row_identifier
            )
            
            logger.debug(f"judge_response={judge_response}")
            if judge_response is None:
                logger.debug("Judge response is None -> final_ruling stays 'No decision'.")
            else:
                logger.debug("Judge responded with a valid DebateResponse object.")


            transcript_path = None
            final_ruling = {
                "prediction": "No decision",
                "confidence": 0,
                "content": "",
                "reasoning": [],
                "critique": None
            }
            if judge_response:
                # Insert two lines of '=' after judge's ruling
                self.debate_history.append(
                    f"Judge's Ruling: {judge_response.content}\n{'='*30}\n{'='*30}"
                )
                
                final_ruling = {
                    "prediction": judge_response.prediction or "UNKNOWN",
                    "confidence": judge_response.confidence,
                    "content": judge_response.content,
                    "reasoning": judge_response.reasoning,
                    "critique": judge_response.critique
                }
                # Write final transcript, including row_identifier
                transcript_path = self.transcript.write_transcript(
                    case_summary={
                        "ntop_text_summary": self.selected_case[NTOP_SUMMARY_COL],
                        "age": self.selected_case['age'],
                        "numberofarrestsby2002": self.selected_case['numberofarrestsby2002']
                    },
                    row_identifier=row_identifier
                )
                logger.info(f"Transcript written to: {transcript_path}") 
            
            # Return complete debate results with opinion evolution
            return {
                "case": {
                    "ntop_text_summary": self.selected_case[NTOP_SUMMARY_COL],
                    "age": int(self.selected_case['age']),
                    "numberofarrestsby2002": int(self.selected_case['numberofarrestsby2002']),
                    "y_arrestedafter2002": bool(self.selected_case[TARGET_COL]) # ['target'])
                },
                "debate_history": self.debate_history,
                "judge_opinion_evolution": [
                    {
                        "timestamp": op.timestamp.isoformat(),
                        "after_speaker": op.after_speaker,
                        "prediction": op.prediction,
                        "confidence": op.confidence
                    }
                    for op in self.judge_opinions
                ],
                "final_ruling": final_ruling,
                "transcript_path": str(transcript_path) if transcript_path else None,
                "timestamp": datetime.now().isoformat()
            }
        
        # except Exception as e:
        #     logger.error(f"Error in debate: {str(e)}")
        #     return {"error": str(e)}

        except Exception as e:
            logger.error(f"Error in debate: {str(e)}")
            # Also check if judge_response is defined or None
            logger.debug(f"judge_response in except block: {judge_response}")
            return {
                "error": str(e),
                "final_ruling": {
                    "prediction": judge_response.prediction or "UNKNOWN",
                    "confidence": judge_response.confidence,
                    "content": judge_response.content,
                    "reasoning": judge_response.reasoning,
                    "critique": judge_response.critique
                }
            }



###############################################################################
# 6. Create a utility function to parse and convert the log output into a
#    "human-readable" court transcript with timestamped sections.
###############################################################################
def parse_logs_and_write_court_transcript(log_path: str, output_path: str) -> None:
    """
    Read the entire log at `log_path` and parse out the interesting sections:
    - Timestamp
    - Role (Prosecutor, Defense, Judge)
    - Reasoning, confidence, meta info if available
    Then write a human-readable version to `output_path`.
    """
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Log file not found: {log_path}")
        return

    sections = []
    current_section = []

    for line in lines:
        current_section.append(line)
        # Example triggers: "API REQUEST", "API RESPONSE", "Prosecutor:", "Defense:", etc.
        if ("Prosecutor:" in line 
            or "Defense:" in line 
            or "Judge" in line 
            or "API RESPONSE" in line):
            sections.append("".join(current_section))
            current_section = []
    if current_section:
        sections.append("".join(current_section))

    with open(output_path, 'w', encoding='utf-8') as out:
        out.write(f"Human-Readable Court Transcript\n{'='*60}\n")
        out.write(f"Source Log: {log_path}\n\n")
        for idx, sect in enumerate(sections, start=1):
            out.write(f"--- Section {idx} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            out.write(sect.strip() + "\n")
            out.write(f"{'-'*60}\n\n")


###############################################################################
# 7. Main entry point
###############################################################################
async def main():
    # 1) Initialize logging
    logger = setup_custom_logging(level=LogLevel.DEBUG)
    logger.info("Starting nested debates with 3-turn structure each time.")

    # Get top-n features list in col_features_ls
    feature_importance = FEATURE_IMPORTANCE_DT.get(FEATURE_ALGO_NAME, {})
    col_features_ls = [feature[0] for feature in feature_importance.values()][:TOPN_FEATURES_CT]
    logger.info(f"Top {TOPN_FEATURES_CT} features: {', '.join(col_features_ls)}")

    # 2) Read and select vignettes
    PATH_VIGNETTES_CSV = os.path.join("..", "data", INPUT_VIGNETTES_CSV)
    df_all = read_all_vignettes(PATH_VIGNETTES_CSV, col_features_ls)
    
    # Example: get 30 random rows with seed=42
    df = select_vignettes(df_all, strategy="random", row_ct=CASE_CT, arg_extra=42)

    logger.debug(f"Columns in df_all right after read_all_vignettes:\n {df_all.columns.tolist()}")
    sample_row = df_all.iloc[0].copy()
    logger.debug(f"Sample row index=0 has columns:\n {list(sample_row.index)}")
    logger.debug(f"Sample row[ntop_text_summary] = {sample_row.get(NTOP_SUMMARY_COL, 'MISSING')}")

    df = select_vignettes(df_all, strategy="random", row_ct=CASE_CT, arg_extra=42)
    logger.debug(f"Columns in df right after select_vignettes:\n {df.columns.tolist()}")


    for original_model_name in OLLAMA_MODEL_LS:
        # 3) Clean the model name for OS-safe usage
        safe_model_name = clean_model_name(original_model_name)
        logger.info(f"--- MODEL: {safe_model_name} ---")
        print(f"Loading model {original_model_name}")
        
        # Prepare final transcripts dir
        model_output_dir = os.path.join('..',OUTPUT_SUBDIR, safe_model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        for idx, row in df.iterrows():
            print(f"Processing row #{idx}: {row}")

            # Initialize an empty list to store our descriptions
            descriptions = []
            
            # Iterate through each column and its value in the row
            for column_name, value in row.items():
                # Skip the target column
                if column_name != TARGET_COL: 
                    # 'target':
                    # Add each column-value pair as a descriptive string
                    descriptions.append(f"{column_name} is {value}")

            # Join all descriptions with "and" between them
            short_text = ", and ".join(descriptions)



            # Extract the top N feature column names based on the algo_name
            feature_importance = FEATURE_IMPORTANCE_DT.get(FEATURE_ALGO_NAME, {})
            col_features_ls = [feature[0] for feature in feature_importance.values()][:TOPN_FEATURES_CT]
            
            # Create a new column 'ntop_text_summary'
            # def generate_summary(row):
            #     descriptions = []
            #     for col in col_features_ls:
            #         description = FEATURE_DESCIPTION_DT.get(col, f"Description not found for {col}")
            #         value = row.get(col, "N/A")
            #         descriptions.append(f"{description} is {value}")
            #     return ", and ".join(descriptions)
            
            # Apply the summary generation to each row
            # expanded_df = df.copy(deep=True)
            # ntop_text_summary = df.apply(generate_summary, axis=1)
            # logger.info(f"{NTOP_SUMMARY_COL}: {ntop_text_summary}")
            # expanded_df[NTOP_SUMMARY_COL] = ntop_text_summary

            # row[NTOP_SUMMARY_COL] = ntop_text_summary
            # row['short_text_summary'] = short_text
            actual_reoffended = row[TARGET_COL] # ['target']  # True/False or 1/0
            logger.info(f"Processing row idx={idx}, short_text={short_text[:50]}..., target={actual_reoffended}")

            for attempt_i in range(REPEAT_CT):
                logger.info(f"Starting attempt {attempt_i+1} for model={safe_model_name}, row={idx}")
                
                # 4) Build output filenames
                txt_filename = f"transcript_row-{idx}_ver-{attempt_i+1}.txt"
                json_filename = f"transcript_row-{idx}_ver-{attempt_i+1}.json"
                txt_path = os.path.join(model_output_dir, txt_filename)
                json_path = os.path.join(model_output_dir, json_filename)
                
                # 5) Skip if these transcripts already exist (restartable)
                if os.path.exists(txt_path) or os.path.exists(json_path):
                    logger.info(f"Skipping row={idx}, ver={attempt_i+1} - transcripts already exist.")
                    continue

                if 'ntop_text_summary' not in row.index:
                    logger.error("Missing ntop_text_summary in row, skipping or raising!")
                    continue


                logger.debug(f"Row index={idx} has columns: {list(row.index)}")
                # Or also check row keys if it's a Series: logger.debug(f"Row keys: {row.to_dict().keys()}")
                
                if 'ntop_text_summary' not in row.index:
                    logger.error("Missing ntop_text_summary in row, skipping or raising!")
                    continue

                # 6) Create manager and run debate
                debate_manager = CourtDebateManager(
                    row=row,
                    model_name=original_model_name  # pass the original if needed
                )
                
                # Provide row_identifier for the transcript
                row_id_str = f"row-{idx}_ver-{attempt_i+1}"
                result = await debate_manager.conduct_debate(rounds=3, row_identifier=row_id_str)
                
                # 7) Evaluate correctness
                final_pred_str = result["final_ruling"]["prediction"].strip().upper()
                predicted_reoffend = True if "YES" in final_pred_str else False if "NO" in final_pred_str else None
                if (predicted_reoffend == actual_reoffended):
                    prediction_accurate = 'True'
                elif (predicted_reoffend == True) and (actual_reoffended == False):
                    prediction_accurate = 'False'
                elif (predicted_reoffend == False) and (actual_reoffended == True):
                    prediction_accurate = 'False'
                else:
                    prediction_accurate = 'unknown'
                # prediction_accurate = (predicted_reoffend == actual_reoffended)
                # result["prediction_accurate"] = prediction_accurate
                result["prediction_accurate"] = prediction_accurate

                # Also add correctness line to the transcript so it appears in the .txt
                # We'll add a final entry
                debate_manager.transcript.add_entry(
                    timestamp=datetime.now(),
                    role="System",
                    content=(
                        f"Judge Prediction: {predicted_reoffend}\n"
                        f"Actual reoffended: {actual_reoffended}\n"
                        f"Prediction accurate: {prediction_accurate}"
                    ),
                    metadata={"correctness": prediction_accurate}
                )

                # Because we added a new entry, re-write the transcript to the same path:
                old_transcript_path = result.get("transcript_path", None)
                if old_transcript_path is not None and os.path.exists(old_transcript_path):
                    shutil.move(old_transcript_path, txt_path)
                    logger.info(f"Transcript moved to {txt_path}")
                    logger.debug(f"Transcript original path: {old_transcript_path}")
                    logger.debug(f"Transcript new path: {os.path.abspath(txt_path)}")

                else:
                    # If for some reason it wasn't generated, we can write again:
                    logger.warning("No transcript_path found in debate result. Writing anew.")
                    re_written_path = debate_manager.transcript.write_transcript(
                        case_summary={
                            "summary": row[NTOP_SUMMARY_COL],  # ['short_text_summary'],
                            "age": row['age'],
                            "arrests": row['numberofarrestsby2002']
                        },
                        row_identifier=row_id_str
                    )
                    if os.path.exists(re_written_path):
                        shutil.move(re_written_path, txt_path)
                        logger.info(f"Transcript forcibly moved to {txt_path}")
                        logger.debug(f"Transcript original path: {old_transcript_path}")
                        logger.debug(f"Transcript new path: {os.path.abspath(txt_path)}")

                
                # 8) Write JSON
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(result, jf, indent=2, default=str)
                logger.info(f"Saved JSON results to {json_path}")
            

    logger.info("All nested debates completed successfully.")


if __name__ == "__main__":

    os.makedirs(os.path.dirname(LOG_OUTPUT_PATH), exist_ok=True)
    
    original_stdout = sys.stdout
    sys.stdout = open(LOG_OUTPUT_PATH, 'w', encoding='utf-8')
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    except Exception as e:
        print(f"Unhandled error: {e}")
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout
        logging.shutdown()
    
    print(f"All logs captured in: {LOG_OUTPUT_PATH}")

