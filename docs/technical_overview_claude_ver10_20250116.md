# Court Debate Simulation System: Technical Documentation

## System Overview

This system implements an automated court debate simulation using large language models (LLMs) to generate structured arguments between prosecutors, defenders, and judges. The system processes case vignettes and evaluates recidivism prediction through multi-round debates.

## TODO:

* judge 'No Decision' being recorded as accuracy=True if actual=YES
* PREVIOUS ARGUMENTS: missing in text report
* restartable: first chedk if output file already exists before API calls
* first model check: first check if model loaded, if not skip rather than create empty files
* backoff retry strategy
* QA: make sure no default YES or NO if any error state (eg no model)


## Core Components

### 1. Data Models

#### Base Classes
- `LogLevel`: Enum class extending Python's logging levels
- `JudgeOpinion`: Pydantic model tracking evolving judicial opinions with prediction (YES/NO), confidence (0-100), timestamp, and speaker context
- `MetaData`: Pydantic model for Ollama API metadata, including timing metrics in both nanoseconds and seconds
- `DebateResponse`: Pydantic model for structured debate responses including:
  - content: main argument text
  - reasoning: list of supporting points
  - confidence: 0-100 score
  - critique: self-reflection
  - prediction: YES/NO outcome
- `CourtAgent`: Pydantic model representing debate participants with:
  - role: position in court
  - persona: behavioral characteristics
  - strategy_points: tactical approach list
  - memory: historical interaction list

### 2. Key Manager Classes

#### TranscriptWriter
Handles creation and persistence of court proceedings:
```python
class TranscriptWriter:
    def add_entry(self, timestamp: datetime, role: str, content: str, 
                 metadata: dict = None, metadata_api: dict = None)
    def write_transcript(self, case_summary: dict = None, 
                        row_identifier: str = "") -> Path
```

#### CourtDebateManager
Core orchestrator managing the entire debate simulation:
```python
class CourtDebateManager:
    def __init__(self, case_facts_df: Optional[pd.DataFrame] = None,
                 row: Optional[pd.Series] = None,
                 model_name: str = OLLAMA_MODEL_NAME)
    async def conduct_debate(self, rounds: int = 3, 
                           row_identifier: str = "") -> Dict
```

### 3. Utility Functions

#### Data Processing
```python
def read_all_vignettes(filepath: str) -> pd.DataFrame
def select_vignettes(df_all: pd.DataFrame, strategy: str = "random",
                    row_ct: int = CASE_CT, 
                    arg_extra: Optional[int] = None) -> pd.DataFrame
def clean_model_name(model_name: str) -> str
```

#### Logging & Monitoring
```python
def setup_custom_logging(level: LogLevel = LogLevel.INFO, 
                        base_dir: str = "logs") -> logging.Logger
def log_api_interaction(logger: logging.Logger, stage: str, 
                       data: dict, meta: dict = None)
```

## Control Flow

1. **Initialization**
   - System loads configuration and sets up logging
   - Vignettes are loaded from CSV
   - Model selection and initialization occurs

2. **Debate Cycle**
   - For each vignette and model combination:
     - CourtDebateManager instantiated
     - Multiple debate rounds conducted
     - Each round includes:
       - Prosecutor argument
       - Silent judge evaluation
       - Defense argument
       - Silent judge evaluation
     - Final ruling collected
     - Results and transcripts saved

3. **Output Generation**
   - Detailed transcripts in both TXT and JSON formats
   - Comprehensive logging throughout process
   - Performance and accuracy metrics captured

## Extension Points

1. **New Agent Types**
   - Extend `CourtAgent` for additional roles
   - Implement new persona types
   - Add specialized strategy points

2. **Model Integration**
   - Add new LLM support in `_get_agent_response`
   - Extend model configuration options
   - Implement alternative API interfaces

3. **Analysis Capabilities**
   - Add new metrics collection
   - Implement additional evaluation criteria
   - Extend transcript analysis capabilities

## Potential Improvements

1. **Technical Enhancements**
   - Implement proper connection pooling
   - Add retry mechanisms for API failures
   - Implement caching for common prompts
   - Add batch processing capabilities
   - Implement parallel debate execution

2. **Functional Additions**
   - Add support for more complex debate structures
   - Implement cross-examination phases
   - Add evidence presentation mechanisms
   - Implement jury simulation
   - Add statistical analysis tools

3. **Quality Improvements**
   - Add comprehensive unit tests
   - Implement validation for debate consistency
   - Add automated quality checks for responses
   - Implement prompt optimization
   - Add performance benchmarking tools

## Important Considerations

1. **Performance**
   - System uses async/await for API calls
   - Timeout handling implemented (MAX_API_TIME_SEC)
   - Error recovery mechanisms in place

2. **Data Handling**
   - Structured logging throughout
   - Error states captured and persisted
   - Malformed responses handled gracefully

3. **Configurability**
   - Multiple model options available
   - Customizable debate rounds
   - Flexible logging levels
   - Configurable timeouts and retry logic

## Usage Guidelines

1. **Configuration**
   ```python
   OLLAMA_MODEL_NAME = "model_name"
   DEFAULT_TEMPERATURE = 0.7
   DEFAULT_MAX_TOKENS = 1024
   MAX_API_TIME_SEC = 30.0
   ```

2. **Execution**
   ```python
   debate_manager = CourtDebateManager(case_facts_df=df)
   result = await debate_manager.conduct_debate(rounds=3)
   ```

3. **Output Processing**
   ```python
   transcript_path = debate_manager.transcript.write_transcript(
       case_summary=summary_dict,
       row_identifier=identifier
   )
   ```

## Error Handling

The system implements comprehensive error handling:
- API timeouts
- Malformed responses
- File I/O errors
- Data validation failures
- Model interaction issues

Each error case is logged and can be recovered from, allowing the system to continue processing remaining cases even if individual debates fail.