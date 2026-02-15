import os
import re
import json
from json_repair import repair_json
import re
import pandas as pd
import logging
from typing import Dict, Any, Tuple, List, Optional

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global constants
ENSEMBLE_TYPE_LS = ['size', 'oss', 'reasoning', 'all', 'final']
ENSEMBLE_TYPE_NAME = ENSEMBLE_TYPE_LS[4]
ENSEMBLE_DATE = "20250127"

# Define input directory structure
INPUT_TRANSCRIPTS_DIR = f"transcripts_ensemble_{ENSEMBLE_TYPE_NAME}_{ENSEMBLE_DATE}"
# Use the transcripts directory directly as our root
INPUT_ROOT_DIR = os.path.join('..', INPUT_TRANSCRIPTS_DIR)

# Define output directory structure 
# Create a parallel directory for reports using the same base name
OUTPUT_DIR = os.path.join("..", "transcripts_aggregate_reports", INPUT_TRANSCRIPTS_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define output filename
OUTPUT_FILENAME = f"transcripts_aggregate_{ENSEMBLE_TYPE_NAME}_report_{ENSEMBLE_DATE}.csv"

# Log the directory structure for debugging
logger.info(f"Directory Structure:")
logger.info(f"  Input Root Dir: {os.path.abspath(INPUT_ROOT_DIR)}")
logger.info(f"  Output Dir: {os.path.abspath(OUTPUT_DIR)}")
logger.info(f"  Output File: {OUTPUT_FILENAME}")

def str2dict(metadata_str: str) -> Dict[str, Any]:
    """Convert string representation of metadata to dictionary."""
    try:
        # Remove single quotes and replace with double quotes
        cleaned_str = metadata_str.replace("'", '"')
        logger.debug(f"Cleaned metadata string: {cleaned_str[:100]}...")
        return json.loads(cleaned_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse metadata: {e}")
        return {}

def extract_filename_info(filename: str) -> Tuple[str, str]:
    """Extract row number and version number from filename."""
    match = re.search(r'transcript_row-(\d+)_ver-(\d+)', filename)
    if match:
        logger.debug(f"Extracted row={match.group(1)}, ver={match.group(2)} from {filename}")
        return match.group(1), match.group(2)
    logger.warning(f"Failed to extract row/ver from filename: {filename}")
    return "", ""

def extract_speaker_segments(txt_content: str) -> List[str]:
    """Extract segments for each speaker from the text content."""
    # More flexible pattern matching for speaker markers
    pattern = r'\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]\s+(?:Prosecutor|Defense|Defense Attorney|Judge)'
    segments = re.split(pattern, txt_content)
    valid_segments = [seg for seg in segments if seg.strip()]
    
    logger.debug(f"Found {len(valid_segments)} speaker segments")
    logger.debug(f"First segment preview: {valid_segments[0][:100] if valid_segments else 'None'}")
    return valid_segments



def extract_metadata_from_segment(segment: str) -> Dict[str, Any]:
    """Extract metadata from a single speaker segment with improved parsing."""
    logger.debug(f"\nProcessing segment (first 100 chars): {segment[:100]}")
    
    # Look for metadata section with more flexible pattern matching
    # Updated pattern to handle both forms of metadata headers and capture the entire JSON block
    metadata_pattern = r'(?:Metadata:|Metadata API:)\s*\n\s*metadata_api:\s*(\{[^}]+\})'
    match = re.search(metadata_pattern, segment, re.MULTILINE | re.DOTALL)
    
    if not match:
        logger.warning("No metadata section found in segment")
        return {}
        
    try:
        metadata_str = match.group(1)
        # Don't truncate the metadata string in logging
        logger.debug(f"Found complete metadata string: {metadata_str}")
        
        # Use json_repair to handle malformed JSON
        repaired_json = repair_json(metadata_str)
        
        # Parse the repaired JSON
        metadata_dict = json.loads(repaired_json)
        logger.debug(f"Successfully parsed metadata: {list(metadata_dict.keys())}")
        return metadata_dict
            
    except Exception as e:
        logger.error(f"Failed to parse metadata: {str(e)}\nOriginal string: {metadata_str}")
        # Return empty dict but don't fail completely
        return {}







def extract_speaker_sequence(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract speaker sequence from judge_opinion_evolution."""
    evolution_data = json_data.get('judge_opinion_evolution', [])
    if not evolution_data:
        logger.warning("No judge_opinion_evolution data found in JSON")
        return []
    
    sequence = []
    for i, opinion in enumerate(evolution_data):
        entry = {
            'after_speaker': opinion.get('after_speaker', ''),
            'prediction': opinion.get('prediction', ''),
            'confidence': opinion.get('confidence', 0)
        }
        sequence.append(entry)
        logger.debug(f"Speaker {i}: {entry['after_speaker']}, "
                    f"prediction={entry['prediction']}, "
                    f"confidence={entry['confidence']}")
    
    return sequence

def process_transcript_pair(txt_path: str, json_path: str, model_name: str) -> Dict[str, Any]:
    """Process a pair of transcript files with improved JSON handling."""
    logger.info(f"\nProcessing files:\n  TXT: {txt_path}\n  JSON: {json_path}")
    row_data = {'model_name': model_name}
    
    try:
        # Extract file info
        base_filename = os.path.splitext(os.path.basename(txt_path))[0]
        row_no, ver_no = extract_filename_info(base_filename)
        row_data.update({'row_no': row_no, 'ver_no': ver_no})
        
        # Read and parse JSON file
        with open(json_path, 'r') as f:
            json_content = f.read()
            
        # Use json_repair to handle potentially malformed JSON
        repaired_json = repair_json(json_content)
        json_data = json.loads(repaired_json)
        
        # Don't immediately return on "No decision" - log it and continue
        if '"prediction": "No decision"' in json_content:
            logger.warning(f"'No decision' found in {json_path} - continuing with partial data")
        
        logger.debug(f"JSON keys: {list(json_data.keys())}")
        
        # Extract case data with proper type handling and default values
        case_data = json_data.get('case', {})
        final_ruling = json_data.get('final_ruling', {})
        
        row_data.update({
            'age': float(case_data.get('age', 0)) if case_data.get('age') else None,
            'numberofarrestsby2002': int(case_data.get('numberofarrestsby2002', 0)) if case_data.get('numberofarrestsby2002') else None,
            'y_arrestedafter2002': bool(case_data.get('y_arrestedafter2002', False)),
            'prediction': final_ruling.get('prediction', ''),
            'confidence': float(final_ruling.get('confidence', 0)) if final_ruling.get('confidence') else None,
            'transcript_path': json_data.get('transcript_path', ''),
            'timestamp': json_data.get('timestamp', ''),
            'prediction_accurate': bool(json_data.get('prediction_accurate', False))
        })
        
        # Process speaker sequence
        speaker_sequence = extract_speaker_sequence(json_data)
        logger.debug(f"Extracted {len(speaker_sequence)} speakers from JSON")
        
        # Read and process text file with proper encoding
        with open(txt_path, 'r', encoding='utf-8') as f:
            txt_content = f.read()
        
        txt_segments = extract_speaker_segments(txt_content)
        logger.debug(f"Extracted {len(txt_segments)} segments from TXT")
        
        # Only process metadata if we have matching segments and sequence
        if len(txt_segments) >= len(speaker_sequence):
            speaker_data = process_speaker_metadata(txt_segments, speaker_sequence)
            row_data.update(speaker_data)
        else:
            logger.warning(f"Segment count mismatch: {len(txt_segments)} segments vs {len(speaker_sequence)} sequence items")
        
    except Exception as e:
        logger.error(f"Error processing files: {e}", exc_info=True)
        # Return partial data instead of empty dict
        return row_data
    
    return row_data

def process_speaker_metadata(segments: List[str], sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process metadata for each speaker turn with improved error handling."""
    combined_data = {}
    
    for speaker_ct, (segment, speaker_data) in enumerate(zip(segments, sequence)):
        logger.debug(f"\nProcessing speaker {speaker_ct}")
        
        try:
            # Extract metadata with improved error handling
            metadata = extract_metadata_from_segment(segment)
            
            # Build speaker-specific fields with explicit type conversion and default values
            speaker_fields = {
                f'speaker_{speaker_ct}_after': str(speaker_data.get('after_speaker', '')),
                f'speaker_{speaker_ct}_prediction': str(speaker_data.get('prediction', '')),
                f'speaker_{speaker_ct}_confidence': float(speaker_data.get('confidence', 0)),
                f'total_duration_ns_{speaker_ct}': float(metadata.get('total_duration_ns', 0)),
                f'prompt_eval_ct_{speaker_ct}': int(metadata.get('prompt_eval_count', 0)),
                f'eval_ct_{speaker_ct}': int(metadata.get('eval_count', 0)),
                f'total_duration_sec_{speaker_ct}': float(metadata.get('total_duration_sec', 0)),
                f'python_api_duration_sec_{speaker_ct}': float(metadata.get('python_api_duration_sec', 0))
            }
            
            # Log successful field additions
            for key, value in speaker_fields.items():
                if value:  # Only log non-zero/non-empty values
                    logger.debug(f"Adding {key}: {value}")
            
            combined_data.update(speaker_fields)
            
        except Exception as e:
            logger.error(f"Error processing speaker {speaker_ct}: {e}")
            # Continue with next speaker instead of failing
            continue
    
    return combined_data

def main():
    """Main function to process all transcript files."""
    all_data = []
    
    for model_dir in os.listdir(INPUT_ROOT_DIR):
        model_path = os.path.join(INPUT_ROOT_DIR, model_dir)
        if not os.path.isdir(model_path):
            continue
        
        logger.info(f"\nProcessing model directory: {model_dir}")
        json_files = [f for f in os.listdir(model_path) if f.endswith('.json')]
        
        for json_file in json_files:
            base_name = os.path.splitext(json_file)[0]
            txt_file = base_name + '.txt'
            
            json_path = os.path.join(model_path, json_file)
            txt_path = os.path.join(model_path, txt_file)
            
            if os.path.exists(txt_path):
                row_data = process_transcript_pair(txt_path, json_path, model_dir)
                all_data.append(row_data)
    
    df = pd.DataFrame(all_data)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    df.to_csv(output_path, index=False)
    logger.info(f"\nCreated aggregate report at: {output_path}")
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

if __name__ == "__main__":
    main()