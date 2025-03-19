import argparse
import configparser
import logging
import os
import sys
import subprocess
import tempfile
import time
import concurrent.futures
import math

from pydub import AudioSegment, silence
from tqdm import tqdm

def read_config(config_path):
    """Reads configuration from config.ini file."""
    config = configparser.ConfigParser()
    if os.path.exists(config_path):
        config.read(config_path, encoding="utf-8")
    else:
        logging.warning(f"Configuration file {config_path} not found. Probably you should create it in the script directory. Using default values.")
    return config

def check_whisper_cli(cli_path):
    """Verifies whisper-cli by running '-h' and checking exit code."""
    try:
        subprocess.run([cli_path, "-h"], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        logging.error("whisper-cli is not working. Check the path.")
        sys.exit(1)
    except FileNotFoundError:
        logging.error("whisper-cli executable not found.")
        sys.exit(1)

def check_model_file(model_path):
    """Checks that the model file exists and has non-zero size."""
    if not os.path.exists(model_path):
        logging.error(f"Model file at path {model_path} not found.")
        sys.exit(1)
    if os.path.getsize(model_path) == 0:
        logging.error(f"Model file {model_path} has zero size. Check file correctness.")
        sys.exit(1)

def check_audio_file(mp3_file):
    """Checks that the audio file exists and can be loaded."""
    if not os.path.exists(mp3_file):
        logging.error(f"Audio file {mp3_file} not found.")
        sys.exit(1)
    try:
        audio = AudioSegment.from_file(mp3_file)
        return audio
    except Exception as e:
        logging.error(f"Error loading audio file {mp3_file}: {e}")
        sys.exit(1)

def merge_segments(segments, merge_gap):
    """
    Merges adjacent segments if the gap between them does not exceed merge_gap (in ms).
    segments - list of tuples (start, end, segment), sorted by start time.
    Returns a list of merged segments.
    """
    if not segments:
        return []
    merged = []
    current_start, current_end, current_seg = segments[0]
    for seg in segments[1:]:
        start, end, seg_audio = seg
        if start - current_end <= merge_gap:
            current_end = end
            current_seg += seg_audio  # merging audio using pydub
        else:
            merged.append((current_start, current_end, current_seg))
            current_start, current_end, current_seg = seg
    merged.append((current_start, current_end, current_seg))
    return merged

def save_error_segment(segment, participant, start_ms, base_output_dir):
    """
    Saves a problematic segment for debugging.
    The segment is saved in the error_segments folder inside base_output_dir,
    and the filename contains the participant name and segment start time.
    """
    error_dir = os.path.join(base_output_dir, "error_segments")
    os.makedirs(error_dir, exist_ok=True)
    filename = os.path.join(error_dir, f"{participant}_{start_ms}.wav")
    try:
        segment.export(filename, format="wav")
        tqdm.write(f"Saved problematic segment for {participant} to {filename}")
    except Exception as e:
        tqdm.write(f"Failed to save problematic segment: {e}")

def transcribe_segment(segment, cli_path, model_path, language, extra_params):
    """
    Saves audio segment to a temporary WAV file and calls whisper-cli for its transcription.
    The extra_params parameter is a list of additional parameters (e.g., temperature, beam_size, etc.).
    The temporary file is deleted after completion.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            temp_filename = tmp_file.name
            segment.export(temp_filename, format="wav")
    except Exception as e:
        logging.error(f"Error when saving temporary file: {e}")
        return ""
    
    cmd = [cli_path, "-m", model_path, "-f", temp_filename, "-l", language, "-np", "-nt"] + extra_params
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        transcription = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Error when transcribing segment: {e}")
        transcription = ""
    finally:
        try:
            os.remove(temp_filename)
        except Exception as e:
            logging.warning(f"Failed to delete temporary file {temp_filename}: {e}")
    return transcription

def transcribe_full_file(audio_file, cli_path, model_path, language, extra_params):
    """
    Transcribes the audio file as a whole without segmentation.
    Used when the --no-segmentation flag is set.
    """
    cmd = [cli_path, "-m", model_path, "-f", audio_file, "-l", language, "-np", "-nt"] + extra_params
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        transcription = result.stdout.strip()
        return transcription
    except subprocess.CalledProcessError as e:
        logging.error(f"Error when transcribing full file: {e}")
        return ""

def process_channel(channel_audio, silence_min_len, silence_thresh, description="Processing channel"):
    """
    Identifies speech sections in an audio channel.
    Returns a list of tuples (start, end, segment) for non-silent segments.
    Displays processing progress using tqdm.
    """
    if silence_thresh is None:
        silence_thresh = channel_audio.dBFS - 16
    
    # Split audio into blocks for progress display
    block_size = 30000  # 30 seconds in ms
    total_length_ms = len(channel_audio)
    num_blocks = math.ceil(total_length_ms / block_size)
    
    nonsilent_ranges = []
    
    with tqdm(total=num_blocks, desc=description, dynamic_ncols=True) as pbar:
        for block_idx in range(num_blocks):
            start_ms = block_idx * block_size
            end_ms = min(start_ms + block_size, total_length_ms)
            block_audio = channel_audio[start_ms:end_ms]
            
            # Offset for the block
            offset_ms = start_ms
            
            # Identifying non-silent ranges in the current block
            block_ranges = silence.detect_nonsilent(
                block_audio, 
                min_silence_len=silence_min_len, 
                silence_thresh=silence_thresh
            )
            
            # Adjusting ranges considering the offset
            for start, end in block_ranges:
                nonsilent_ranges.append((start + offset_ms, end + offset_ms))
            
            pbar.update(1)
    
    # Merging adjacent ranges
    if not nonsilent_ranges:
        return []
    
    merged_ranges = [nonsilent_ranges[0]]
    for curr_start, curr_end in nonsilent_ranges[1:]:
        prev_start, prev_end = merged_ranges[-1]
        if curr_start - prev_end < silence_min_len:  # Minimum gap for merging
            merged_ranges[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged_ranges.append((curr_start, curr_end))
    
    # Creating audio segments for each range
    segments = []
    for start, end in merged_ranges:
        seg = channel_audio[start:end]
        segments.append((start, end, seg))
    
    return segments

def process_audio(audio, cli_path, model_path, language, extra_params, 
                 silence_min_len, silence_thresh, merge_gap, num_threads,
                 save_errors=False, include_errors=False, output_dir=".",
                 disable_segmentation=False, mp3_file=None):
    """
    Unified function for audio processing (both mono and stereo).
    Performs segmentation, parallel transcription, and result merging.
    
    If disable_segmentation=True, then transcribes the file as a whole without segmentation.
    
    Returns a string with complete transcription and error count.
    """
    error_count = 0
    
    # If segmentation is disabled and the file is mono, process it as a whole
    if disable_segmentation and audio.channels == 1:
        logging.info("Segmentation disabled. Transcribing mono file as a whole.")
        transcription = transcribe_full_file(mp3_file, cli_path, model_path, language, extra_params)
        return transcription, 0 if transcription else 1
    
    # If segmentation is disabled and the file is stereo, we still need to process channels separately
    # But we'll process each channel as a whole
    if disable_segmentation and audio.channels == 2:
        logging.info("Segmentation disabled. Processing each stereo file channel as a whole.")
        channels = audio.split_to_mono()
        transcript_parts = []
        
        for idx, channel in enumerate(channels):
            participant = f"Participant {idx+1}"
            logging.info(f"Processing channel for {participant}.")
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_filename = tmp_file.name
                channel.export(temp_filename, format="wav")
            
            try:
                text = transcribe_full_file(temp_filename, cli_path, model_path, language, extra_params)
                if not text:
                    error_count += 1
                    if include_errors:
                        transcript_parts.append(f"[{participant}] — <Transcription Error>")
                else:
                    transcript_parts.append(f"[{participant}] — {text}")
            finally:
                try:
                    os.remove(temp_filename)
                except Exception as e:
                    logging.warning(f"Failed to delete temporary file {temp_filename}: {e}")
        
        transcript = "\n".join([line for line in transcript_parts if line])
        return transcript, error_count
    
    # If we're here, then segmentation is enabled - continue with normal processing with segmentation
    all_segments = []
    
    if audio.channels == 1:
        # Mono: process one channel
        logging.info("Performing mono file segmentation.")
        segs = process_channel(audio, silence_min_len, silence_thresh, "Segmenting mono file")
        merged_segs = merge_segments(segs, merge_gap)
        for start_seg, end_seg, seg_audio in merged_segs:
            all_segments.append((start_seg, "Speaker", seg_audio))
    elif audio.channels == 2:
        # Stereo: split channels and process them
        logging.info("Performing channel separation and stereo file segmentation.")
        channels = audio.split_to_mono()
        for idx, channel in enumerate(channels):
            participant = f"Participant {idx+1}"
            segs = process_channel(channel, silence_min_len, silence_thresh, f"Segmenting channel {idx+1}")
            merged_segs = merge_segments(segs, merge_gap)
            for start_seg, end_seg, seg_audio in merged_segs:
                all_segments.append((start_seg, participant, seg_audio))
    else:
        logging.error("Unsupported number of audio channels.")
        return "", error_count
    
    # If no segments, return empty string
    if not all_segments:
        logging.warning("No speech segments detected.")
        return "", 0
    
    # Sort segments by start time for correct chronology
    all_segments.sort(key=lambda x: x[0])
    
    logging.info(f"Total of {len(all_segments)} segments identified for transcription.")
    
    transcript_lines = [None] * len(all_segments)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {}
        for idx, (start_seg, participant, seg_audio) in enumerate(all_segments):
            future = executor.submit(transcribe_segment, seg_audio, cli_path, model_path, language, extra_params)
            futures[future] = (idx, participant, start_seg, seg_audio)
        
        with tqdm(total=len(futures), desc="Transcribing segments", dynamic_ncols=True) as pbar:
            for future in concurrent.futures.as_completed(futures):
                idx, participant, start_seg, seg_audio = futures[future]
                try:
                    text = future.result()
                except Exception as e:
                    tqdm.write(f"Error in transcription thread for {participant} (start: {start_seg} ms): {e}")
                    text = ""
                if not text:
                    error_count += 1
                    if save_errors:
                        save_error_segment(seg_audio, participant, start_seg, output_dir)
                    if include_errors:
                        transcript_lines[idx] = f"[{participant}] — <Transcription Error>"
                    else:
                        transcript_lines[idx] = None
                else:
                    transcript_lines[idx] = f"[{participant}] — {text}"
                pbar.update(1)
    
    transcript = "\n".join([line for line in transcript_lines if line])
    return transcript, error_count

def main():
    # Define path to configuration file (file should be in the same directory as the script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.ini")
    config = read_config(config_path)
    
    # Read path parameters from [paths] section
    default_cli = config.get("paths", "whisper_cli", fallback="./build/bin/whisper-cli")
    default_model = config.get("paths", "model", fallback="")
    
    # Read segmentation and language parameters from [settings] section
    silence_min_len = config.getint("settings", "silence_min_len", fallback=500)
    silence_thresh_config = config.get("settings", "silence_thresh", fallback="")
    silence_thresh = float(silence_thresh_config) if silence_thresh_config else None
    merge_gap = config.getint("settings", "merge_gap", fallback=1000)
    language = config.get("settings", "language", fallback="ru")
    num_threads = config.getint("settings", "threads", fallback=4)
    
    # Read parameters for transcription quality
    temperature = config.getfloat("settings", "temperature", fallback=0.0)
    temperature_inc = config.getfloat("settings", "temperature_inc", fallback=0.2)
    best_of = config.getint("settings", "best_of", fallback=5)
    beam_size = config.getint("settings", "beam_size", fallback=5)
    
    # Form a list of additional parameters for whisper-cli
    extra_params = ["-tp", str(temperature),
                    "-tpi", str(temperature_inc),
                    "-bo", str(best_of),
                    "-bs", str(beam_size)]
    
    parser = argparse.ArgumentParser(description="Meeting transcription using whisper.cpp (optimized)")
    parser.add_argument("mp3_file", help="Path to mp3 file")
    parser.add_argument("--cli", help="Path to whisper-cli (if not specified, taken from config.ini)", default=default_cli)
    parser.add_argument("--model", help="Path to whisper model (if not specified, taken from config.ini)", default=default_model)
    # Flags for error handling
    parser.add_argument("--include-errors", action="store_true",
                        help="Include lines with transcription errors in the final txt (by default errors are not included)")
    parser.add_argument("--save-errors", action="store_true",
                        help="Save problematic segments in the error_segments folder for listening")
    # Flag to disable segmentation
    parser.add_argument("--no-segmentation", action="store_true",
                        help="Disable audio segmentation and process the file as a whole (may be slower for long files)")
    args = parser.parse_args()

    mp3_file = args.mp3_file
    cli_path = args.cli
    model_path = args.model
    include_errors = args.include_errors
    save_errors = args.save_errors
    disable_segmentation = args.no_segmentation

    # Check availability of whisper-cli, model file, and audio
    check_whisper_cli(cli_path)
    check_model_file(model_path)
    audio = check_audio_file(mp3_file)

    logging.info(f"File {mp3_file} successfully loaded. Channels: {audio.channels}, duration: {len(audio)/1000:.2f} sec.")
    
    if disable_segmentation:
        logging.info("Flag --no-segmentation set. Audio segmentation disabled.")

    base_name = os.path.splitext(os.path.basename(mp3_file))[0]
    output_file = os.path.join(os.path.dirname(mp3_file), base_name + ".txt")
    output_dir = os.path.dirname(mp3_file)

    start_time = time.time()
    
    config = {
        "cli_path": cli_path,
        "model_path": model_path,
        "language": language,
        "extra_params": extra_params,
        "silence_min_len": silence_min_len,
        "silence_thresh": silence_thresh,
        "merge_gap": merge_gap,
        "num_threads": num_threads,
        "save_errors": save_errors,
        "include_errors": include_errors,
        "output_dir": output_dir,
        "disable_segmentation": disable_segmentation,
        "mp3_file": mp3_file,
    }

    # Processing audio with parameters from config
    transcript, error_count = process_audio(audio, **config)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcript)
        logging.info(f"Transcription saved to file {output_file}")
    except Exception as e:
        logging.error(f"Error writing transcription to file: {e}")
        sys.exit(1)

    total_time = time.time() - start_time
    tqdm.write(f"Total processing speed: {total_time:.2f} seconds")
    tqdm.write(f"Total number of unrecognized segments: {error_count}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user. Exiting.")
        sys.exit(1)
    except Exception:
        logging.exception("Unexpected error:")
        sys.exit(1)