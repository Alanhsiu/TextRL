import random
import numpy as np
import torch
import torchaudio
from datasets import Dataset, DatasetDict
from encodec import EncodecModel
from encodec.utils import convert_audio
from argparse import ArgumentParser, Namespace
from time import time
import shutil
import os
import json
import librosa
import random
from functools import partial
from pathlib import Path
from time import time
import tqdm



def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def get_new_arrow_filename(target_dir, base_filename):
    """
    Generates a new filename by appending a numerical suffix to avoid overwriting existing files.
    """
    counter = 1  # Start with _1 for the first duplicate
    filename, extension = os.path.splitext(base_filename)
    # Generate filename with incremented counter until a new, unused name is found
    while True:
        new_filename = f"{filename}_{counter}{extension}"
        if not os.path.exists(os.path.join(target_dir, new_filename)):
            break
        counter += 1
    return new_filename

def move_and_update_state(base_dir, target_dir):
    """
    Moves .arrow files with unique names and conditionally moves or updates state.json.
    """
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Initialize variables
    arrow_files_moved = []
    state_json_path = os.path.join(base_dir, 'state.json')
    
    # Move and rename arrow files
    for filename in os.listdir(base_dir):
        if filename.endswith('.arrow'):
            new_filename = get_new_arrow_filename(target_dir, filename)
            src_path = os.path.join(base_dir, filename)
            dst_path = os.path.join(target_dir, new_filename)
            shutil.move(src_path, dst_path)
            arrow_files_moved.append({"filename": new_filename})
    
    # Check and handle state.json
    target_state_json_path = os.path.join(target_dir, 'state.json')
    if not os.path.exists(target_state_json_path):
        # If state.json doesn't exist in target, move and update it
        if os.path.exists(state_json_path):
            if arrow_files_moved:
                with open(state_json_path, 'r') as f:
                    state_data = json.load(f)
                    state_data["_data_files"] = arrow_files_moved
                with open(state_json_path, 'w') as f:
                    json.dump(state_data, f, indent=2)
            shutil.move(state_json_path, target_dir)
    else:
        # If state.json exists, update it
        with open(target_state_json_path, 'r+') as f:
            state_data = json.load(f)
            # Append new arrow files
            state_data["_data_files"] = arrow_files_moved + state_data["_data_files"]
            # Remove duplicate dictionaries in "_data_files" list
            state_data["_data_files"] = [i for n, i in enumerate(state_data["_data_files"]) if i not in state_data["_data_files"][n + 1:]]
            # Move the pointer to the beginning of the file and truncate it
            f.seek(0)
            json.dump(state_data, f, indent=2)
            f.truncate()



def process_audio(source_audio_path, output_dir, temp_dir, instruction= None, transcription=None, seed=0, device="cuda"):
    
    set_seed(seed)
    device = torch.device(device)



    # Setup model
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.to(device)

    # Initialize dataset
    dataset = {
        "file_id": ["single_file_1"],
        "instruction": [instruction],
        "transcription": [transcription]
    }

    for i in range(8):
        dataset[f"src_encodec_{i}"] = []

    start = time()

    # Process source audio file
    wav, sr = torchaudio.load(source_audio_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels).unsqueeze(0)

    with torch.no_grad():
        wav = wav.to(device)
        encoded_frames = model.encode(wav)

    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).detach().cpu().squeeze(0).numpy()
    for i in range(8):
        dataset[f"src_encodec_{i}"].append(codes[i])

    print(f"[INFO] It took {time() - start} seconds to process the file.")

    # Create and save dataset
    dataset = Dataset.from_dict(dataset)
    dataset.save_to_disk(temp_dir)

    move_and_update_state(temp_dir, output_dir)

def audio_path_from_id(audio_id, dir):
    return Path(dir) / f"{audio_id}.wav"

def get_instruction(audio_id, instruction_dir):
    instruction_path = Path(instruction_dir) / f"{audio_id}.txt"
    with open(instruction_path, "r") as f:
        instruction = f.read()
    return instruction

def get_transcription(audio_id, transcription_dir):
    transcription_path = Path(transcription_dir) / f"{audio_id}.txt"
    with open(transcription_path, "r") as f:
        transcription = f.read()
    return transcription

def main(args):
    set_seed(args.seed)
    device = args.device

    # Setup dataset
    dataset_dict = {}

    # Setup model and codebook
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.to(device)

    for split in args.splits:
        print(f"[INFO] Process {split.upper()} split.")

        # Setup dataset
        dataset = {"file_id": [], "instruction": [], "transcription": [],}
        for i in range(8):
            dataset[f"src_encodec_{i}"] = []
        for i in range(8):
            dataset[f"tgt_encodec_{i}"] = []
        print(dataset)
            
        instruction_dir = f"{args.data_dir}/{split}/instruction"
        transcription_dir = f"{args.data_dir}/{split}/transcription"
        source_audio_dir = f"{args.data_dir}/{split}/source"
        target_audio_dir = f"{args.data_dir}/{split}/target"
        source_audios = librosa.util.find_files(source_audio_dir, ext=["wav"])
        target_audios = librosa.util.find_files(target_audio_dir, ext=["wav"])
        file_ids = [Path(audio).stem for audio in source_audios]
        assert len(source_audios) == len(target_audios)
        assert len(source_audios) > 0
        assert len(target_audios) > 0
        assert len(source_audios) == len(file_ids)

        source_audio_path = partial(audio_path_from_id, dir=source_audio_dir)
        target_audio_path = partial(audio_path_from_id, dir=target_audio_dir)

        print(f"[INFO] There are {len(file_ids)} files to be processed.")
        start = time()
        for idx, file_id in enumerate(tqdm(file_ids, desc="Converting", ascii=False, ncols=100)):
            instruction = get_instruction(file_id, instruction_dir)
            transcription = get_transcription(file_id, transcription_dir)
            src_path = source_audio_path(file_id)
            tgt_path = target_audio_path(file_id)

            wav, sr = torchaudio.load(src_path)
            wav = convert_audio(wav, sr, model.sample_rate, model.channels).unsqueeze(0)
    
            # Extract discrete codes from EnCodec
            with torch.no_grad():
                wav = wav.to(device)
                encoded_frames = model.encode(wav)
    
            codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
            src_code = list(codes.detach().cpu().squeeze(0).numpy())

            wav, sr = torchaudio.load(tgt_path)
            wav = convert_audio(wav, sr, model.sample_rate, model.channels).unsqueeze(0)
    
            # Extract discrete codes from EnCodec
            with torch.no_grad():
                wav = wav.to(device)
                encoded_frames = model.encode(wav)
    
            codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
            tgt_code = list(codes.detach().cpu().squeeze(0).numpy())
            
            dataset["file_id"].append(file_id)
            dataset["instruction"].append(instruction)
            dataset["transcription"].append(transcription)
            for i in range(8):
                dataset[f"src_encodec_{i}"].append(src_code[i])
                dataset[f"tgt_encodec_{i}"].append(tgt_code[i])

        print(f"[INFO] It takes {time() - start} seconds to process all files.")

        dataset = Dataset.from_dict(dataset)
        dataset_dict[split] = dataset

    Soxdataset = DatasetDict(dataset_dict)
    
    Soxdataset.save_to_disk(args.output_dir)


# Example usage
if __name__ == "__main__":
    source_audio_path = "/work/b0990106x/TextRL/output/0.wav"
    output_dir = "./data/soxdata_encodec/" #base
    #instruction = "Example instruction text"
    #transcription = "Example transcription text"
    seed = 0
    device = "cuda"
    target_dir = "./data/final"  # Final directory to move files to
    
    process_audio(source_audio_path, target_dir, output_dir, seed, device)
