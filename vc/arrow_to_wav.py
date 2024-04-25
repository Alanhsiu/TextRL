import torch
from datasets import load_from_disk
from encodec import EncodecModel
import torchaudio
from pathlib import Path
from tqdm import tqdm

def decode_audio(encoded_frames, model, device):
    ''' Decode discrete codes to waveform using EnCodec. '''
    decoded_audio = []
    with torch.no_grad():
        for frame in encoded_frames:
            # Assuming each frame does not need to be unpacked into codes and scale
            # And that model.decode() can handle them directly if put in the right format
            frame_audio = model.decode(torch.tensor([frame]).to(device))
            decoded_audio.append(frame_audio)
    # Assuming you need to concatenate decoded frames
    decoded_audio = torch.cat(decoded_audio, dim=1)
    return decoded_audio.squeeze(0).cpu()


def main(args):
    device = args.device

    # Load dataset
    dataset = load_from_disk(args.input_dir)

    # Load model
    model = EncodecModel.encodec_model_24khz()
    model.to(device)

    # Ensure output directories exist
    output_dirs = [args.source_output_dir, args.target_output_dir]
    for output_dir in output_dirs:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("[INFO] Processing dataset.")
    for i in tqdm(range(len(dataset)), desc="Decoding", ascii=False, ncols=100):
        file_id = dataset["file_id"][i]
        
        # Source decoding
        src_encoded = [dataset[f"src_encodec_{j}"][i] for j in range(8)]
        src_wav = decode_audio(src_encoded, model, device)
        torchaudio.save(Path(args.source_output_dir) / f"{file_id}.wav", src_wav.unsqueeze(0), model.sample_rate)
        
        # Target decoding
        tgt_encoded = [dataset[f"tgt_encodec_{j}"][i] for j in range(8)]
        tgt_wav = decode_audio(tgt_encoded, model, device)
        torchaudio.save(Path(args.target_output_dir) / f"{file_id}.wav", tgt_wav.unsqueeze(0), model.sample_rate)

        # Additional processing can be done here (e.g., handling text data)

    print("[INFO] Completed processing dataset.")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./data/init_arrow")
    parser.add_argument("--source_output_dir", type=str, default="./data/final_source")
    parser.add_argument("--target_output_dir", type=str, default="./data/final_target")
    parser.add_argument("--device", type=torch.device, default="cuda")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
