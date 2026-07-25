from pathlib import Path
import torchaudio
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class ASVspoofDataset(BaseDataset):
    def __init__(
        self, protocol_path, audio_dir, name="train", max_len=64000, *args, **kwargs
    ):
        self.max_len = max_len

        index_path = ROOT_PATH / "data" / "asvspoof" / f"{name}_index.json"
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(protocol_path, audio_dir, index_path)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, protocol_path, audio_dir, index_path):
        index = []
        audio_dir = Path(audio_dir)
        with open(protocol_path) as f:
            for line in f:
                _, utt_id, _, _, label = line.strip().split()
                path = (audio_dir / f"{utt_id}.flac").as_posix()
                index.append(
                    {
                        "path": path,
                        "label": 1 if label == "bonafide" else 0,
                        "utt_id": utt_id,
                    }
                )

        index_path.parent.mkdir(exist_ok=True, parents=True)
        write_json(index, str(index_path))
        return index

    def load_object(self, path):
        wav, sr = torchaudio.load(path)
        wav = wav.squeeze(0)

        if wav.shape[0] > self.max_len:
            wav = wav[: self.max_len]
        elif wav.shape[0] < self.max_len:
            num_repeats = self.max_len // wav.shape[0] + 1
            wav = wav.repeat(num_repeats)[: self.max_len]
        return wav
