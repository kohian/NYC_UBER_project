import torch
import gcsfs
########################## Save Load Torch state ##############################################################################################################
def save_torch_state_dict_to_gcs(
    state_dict: dict,
    base_path: str,
    filename: str = "best_model.pt",
) -> str:
    """
    Save a PyTorch state_dict to GCS.
    """
    fs = gcsfs.GCSFileSystem()
    path = f"{base_path.rstrip('/')}/{filename}"

    with fs.open(path, "wb") as f:
        torch.save(state_dict, f)

    print(f"Saved model state_dict to {path}")
    return path


def load_torch_state_dict_from_gcs(path: str, map_location: str | None = None) -> dict:
    """
    Load a PyTorch state_dict from GCS.
    """
    fs = gcsfs.GCSFileSystem()
    with fs.open(path, "rb") as f:
        return torch.load(f, map_location=map_location)