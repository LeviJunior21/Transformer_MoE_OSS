import wandb, torch, os


def fetch_weights_and_bias(user, project, name, version, file_name):
    try:
        api = wandb.Api()
        artifact = api.artifact(f"{user}/{project}/{name}:{version}", type="model")
        artifact_dir = artifact.download()
        file_path = os.path.join(artifact_dir, file_name)
        print(f"Fetch success -> {file_path}")
        return True

    except Exception as e:
        print(f"Fetch Error: {e}")
        return False


def load_weights_and_bias(file_name):
    try:
        checkpoint = torch.load(file_name, map_location="cpu")
        return True, checkpoint
    except Exception as e:
        print(f"Load error: {e}")
        return False, {}