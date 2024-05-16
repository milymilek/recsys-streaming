import io
import torch

def _get_latest_model_version_document(client):
    latest_version = client['model_versions'].find_one({}, sort=[("timestamp", -1)])
    return latest_version

def _get_model_buffer(model_version_document):
    #return model_version["binary"]
    return io.BytesIO(model_version_document["binary"])

def load_model(client):
    latest_version_document = _get_latest_model_version_document(client=client)
    model_buffer = _get_model_buffer(model_version_document=latest_version_document)

    # Assuming 'path' is the buffer containing the model's binary data
    print(f'Loading model: {latest_version_document["model"]}:v{latest_version_document["version"]}-{latest_version_document["timestamp"]}')
    model = torch.jit.load(model_buffer, map_location='cpu')
    return model


def build_input_tensor(user_id: int, item_id: int) -> torch.Tensor:
    return torch.tensor([user_id, item_id, 0], dtype=torch.long).view(1, -1)