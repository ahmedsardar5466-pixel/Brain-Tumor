import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']


def load_model(model_path, device):
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])


def is_mri_like(image):
    import numpy as np
    img = np.array(image)
    return img.std() < 80


def predict_image(model, image, device, glioma_threshold=None, conf_threshold=0.6):
    transform = get_transform()
    img = transform(image).unsqueeze(0).to(device)

    if not is_mri_like(image):
        return "invalid_image", None

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)[0]

    max_prob = torch.max(probs).item()
    pred = torch.argmax(probs).item()

    if max_prob < conf_threshold:
        return "low_confidence", probs.cpu().numpy()

    if glioma_threshold is not None and probs[0].item() > glioma_threshold:
        pred = 0

    return pred, probs.cpu().numpy()