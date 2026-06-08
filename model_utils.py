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

    img = np.array(image.convert("RGB")).astype("float32")
    gray = img.mean(axis=2)

    channel_diff = (
        np.abs(img[:, :, 0] - img[:, :, 1]).mean()
        + np.abs(img[:, :, 1] - img[:, :, 2]).mean()
        + np.abs(img[:, :, 0] - img[:, :, 2]).mean()
    ) / 3
    if channel_diff > 12:
        return False

    gray_std = gray.std()
    if gray_std < 18 or gray_std > 95:
        return False

    height, width = gray.shape
    border = max(1, min(height, width) // 12)
    border_pixels = np.concatenate(
        [
            gray[:border, :].ravel(),
            gray[-border:, :].ravel(),
            gray[:, :border].ravel(),
            gray[:, -border:].ravel(),
        ]
    )

    dark_fraction = np.mean(gray < 35)
    bright_fraction = np.mean(gray > 80)
    border_dark_fraction = np.mean(border_pixels < 35)

    center = gray[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    center_is_brighter = center.mean() > border_pixels.mean() + 8

    score = 0
    score += dark_fraction >= 0.15
    score += 0.04 <= bright_fraction <= 0.75
    score += border_dark_fraction >= 0.35
    score += center_is_brighter

    return score >= 3


def predict_image(model, image, device, conf_threshold=0.6):
    if not is_mri_like(image):
        return "invalid_image", None

    transform = get_transform()
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)[0]

    max_prob = torch.max(probs).item()
    pred = torch.argmax(probs).item()

    if max_prob < conf_threshold:
        return "low_confidence", probs.cpu().numpy()

    return pred, probs.cpu().numpy()
