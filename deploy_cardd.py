import gradio as gr
import torch
from torch import nn
from torchvision import models, transforms

model = models.vgg16_bn(weights="DEFAULT")

model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 4))

new_classifier = torch.nn.Sequential(
    nn.Linear(8192, 4096),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(4096, 5),
)


model.classifier[0].in_features = 8192
model.classifier = new_classifier


model.load_state_dict(torch.load("VGG_Adamax.pt"))
model.eval()


def preprocess(image):
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4992, 0.4839, 0.4827], std=[0.2325, 0.2332, 0.2327]
            ),
        ]
    )

    image = transform(image)
    return image


labels = ["dent", "glass shatter", "lamp broken", "scratch", "tire flat"]


def predict(image):
    try:
        image = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            prediction = torch.nn.functional.softmax(model(image)[0], dim=0)
        return {labels[i]: float(prediction[i]) for i in range(5)}
    except Exception as e:
        print(f"Error predicting image: {e}")
        return []


title = "Car Damage Detection"

gr.Interface(
    title=title,
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    interpretation="default",
).launch(share=True)
