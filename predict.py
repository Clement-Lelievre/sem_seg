import numpy as np
import torch
from cog import BasePredictor, Input, Path
from PIL import Image
from torchvision import models, transforms

COLORS = [
    (155, 155, 155),  # unlabelled
    (60, 16, 152),  # building
    (132, 41, 246),  # land
    (110, 193, 228),  # road
    (254, 221, 58),  # vegetation
    (226, 169, 41),  # boat
]


class Predictor(BasePredictor):

    def setup(self) -> None:
        # load_the_model
        self.model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=6)
        # model.classifier[4] = nn.Conv2d(512, 6, kernel_size=1)  # Change output to 6 classes
        self.model.load_state_dict(torch.load("sem_seg_model.pth"), strict=False)
        # torch.load with map_location=torch.device('cpu') on CPU machines
        self.cuda_available = torch.cuda.is_available()
        self.model.to("cuda" if self.cuda_available else "cpu")
        self.model.eval()
        print("Model loaded")

    def predict(  # noqa: PLR0915 C901
        self,
        image: Path = Input(
            description="Aerial image to predict on",
        ),
    ) -> Path:
        print("Predicting on {'cpu' if not self.cuda_available else 'cuda'}")
        img = self.preprocess_image(image)
        result = self.infer(img)
        inference: Image.Image = self.inference_to_encoded_img(result)
        inference.save("inference.png")
        return Path("inference.png")

    def preprocess_image(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to("cuda" if self.cuda_available else "cpu")
        return image

    @property
    def transform(self):
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return transform

    @torch.no_grad()
    def infer(self, image_tensor):
        output = self.model(image_tensor)["out"]  # Get the output of the FCN model
        output = (
            torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        )  # Get the predicted class for each pixel
        return output

    def inference_to_encoded_img(self, inference: np.ndarray) -> Image.Image:
        # Create an RGB image from the predicted classes
        h, w = inference.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
        for i, color in enumerate(COLORS):
            rgb_image[inference == i] = color
        return Image.fromarray(rgb_image)
