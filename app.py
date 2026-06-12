import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import resnet18
from torchvision.transforms import functional as TF


APP_DIR = Path(__file__).resolve().parent
MODEL_STATE_PATH = APP_DIR / "shoe_model_state.pth"
CLASSES_PATH = APP_DIR / "classes.json"
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=1):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(size)
        self.mp = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], dim=1)


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)


def build_model(num_classes):
    backbone = resnet18(weights=None)
    body = nn.Sequential(*list(backbone.children())[:-2])
    head = nn.Sequential(
        AdaptiveConcatPool2d(),
        Flatten(),
        nn.BatchNorm1d(1024),
        nn.Dropout(p=0.25),
        nn.Linear(1024, 512, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes, bias=False),
    )
    return nn.Sequential(body, head)


def load_state_dict_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


@st.cache_resource
def load_runtime_model():
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    model = build_model(len(classes))
    state_dict = load_state_dict_cpu(MODEL_STATE_PATH)
    model.load_state_dict(state_dict)
    model.eval()
    return model, classes


def center_crop_resize(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)


def preprocess_image(image):
    image = center_crop_resize(image)
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor.unsqueeze(0)


def predict(model, image):
    batch = preprocess_image(image)
    with torch.inference_mode():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)[0].cpu()
    pred_idx = int(torch.argmax(probs))
    return pred_idx, probs


st.set_page_config(
    page_title="运动鞋智能分类器",
    page_icon="👟",
    layout="centered",
)

try:
    model, categories = load_runtime_model()
    load_error = None
except Exception as exc:
    model, categories = None, []
    load_error = exc

if load_error is not None:
    st.error(f"模型加载失败：{type(load_error).__name__}: {load_error}")

st.title("👟 运动鞋类别智能分类器")
st.write(
    "欢迎使用！本应用由 **Streamlit** 驱动。您可以上传一张鞋子的照片，"
    "系统将自动识别它是 **板鞋**、**篮球鞋** 还是 **足球鞋**。"
)

st.markdown("---")

uploaded_file = st.file_uploader(
    "请选择一张鞋子的照片（支持 JPG, JPEG, PNG 格式）...",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="已上传的图片", width="stretch")

    with col2:
        if model is None:
            st.error("模型未加载，无法预测。")
        else:
            try:
                with st.spinner("模型正在分析中..."):
                    pred_idx, probs = predict(model, image)
                    pred = categories[pred_idx]

                st.success(f"**预测结果：{pred}**")
                st.metric(label="置信度", value=f"{float(probs[pred_idx]) * 100:.2f}%")

                chart_data = pd.DataFrame(
                    {"置信度": [float(prob) for prob in probs]},
                    index=categories,
                )
                st.write("各类别概率分布：")
                st.bar_chart(chart_data)
            except Exception as exc:
                st.error(f"预测失败：{type(exc).__name__}: {exc}")

st.markdown("---")
st.caption("🔍 本项目基于 PyTorch + ResNet18 微调训练，通过 Streamlit 进行 UI 交互与云端在线部署。")
