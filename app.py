import streamlit as st
from fastai.vision.all import *
from PIL import Image

# 设置页面基本信息
st.set_page_config(
    page_title="运动鞋智能分类器",
    page_icon="👟",
    layout="centered"
)

# 兼容 fastai 载入导出模型时对自定义 splitter 的依赖
def my_splitter(items):
    return [], []

# 缓存加载模型，避免每次刷新页面都重新加载
@st.cache_resource
def load_my_model():
    try:
        # 这里使用导出的 pkl 模型名称
        return load_learner('shoe_classifier.pkl')
    except Exception as e:
        st.error(f"模型加载失败，请确保 'shoe_classifier.pkl' 存在于当前目录。错误信息: {e}")
        return None

learn = load_my_model()
categories = ('板鞋', '篮球鞋', '足球鞋')

# 页面排版与视觉设计
st.title("👟 运动鞋类别智能分类器")
st.write("欢迎使用！本应用由 **Streamlit** 驱动。您可以上传一张鞋子的照片，系统将自动识别它是 **板鞋**、**篮球鞋** 还是 **足球鞋**。")

st.markdown("---")

# 上传组件
uploaded_file = st.file_uploader("请选择一张鞋子的照片 (支持 JPG, JPEG, PNG 格式)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 读入并展示图片
    image = Image.open(uploaded_file)
    
    # 左右两栏布局，左边显示图，右边显示预测结果
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="已上传的图片", use_container_width=True)
        
    with col2:
        if learn is not None:
            # 开启加载动画
            with st.spinner("模型正在加速分析中..."):
                # 将 PIL 图像转换为 fastai 的 PILImage
                fastai_img = PILImage.create(uploaded_file)
                pred, pred_idx, probs = learn.predict(fastai_img)
                
            # 显示预测的最终结果
            st.success(f"**预测结果：{pred}**")
            st.metric(label="置信度", value=f"{probs[pred_idx]*100:.2f}%")
            
            st.write("各类别概率分布：")
            # 组合成 pandas DataFrame 方便 Streamlit 绘制柱状图
            import pandas as pd
            chart_data = pd.DataFrame({
                '置信度': [float(p) for p in probs]
            }, index=categories)
            
            # 显示条形图
            st.bar_chart(chart_data)
        else:
            st.error("模型未加载，无法预测。")
            
st.markdown("---")
st.caption("🔍 本项目基于 FastAI + ResNet18 微调训练，通过 Streamlit 进行 UI 交互与云端在线部署。")
