import streamlit as st
# ---------------------- 防休眠心跳接口（必须加） ----------------------
if st.query_params.get("keep_alive") == "true":
    st.write("✅ 心跳正常 - 应用保持活跃")
    st.stop()
import os
import sys
# 强制绕过 libGL 错误
os.environ["OPENCV_VIDEOIO_PRIORITY"] = "0"
os.environ["CV2_OPENCV_HEADLESS"] = "1"
import torch
import torch.nn as nn
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import warnings
import base64
from io import BytesIO
import os
import glob
import time

warnings.filterwarnings("ignore")

# 页面配置
st.set_page_config(
    page_title="InsurEye | 智能定损系统",
    layout="wide",
    page_icon="icon.png",
    initial_sidebar_state="collapsed"
)

# 路由监听
query_params = st.query_params
if "page" in query_params and query_params["page"] == "architecture":
    st.session_state.page_state = "architecture"

# 会话状态初始化
if "first_init" not in st.session_state:
    st.session_state.first_init = True
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "page_state" not in st.session_state:
    st.session_state.page_state = "home"

page_loader = st.empty()

# 全局加载动画样式
GLOBAL_LOADER_STYLE = """
<style>
@keyframes rotateRing {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
@keyframes reverseRing {
    0% { transform: rotate(360deg); }
    100% { transform: rotate(0deg); }
}
@keyframes pulseGlow {
    0%,100% { opacity: 0.6; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.08); }
}
@keyframes floatCar {
    0%,100% { transform: translate(-50%,-50%) translateY(0px); }
    50% { transform: translate(-50%,-50%) translateY(-12px); }
}
@keyframes magnifierScan {
    0% { transform: translate(-50%,-50%) scale(0.85); opacity: 0.25; }
    50% { transform: translate(-50%,-50%) scale(1.25); opacity: 0.85; }
    100% { transform: translate(-50%,-50%) scale(0.85); opacity: 0.25; }
}
@keyframes radarSweep {
    0% { transform: rotate(0deg); opacity: 0.15; }
    100% { transform: rotate(360deg); opacity: 0.6; }
}
@keyframes streamLight {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

.loader-mask {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: #030712;
    z-index: 999999;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    overflow: hidden;
}
.loader-mask::before{
    content:"";
    position:absolute;
    width:200%;
    height:200%;
    background: linear-gradient(90deg,transparent,rgba(0,230,255,0.18),transparent,rgba(0,180,255,0.08),transparent);
    background-size:200% 100%;
    animation:streamLight 3.5s linear infinite;
}
.tech-ring-wrap{
    position:relative;
    width:180px;
    height:180px;
}
.tech-ring-outer {
    width: 180px;
    height: 180px;
    position: absolute;
    top:0;left:0;
    animation: rotateRing 2.8s linear infinite;
}
.tech-ring-inner {
    width: 180px;
    height: 180px;
    position: absolute;
    top:0;left:0;
    animation: reverseRing 3.8s linear infinite;
}
.tech-ring circle {
    fill: none;
    stroke-width: 2;
    stroke-linecap: round;
}
.ring-bg { stroke: rgba(0,230,255,0.1); }
.ring-run { 
    stroke: #00e6ff; 
    stroke-dasharray: 330; 
    stroke-dashoffset: 60;
    filter: drop-shadow(0 0 10px #00e6ff);
}
.radar-sweep-circle{
    position:absolute;
    top:50%;left:50%;
    width:260px;height:260px;
    transform:translate(-50%,-50%);
    border-radius:50%;
    border:1px solid rgba(0,230,255,0.2);
    animation:radarSweep 5s linear infinite;
}
.magnifier-border{
    position:absolute;
    top:50%;left:50%;
    width:120px;height:120px;
    border-radius:50%;
    border:2px solid rgba(0,230,255,0.5);
    box-shadow: 0 0 20px rgba(0,230,255,0.35) inset;
    animation:magnifierScan 2.2s ease-in-out infinite;
    z-index:1;
}
.car-icon {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%,-50%);
    width: 75px;
    height: 75px;
    animation: floatCar 3.2s ease-in-out infinite, pulseGlow 2.6s ease-in-out infinite;
    filter: drop-shadow(0 0 16px #00e6ff);
    z-index:3;
}
.loader-text {
    margin-top: 38px;
    font-size: 25px;
    color: #00e6ff;
    letter-spacing: 4px;
    text-shadow: 0 0 22px rgba(0,230,255,0.75);
    animation: pulseGlow 2.8s ease-in-out infinite;
}
</style>
"""

# ====================== 模型结构定义 ======================
class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(True), nn.Conv2d(hidden, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx,_ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x)*x)*x

class MultiTaskAssessor(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet50
        m = resnet50(weights=None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.cbam = CBAM(2048)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.15)
        self.type_head = nn.Linear(2048, 5)
        self.severity_head = nn.Linear(2048, 3)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.type_head(x), self.severity_head(x)

# 加载模型
@st.cache_resource
def load_models():
    stage1 = YOLO("best.pt")
    model = MultiTaskAssessor()
    ckpt = torch.load("small_stage2_best.pt", map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    return stage1, model

# ====================== 启动动画+预加载模型 ======================
first_load = st.session_state.first_init
stage1 = None
model = None

# 动画期间 彻底阻塞所有页面渲染
if first_load:
    with page_loader.container():
        st.markdown(GLOBAL_LOADER_STYLE, unsafe_allow_html=True)
        st.markdown("""
        <div class="loader-mask">
            <div class="tech-ring-wrap">
                <div class="radar-sweep-circle"></div>
                <svg class="tech-ring-outer" viewBox="0 0 100 100">
                    <circle class="ring-bg" cx="50" cy="50" r="46"></circle>
                    <circle class="ring-run" cx="50" cy="50" r="46"></circle>
                </svg>
                <svg class="tech-ring-inner" viewBox="0 0 100 100">
                    <circle class="ring-bg" cx="50" cy="50" r="40"></circle>
                    <circle class="ring-run" cx="50" cy="50" r="40"></circle>
                </svg>
                <div class="magnifier-border"></div>
                <div class="car-icon">
                    <svg viewBox="0 0 24 24" fill="#00e6ff">
                        <path d="M18.92 6.01C18.72 5.42 18.16 5 17.5 5H6.5c-.66 0-1.21.42-1.42 1.01L3 12v8c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-1h12v1c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-8l-2.08-5.99zM6.5 16c-.83 0-1.5-.67-1.5-1.5S5.67 13 6.5 13s1.5.67 1.5 1.5S7.33 16 6.5 16zm11 0c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5zM5 11l1.5-4.5h11L19 11H5z"/>
                        <circle cx="19" cy="6" r="3" fill="#00e6ff" opacity="0.9"/>
                        <path d="M20 3l-1.5 1.5" stroke="#00e6ff" stroke-width="1.2"/>
                    </svg>
                </div>
            </div>
            <div class="loader-text">InsurEye· 车辆损伤智能评估系统</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 模型加载 + 强制等待动画结束
    stage1, model = load_models()
    time.sleep(1.8)
    st.session_state.first_init = False
    page_loader.empty()

    # 强制 rerun，让所有内容一次性渲染
    st.rerun()

# 非首次加载直接取缓存
if not first_load:
    stage1, model = load_models()

st.session_state.models_loaded = True
# =================================================================

# 价格规则
PRICING = {
    'Scratch':{'base':220,'part':{'Door':1.2,'Fender':1.1,'Hood':1.25,'Bumper':1.1,'default':1.0}},
    'Dent':{'base':480,'part':{'Door':1.2,'Fender':1.1,'Hood':1.25,'Bumper':1.1,'default':1.0}},
    'Glass Break':{'base':1300,'part':{'Windshield':1.0,'default':1.0}},
    'Other Damage':{'base':650,'part':{'Door':1.2,'Fender':1.1,'Bumper':1.1,'default':1.0}},
}
SEV_MULT = {'Minor':1.0,'Moderate':1.6,'Severe':2.4}
def estimate_cost(part,typ,sev):
    if typ=='Normal':return 0
    r=PRICING.get(typ,{'base':300,'part':{}})
    return round(r['base'] * r['part'].get(part, r['part']['default']) * SEV_MULT.get(sev,1.0))

# 定损动画
def show_infer_loader():
    infer_mask = st.empty()
    with infer_mask.container():
        st.markdown(GLOBAL_LOADER_STYLE, unsafe_allow_html=True)
        st.markdown("""
        <div class="loader-mask">
            <div class="tech-ring-wrap">
                <div class="radar-sweep-circle"></div>
                <svg class="tech-ring-outer" viewBox="0 0 100 100">
                    <circle class="ring-bg" cx="50" cy="50" r="46"></circle>
                    <circle class="ring-run" cx="50" cy="50" r="46"></circle>
                </svg>
                <svg class="tech-ring-inner" viewBox="0 0 100 100">
                    <circle class="ring-bg" cx="50" cy="50" r="40"></circle>
                    <circle class="ring-run" cx="50" cy="50" r="40"></circle>
                </svg>
                <div class="magnifier-border"></div>
                <div class="car-icon">
                    <svg viewBox="0 0 24 24" fill="#00e6ff">
                        <path d="M18.92 6.01C18.72 5.42 18.16 5 17.5 5H6.5c-.66 0-1.21.42-1.42 1.01L3 12v8c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-1h12v1c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-8l-2.08-5.99zM6.5 16c-.83 0-1.5-.67-1.5-1.5S5.67 13 6.5 13s1.5.67 1.5 1.5S7.33 16 6.5 16zm11 0c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5zM5 11l1.5-4.5h11L19 11H5z"/>
                        <circle cx="19" cy="6" r="3" fill="#00e6ff" opacity="0.9"/>
                        <path d="M20 3l-1.5 1.5" stroke="#00e6ff" stroke-width="1.2"/>
                    </svg>
                </div>
            </div>
            <div class="loader-text">AI图像解析与损伤扫描中</div>
        </div>
        """, unsafe_allow_html=True)
    time.sleep(1.2)
    return infer_mask

# 其他状态初始化
if "current_img_idx" not in st.session_state:
    st.session_state.current_img_idx = 0
if "fullscreen_img" not in st.session_state:
    st.session_state.fullscreen_img = None
if "cached_results" not in st.session_state:
    st.session_state.cached_results = None
if "sample_img_list" not in st.session_state:
    st.session_state.sample_img_list = None

# 全局样式
st.markdown("""
<style>
    div[data-testid="stToolbar"] {visibility: hidden !important;}
    header {display: none !important;}
    footer {display: none;}

    .stApp {
    background-color: #0A101F;
    background-image:
        radial-gradient(circle at 10% 20%, rgba(0, 150, 255, 0.15) 0%, transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(0, 200, 255, 0.15) 0%, transparent 40%),
        linear-gradient(rgba(0, 180, 255, 0.08) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 180, 255, 0.08) 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 32px 32px, 32px 32px;
    background-attachment: fixed;
    color: #E0EFFF;
    font-family: 'Segoe UI', Roboto, 'Microsoft YaHei', sans-serif;
}

    .block-container { max-width: 1000px; margin: auto; padding: 2rem 1rem; }

    .title { 
        text-align: center; 
        font-size: 48px; 
        font-weight: 700; 
        margin-bottom: 5px; 
        color: #00CCFF; 
        text-shadow: 0 0 12px rgba(0,204,255,0.6);
    }
    .subtitle { 
        text-align: center; 
        font-size: 15px; 
        color: #8AB8E6; 
        margin-bottom: 40px; 
    }

    .tech-title {
        font-size: 20px !important;
        font-weight: 600 !important;
        color: #00CCFF !important;
        text-shadow: 0 0 8px rgba(0,204,255,0.4);
        margin-bottom: 12px !important;
    }

    div[data-testid="stFileUploader"] {
    background: #121B33 !important;
    border: 1px solid #253B70 !important;
    border-radius: 12px;
    min-height: 220px;
    transition: all 0.3s ease;
    position: relative;
    }
    div[data-testid="stFileUploader"]:hover {
        box-shadow: 0 0 20px rgba(0,204,255,0.2);
        border-color: #00CCFF !important;
    }
    div[data-testid="stFileUploader"] section {
        background: #182342 !important;
        border: 1px dashed #324A80 !important;
        border-radius: 12px;
        min-height: 220px;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
    }
    div[data-testid="stFileUploader"] button {
        background: #1B2B52 !important;
        color: #00CCFF !important;
        border-radius: 50px;
        border: 1px solid #324A80;
        margin-bottom: 8px;
    }
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] small {
        color: #C0D8FF !important;
        font-size: 12px !important;
    }
    div[data-testid="stFileUploader"] button[data-testid="baseButton-secondary"],
    div[data-testid="stFileUploader"] a {
        display: none !important;
    }

    div[data-testid="stFileUploader"]::after {
        content: "💡 点击选择文件 或 拖拽图片至此上传（最多4张）";
        position: absolute;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(20, 30, 50, 0.85);
        padding: 10px 28px;
        border-radius: 30px;
        color: #66D9FF;
        font-size: 14px;
        letter-spacing: 1px;
        opacity: 0;
        transition: opacity 0.3s ease;
        pointer-events: none;
        box-shadow: 0 0 15px rgba(0, 204,255,0.2);
        white-space: nowrap;
        }
    div[data-testid="stFileUploader"]:hover::after {
        opacity: 1;
    }

    .scroll-preview-container {
        background: #121B33;
        border: 1px solid #253B70;
        border-radius: 12px;
        padding: 12px;
        height: 220px;
        overflow-x: auto;
        overflow-y: hidden;
        display: flex;
        align-items: center;
        gap: 16px;
        justify-content: center;
    }
    .scroll-preview-container p {
        margin: 0;
        width: 100%;
        text-align: center;
        color: #8AB8E6;
    }
    .scroll-preview-container img {
        height: 140px;
        width: auto;
        object-fit: contain;
        border-radius: 8px;
        cursor: pointer;
        box-shadow: 0 0 8px rgba(0,204,255,0.15);
    }

    .stButton > button {
        background: linear-gradient(90deg, #0A2345, #103670) !important;
        color: #00CCFF !important;
        border-radius: 24px !important;
        height: 46px !important;
        font-weight: 600 !important;
        border: 1px solid #253B70 !important;
        display: block !important;
        margin: 0 auto !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 10px rgba(0,204,255,0.1);
    }
    .stButton > button:hover {
        transform: scale(1.05) !important;
        background: linear-gradient(90deg, #003366, #0055AA) !important;
        box-shadow: 0 0 18px rgba(0,204,255,0.4) !important;
        border-color: #00CCFF !important;
        color: #FFFFFF !important;
    }
    .main-action-btn > button {
    background: linear-gradient(90deg, #0A2345, #103670) !important;
    color: #00CCFF !important;
    border-radius: 0 !important;        /* 直角 */
    height: 58px !important;             /* 按钮高度 */
    font-size: 18px !important;          /* 文字大小 */
    max-width: 450px !important;         /* 按钮宽度 */
    width: 100% !important;              /* 铺满 */
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    border: 1px solid #00CCFF !important;
    display: block !important;
    margin: 0 auto !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 15px rgba(0,204,255,0.25);
}
    .main-action-btn > button:hover {
        transform: scale(1.06) !important;
        background: linear-gradient(90deg, #003366, #0055AA) !important;
        box-shadow: 0 0 25px rgba(0,204,255,0.5) !important;
        border-color: #00CCFF !important;
        color: #FFFFFF !important;
    }

    .damage-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        background: #121B33;
        border: 1px solid #253B70;
        border-radius: 16px;
        padding: 20px;
        min-height: 450px;
        box-shadow: 0 0 20px rgba(0,204,255,0.1);
    }
    .damage-wrapper img {
        max-width: 100%;
        max-height: 450px;
        width: auto;
        height: auto;
        object-fit: contain;
        border-radius: 12px;
        box-shadow: 0 0 15px rgba(0,204,255,0.2);
    }

    .page-info {
        text-align: center;
        margin: 16px 0;
        font-size: 16px;
        color: #8AB8E6;
    }
    
    .grid-header {
        display: grid;
        grid-template-columns: 12% 28% 20% 18% 22%;
        border-bottom: 2px solid #00CCFF;
        margin-bottom: 0;
        font-weight: 600;
        color: #00CCFF;
        padding-bottom: 8px;
    }
    .grid-row {
        display: grid;
        grid-template-columns: 12% 28% 20% 18% 22%;
        border-bottom: 1px solid #253B70;
    }
    .grid-cell { padding: 10px 8px; color: #C0D8FF; }
    .grid-row:hover {
    background-color: rgba(0, 204, 255, 0.15);
    transition: all 0.3s ease;
    cursor: pointer;
}
    .float-btn-left {
        position: fixed;
        top: 100px;
        left: 8%;
        z-index: 9999;
    }
    .float-btn-right {
        position: fixed;
        top: 100px;
        right: 8%;
        z-index: 9999;
    }
    .arch-btn {
        background: linear-gradient(90deg, #004477, #0066bb);
        color: white !important;
        border-radius: 24px;
        padding: 10px 22px;
        font-size: 14px;
        font-weight: 600;
        border: 2px solid #00CCFF;
        white-space: nowrap;
        box-shadow: 0 0 15px rgba(0,204,255,0.4);
        transition: all 0.25s ease;
        cursor: pointer;
        text-decoration: none !important;
        display: inline-block;
        opacity: 0.5;
        animation: archSweepLight 1s infinite ease-in-out;
    }
    .arch-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 0 22px rgba(0,204,255,0.6);
        color: white !important;
        opacity: 1;
    }
    .arch-preview-card {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        top: 65px;
        right: 0;
        width: 360px;
        background: #0A101F;
        border: 2px solid #00CCFF;
        border-radius: 16px;
        padding: 14px;
        box-shadow: 0 0 30px rgba(0,204,255,0.35);
        transition: all 0.25s ease;
        z-index: 99999;
    }
    .float-btn-right:hover .arch-preview-card {
        visibility: visible;
        opacity: 1;
    }
    .arch-preview-card img {
        width: 100%;
        border-radius: 10px;
        display: block;
    }
/* 🔥 强制呼吸动画 - 最高优先级！！！ */
@keyframes btnBreath {
    0%  { transform: scale(1) !important; box-shadow: 0 0 15px rgba(0,230,255,0.7) !important; }
    50% { transform: scale(1.08) !important; box-shadow: 0 0 45px rgba(0,230,255,1) !important; }
    100%{ transform: scale(1) !important; box-shadow: 0 0 15px rgba(0,230,255,0.7) !important; }
}

/* 🔥 绝对命中示例按钮 - 覆盖所有样式 */
div.sample-btn-wrapper button {
    animation: btnBreath 2s infinite ease-in-out !important;
    border: 2px solid #00e6ff !important;
    background: linear-gradient(90deg, #004080, #0066cc) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    border-radius: 28px !important;
    height: 50px !important;
    font-size: 15px !important;
    pointer-events: auto !important;
}
/* 真正的横向流光扫过 */
@keyframes archSweepLight {

  /* 平缓横向划过 */
  50% {
    background-position: 100% 50%;
    opacity: 0.85;
  }

  
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>InsurEye · 车辆损伤智能评估系统</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>事故现场 · 秒级定损 · 维修估价</div>", unsafe_allow_html=True)

# 中英文映射
part_cn = {
    'Back-bumper': '后保险杠','Back-door': '后门','Back-wheel': '后轮','Back-window': '后窗','Back-windshield': '后挡风玻璃',
    'Fender': '叶子板','Front-bumper': '前保险杠','Front-door': '前门','Front-wheel': '前轮','Front-window': '前窗',
    'Grille': '进气格栅','Headlight': '大灯','Hood': '发动机盖','License-plate': '车牌','Mirror': '后视镜',
    'Quarter-panel': '后侧围','Rocker-panel': '下边梁','Roof': '车顶','Tail-light': '尾灯','Trunk': '后备箱盖','Windshield': '挡风玻璃',
}
type_cn = {'Normal':'正常','Scratch':'划痕','Dent':'凹痕','Glass Break':'玻璃破裂','Other Damage':'其他损伤'}
sev_cn = {'Minor':'轻度','Moderate':'中度','Severe':'重度'}
type_names = ['Normal','Scratch','Dent','Glass Break','Other Damage']
sev_names = ['Minor','Moderate','Severe']
tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.48,0.45,0.40],[0.229,0.224,0.225])])

# 推理&绘图
def infer_one(img):
    img=img.convert('RGB')
    res=stage1(img)
    items=[]
    for r in res:
        if not r.boxes: continue
        for box in r.boxes:
            x1,y1,x2,y2=map(int,box.xyxy[0])
            part=r.names[int(box.cls[0])]
            crop=img.crop((x1,y1,x2,y2))
            with torch.no_grad():
                t,s=model(tf(crop).unsqueeze(0))
            typ=type_names[t.argmax().item()]
            sev=sev_names[s.argmax().item()]
            cost=estimate_cost(part,typ,sev)
            items.append((part,typ,sev,cost,(x1,y1,x2,y2)))
    return items

def draw_boxes(img, items):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("simhei.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    color_palette = [
        "#FF4444","#44FF44","#4488FF","#FFFF44",
        "#FF44FF","#44FFFF","#FF8844","#88FF44",
        "#FF4488","#44FF88","#8844FF","#FF8844"
    ]
    
    for idx, (p, t, s, c, (x1, y1, x2, y2)) in enumerate(items):
        if t == "Normal": continue
        col = color_palette[idx % len(color_palette)]
        draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
        label = f"{part_cn.get(p, p)}·{type_cn[t]}"
        draw.text((x1 + 5, y1 + 5), label, fill=col, font=font)
    return img

# -------------------------- 首页 --------------------------
if st.session_state.page_state == "home":

    # 🔥 稳定版：Streamlit 原生按钮 + 强制呼吸动画（不刷新、不闪屏）
    st.markdown("""
    <style>
    /* 给所有按钮加动画 */
    div.stButton > button[kind="secondary"]:not([key="start_btn"]) {
    animation: htmlBreath 2s infinite ease-in-out !important;
    background: linear-gradient(90deg, #004080, #0066cc) !important;
    color: white !important;
    font-weight:bold !important;
    border: 2px solid #00e6ff !important;
    height: 52px !important;
    font-family: "Microsoft YaHei", Orbitron, sans-serif !important;
    font-size:29px !important;
    border-radius:18px !important;
    max-width: 300px !important;
    font-weight: 900;
    margin: 8px auto 20px auto !important;     
    border-radius: 20px !important;        /* 圆角改小，硬朗科技直角风 */
    letter-spacing: 2.5px !important;      /* 字间距拉开，更炫酷科技感 */      
}   
    
    @keyframes htmlBreath {
        0%  { transform: scale(1); box-shadow: 0 0 20px #00e6ff; }
        50% { transform: scale(1.09); box-shadow: 0 0 40px #00e6ff; }
        100%{ transform: scale(1); box-shadow: 0 0 20px #00e6ff; }
    }
    </style>
    """, unsafe_allow_html=True)

    # 原生按钮，不刷新页面！
    if st.button("暂无实拍？点此加载示例样图！", use_container_width=True):
        sample_dir = "sample_demo"
        img_list = []
        if os.path.exists(sample_dir):
            exts = ['*.png', '*.jpg', '*.jpeg', 'webp']
            sample_paths = []
            for ext in exts:
                sample_paths.extend(glob.glob(os.path.join(sample_dir, ext)))
            sample_paths = sorted(sample_paths)[:4]
            for path in sample_paths:
                img = Image.open(path).convert("RGB")
                img_list.append(img)
        st.session_state.sample_img_list = img_list
        st.rerun()
    # ===================== 下面所有代码 100% 不动 =====================
    try:
        with open("arch.png", "rb") as f:
            b64_img = base64.b64encode(f.read()).decode()
    except:
        b64_img = ""

    st.markdown(f'''
    <div class="float-btn-right">
        <a class="arch-btn" href="?page=architecture">点击查看技术示意图</a>
        <div class="arch-preview-card">
            <img src="data:image/png;base64,{b64_img}">
        </div>
    </div>
    ''', unsafe_allow_html=True)

    col_upload, col_preview = st.columns([1,1], gap="large")
    with col_upload:
        st.markdown('<div class="tech-title">📸 上传事故照片</div>', unsafe_allow_html=True)
        if st.session_state.sample_img_list:
            uploaded = None
            st.info("✅ 已加载示例样图，请于加载完成后点击“开始智能定损”")
        else:
            uploaded = st.file_uploader(" ",type=['jpg','jpeg','png','webp'],accept_multiple_files=True,label_visibility="collapsed", key="uploader_1")
    with col_preview:
        st.markdown('<div class="tech-title">🖼️ 原图预览</div>', unsafe_allow_html=True)
        if st.session_state.sample_img_list:
            html = "<div class='scroll-preview-container'>"
            for img in st.session_state.sample_img_list:
                buf = BytesIO()
                img.save(buf, "PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                html += f"<img src='data:image/png;base64,{b64}' />"
            st.markdown(html+"</div>", unsafe_allow_html=True)
        elif uploaded:
            html = "<div class='scroll-preview-container'>"
            for f in uploaded:
                img = Image.open(f).convert('RGB')
                buf = BytesIO()
                img.save(buf, "PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                html += f"<img src='data:image/png;base64,{b64}' />"
            st.markdown(html+"</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='scroll-preview-container'><p>请先上传事故照片</p></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn, _ = st.columns([1, 2, 1])
    with btn:
        can_run = (uploaded is not None and len(uploaded) > 0) or (st.session_state.sample_img_list is not None and len(st.session_state.sample_img_list) > 0)
        st.markdown('<div class="main-action-btn">', unsafe_allow_html=True)
        run = st.button("开始智能定损", use_container_width=True, disabled=not can_run, key="start_btn")
        st.markdown('</div>', unsafe_allow_html=True)

    if run and can_run:
        infer_loader = show_infer_loader()
        if st.session_state.sample_img_list:
            imgs = st.session_state.sample_img_list
        else:
            imgs = [Image.open(f).convert('RGB') for f in uploaded]
        all_items = [infer_one(i) for i in imgs]
        marked_images = [draw_boxes(i, it) for i, it in zip(imgs, all_items)]
        infer_loader.empty()
        st.session_state.cached_results = {"imgs": imgs,"all_items": all_items,"marked_images": marked_images}
        st.session_state.current_img_idx = 0
        st.session_state.page_state = "result"
        st.rerun()
# -------------------------- 结果页 --------------------------
elif st.session_state.page_state == "result":
    st.session_state.sample_img_list = None
    results = st.session_state.cached_results
    if not results:
        st.session_state.page_state = "home"
        st.rerun()
    marked = results["marked_images"]
    all_items = results["all_items"]
    total = len(marked)
    current = st.session_state.current_img_idx

    st.markdown('<div class="tech-title">🔬 损伤标注结果</div>', unsafe_allow_html=True)
    buf = BytesIO()
    marked[current].save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(f"<div class='damage-wrapper'><img src='data:image/png;base64,{b64}'/></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-info'>当前显示: 图{current+1} / {total}</div>", unsafe_allow_html=True)

    _, btnZoom, _ = st.columns([1,2,1])
    with btnZoom:
        if st.button("🔍 点击放大图片", use_container_width=True):
            st.session_state.fullscreen_img = b64
            st.session_state.page_state = "fullscreen"
            st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("◀ 上一张", disabled=(current == 0), use_container_width=True):
            st.session_state.current_img_idx -= 1
            st.rerun()
    with c2:
        if st.button("下一张 ▶", disabled=(current == total - 1), use_container_width=True):
            st.session_state.current_img_idx += 1
            st.rerun()

    st.markdown("---")
    
    st.markdown("""
<style>
.down-arrow {
    position: absolute;
    left: 37px;
    top: -420px;
    z-index: 1000;
    opacity: 0.95;
    pointer-events: none;
    animation: bounce 1.2s infinite ease-in-out;
}
.down-arrow-text {
    position: absolute;
    left: 50px;
    top: -570px;
    font-size: 14px;
    font-weight: 400;
    color: #aaddff;
    text-shadow: 0 0 8px rgba(0,204,255,0.6);
    writing-mode: vertical-rl;
    text-orientation: mixed;
    letter-spacing: 5px;
    animation: bounce 1.2s infinite ease-in-out, textGlow 2s infinite alternate;
    z-index: 1000;
    white-space: nowrap;
    pointer-events: none;
    font-family: 'Segoe UI', 'Microsoft YaHei', monospace;
}
@keyframes bounce {
    0%,100% { transform: translateY(0px); }
    50% { transform: translateY(10px); }
}
@keyframes textGlow {
    0% { text-shadow: 0 0 4px rgba(0,204,255,0.4); }
    100% { text-shadow: 0 0 12px rgba(0,204,255,0.9); }
}
</style>

<div class="down-arrow-container">
    <div class="down-arrow">
        <svg width="50" height="50" viewBox="0 0 50 50" style="display:block;">
            <circle cx="25" cy="25" r="20" stroke="#00e6ff" stroke-width="1.5" fill="none" stroke-dasharray="4 4" opacity="0.6">
                <animateTransform attributeName="transform" type="rotate" from="0 25 25" to="360 25 25" dur="3s" repeatCount="indefinite"/>
            </circle>
            <circle cx="25" cy="25" r="14" stroke="#00e6ff" stroke-width="1" fill="none" opacity="0.4">
                <animate attributeName="r" values="14;18;14" dur="1.5s" repeatCount="indefinite"/>
                <animate attributeName="opacity" values="0.4;0.1;0.4" dur="1.5s" repeatCount="indefinite"/>
            </circle>
            <path d="M25 8 L25 32 M18 25 L25 33 L32 25" stroke="#00e6ff" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round">
                <animate attributeName="stroke" values="#00e6ff;#66ffff;#00e6ff" dur="1.5s" repeatCount="indefinite"/>
            </path>
            <circle cx="25" cy="25" r="2" fill="#00e6ff" opacity="0.8">
                <animate attributeName="cy" values="25;32;25" dur="1.2s" repeatCount="indefinite"/>
                <animate attributeName="opacity" values="0.8;0.2;0.8" dur="1.2s" repeatCount="indefinite"/>
            </circle>
        </svg>
    </div>
    <div class="down-arrow-text">下方查看维修工单</div>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="tech-title">⚙️ 维修工单</div>', unsafe_allow_html=True)

    total_cost = 0
    st.markdown('<div class="grid-header"><div class="grid-cell">图片</div><div class="grid-cell">部件</div><div class="grid-cell">损伤类型</div><div class="grid-cell">严重程度</div><div class="grid-cell">预估维修费</div></div>', unsafe_allow_html=True)
    
    for i in range(total):
        items = all_items[i]
        damaged = [d for d in items if d[1] != "Normal"]
        if not damaged:
            st.markdown(f'''
            <div class="grid-row">
                <div class="grid-cell">图 {i+1}</div>
                <div class="grid-cell">-</div>
                <div class="grid-cell">未见明显损伤</div>
                <div class="grid-cell">-</div>
                <div class="grid-cell">-</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            for p, t, s, c, _ in damaged:
                total_cost += c
                st.markdown(f'''
                <div class="grid-row">
                    <div class="grid-cell">图 {i+1}</div>
                    <div class="grid-cell">{part_cn.get(p, p)}</div>
                    <div class="grid-cell">{type_cn[t]}</div>
                    <div class="grid-cell">{sev_cn[s]}</div>
                    <div class="grid-cell">{c} 元</div>
                </div>
                ''', unsafe_allow_html=True)

    st.markdown(f"<div style='text-align:right; font-size:20px; color:#FFFFFF; margin-top:16px; font-weight:700; letter-spacing:1px;'>合计预估维修费：{total_cost} 元</div>", unsafe_allow_html=True)
    st.caption("AI评估结果仅供参考，最终以保险公司定损为准。")
    _, back, _ = st.columns([1, 2, 1])
    with back:
        if st.button("返回首页", use_container_width=True):
            st.session_state.page_state = "home"
            st.query_params.clear()
            st.rerun()

# -------------------------- 放大页 --------------------------
elif st.session_state.page_state == "fullscreen":
    st.markdown('<div class="tech-title">拖动滑块，按比例缩放</div>', unsafe_allow_html=True)
    scale = st.slider("缩放", 0.5, 3.0, 1.0, 0.1)
    st.markdown(f"""
    <div style='display:flex;justify-content:center;align-items:center;min-height:70vh;overflow:auto;'>
        <img src='data:image/png;base64,{st.session_state.fullscreen_img}' style='
            width: {scale * 400}px;
            max-width: unset;
            max-height: 90vh;
            object-fit: contain;
            border-radius: 16px;
            box-shadow: 0 0 30px rgba(0,204,255,0.2);
        ' />
    </div>""", unsafe_allow_html=True)
    _, backBtn, _ = st.columns([1,2,1])
    with backBtn:
        if st.button(" 返回结果", use_container_width=True):
            st.session_state.page_state = "result"
            st.rerun()

# -------------------------- 技术架构页 --------------------------
elif st.session_state.page_state == "architecture":
    st.markdown('<div class="title" style="font-size: 34px;">完整技术路线与系统架构</div>', unsafe_allow_html=True)
    st.divider()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    arch_path = os.path.join(current_dir, "arch.png")
    if os.path.exists(arch_path):
        st.image(arch_path, use_container_width=True)
    else:
        st.info("技术架构图暂不可用")
    st.divider()
    st.markdown("""
    <style>
    .tech-section {
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
    }
    .tech-section h3 {
    font-size: 24px;
    font-weight: 700;
    color: #00CCFF;
    text-shadow: 0 0 8px rgba(0,204,255,0.4);
    margin: 20px 0 15px 0;
    letter-spacing: 2px;
    border-left: 4px solid #00CCFF;
    padding-left: 15px;
    }
    .tech-section h4 {
    font-size: 18px;
    font-weight: 600;
    color: #66D9FF;
    margin: 15px 0 10px 0;
    letter-spacing: 1px;
    }
    .tech-section ul {
    list-style-type: none;
    padding-left: 20px;
    }
    .tech-section li {
    font-size: 15px;
    line-height: 1.8;
    color: #C0D8FF;
    margin: 6px 0;
    position: relative;
    padding-left: 20px;
    }
    .tech-section li::before {
    content: "▹";
    position: absolute;
    left: 0;
    color: #00CCFF;
    font-size: 14px;
    }
    .tech-section li strong {
    color: #00CCFF;
    font-weight: 600;
    }
    .highlight-box {
    background: rgba(0, 204, 255, 0.08);
    border-left: 3px solid #00CCFF;
    padding: 8px 0 8px 15px;
    margin: 12px 0;
    border-radius: 0 8px 8px 0;
    }
    </style>

    <div class="tech-section">
    <h3>🔹 1. 整体处理流程</h3>
    <ul>
        <li><strong>输入层</strong>：车辆事故现场图像（支持复杂光照、多角度拍摄）</li>
        <li><strong>一阶部件检测</strong>：使用 YOLOv8n 模型实现车辆部件精准定位与区域提取</li>
        <li><strong>ROI 区域裁剪</strong>：根据检测框自动提取损伤感兴趣区域，为后续评估做准备</li>
        <li><strong>二阶损伤评估</strong>：基于 ResNet50 + CBAM 注意力机制，实现多任务精细识别
            <ul>
                <li>损伤类型分类（划痕/凹痕/玻璃破裂/其他损伤）</li>
                <li>损伤严重程度分级（轻度/中度/重度）</li>
            </ul>
        </li>
        <li><div class="highlight-box"><strong>质量控制与决策层</strong>：<span style="background:linear-gradient(135deg,#244b85,#16325c);color:#fff;padding:3px 10px;border-radius:6px;font-weight:bold;">⚙️ 待完善</span> 后续将支持误检过滤、置信度校准、几何规则约束优化</div></li>
        <li><strong>输出层</strong>：生成可视化标注结果、结构化损伤报告与维修费用预估单</li>
    </ul>

    <h3>🔹 2. 核心技术栈</h3>
    <ul>
        <li><strong>目标检测</strong>：YOLOv8n（轻量高效，适合实时定损场景）</li>
        <li><strong>特征提取</strong>：ResNet50 深度卷积网络</li>
        <li><strong>注意力机制</strong>：CBAM 通道+空间混合注意力模块</li>
        <li><strong>多任务学习</strong>：损伤分类 + 严重程度分级联合输出</li>
        <li><strong>应用输出</strong>：可视化框注 + 智能维修报价</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    _, backBtn, _ = st.columns([1,2,1])
    with backBtn:
        if st.button("返回首页", use_container_width=True):
            st.session_state.page_state = "home"
            st.query_params.clear()
            st.rerun()

# 底部
st.markdown("<div style='text-align:center;color:#00CCFF;padding:20px;font-size:12px;letter-spacing:1px;'>⌘ INSUREYE v1.0 · 智能车辆定损系统 · 王可扬 · 韩天一 · © 2026 ⌘</div>", unsafe_allow_html=True)
