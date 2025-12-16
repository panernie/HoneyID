import streamlit as st
import pandas as pd
import numpy as np
import tqdm
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
import joblib
from PIL import Image
import random
from tqdm import tqdm
import time
import os
import base64
import pickle
from streamlit.components.v1 import html

st.set_page_config()
my_html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>显示年月日、北京时间及星期</title>
<style>
    body {
        font-family: Arial, sans-serif;
        text-align: center;
        padding-top: 50px;
    }
    #clock {
        font-size: 2.5em;
        color: #333;
    }
</style>
</head>
<body>
<div id="clock">加载中...</div>

<script>
function updateClock() {
    const now = new Date(); // 获取当前时间
    const year = now.getFullYear(); // 年
    const month = (now.getMonth() + 1).toString().padStart(2, '0'); // 月，从0开始所以加1
    const day = now.getDate().toString().padStart(2, '0'); // 日
    const hours = now.getHours().toString().padStart(2, '0'); // 小时
    const minutes = now.getMinutes().toString().padStart(2, '0'); // 分钟
    const seconds = now.getSeconds().toString().padStart(2, '0'); // 秒
    const weekDays = ['星期日', '星期一', '星期二', '星期三', '星期四', '星期五', '星期六'];
    const weekDay = weekDays[now.getDay()]; // 星期

    // 格式化时间字符串
    const formattedTime = `${year}年${month}月${day}日 ${hours}:${minutes}:${seconds} ${weekDay}`;
    
    // 更新页面上的时间
    document.getElementById('clock').textContent = formattedTime;
}

// 每秒更新一次时间
setInterval(updateClock, 1000);

// 页面加载时立即更新一次时间
updateClock();
</script>
</body>
</html>


<style>
/* 增加鼠标追踪样式 */
#tracker {
    position: absolute;
    width: 15px;
    height: 10px;
    background-color: red;
    border-radius: 50%;
    pointer-events: none; /* 防止影响其他元素的点击事件 */
}
</style>

<body>
<!-- ... 之前的时间显示部分 ... -->
<div id="tracker"></div>

<script>
// ... 之前的时间更新脚本 ...

document.addEventListener('mousemove', function(event) {
    const tracker = document.getElementById('tracker');
    // 设置追踪器位置，减去半径使追踪更准确
    tracker.style.left = (event.clientX - 10) + 'px';
    tracker.style.top = (event.clientY - 10) + 'px';
});
</script>
</body>
</html>

 
 
 
 
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>全屏雪花效果</title>
<style>
    body, html {
        margin: 0;
        padding: 0;
        overflow: hidden;
        height: 100%;
        background: #000;
    }
    .snowflake {
        position: absolute;
        width: 10px;
        height: 10px;
        background: white;
        border-radius: 50%;
        animation: fall linear infinite;
    }
    @keyframes fall {
        0% { transform: translateY(-100vh); opacity: 0; }
        50% { opacity: 1; }
        100% { transform: translateY(100vh); opacity: 0; }
    }
</style>
</head>
<body>
<script>
    function createSnowflakes(num) {
        for (let i = 0; i < num; i++) {
            let snowflake = document.createElement('div');
            snowflake.className = 'snowflake';

            // 设置随机位置
            snowflake.style.left = Math.random() * 100 + 'vw';
            snowflake.style.top = -Math.random() * 20 + 'vh';

            // 设置随机大小
            let size = Math.random() * 5 + 5;
            snowflake.style.width = size + 'px';
            snowflake.style.height = size + 'px';

            // 设置随机速度
            snowflake.style.animationDuration = Math.random() * 3 + 6 + 's';

            // 设置随机旋转
            snowflake.style.transform = 'rotate(' + Math.random() * 360 + 'deg)';

            // 添加到文档
            document.body.appendChild(snowflake);
        }
    }

    window.onload = function() {
        createSnowflakes(100); // 创建100个雪花
    };
</script>
</body>
</html>


"""
 # 使用 Streamlit 显示 HTML 内容
st.components.v1.html(my_html)   
    
#give the path of the model you want to use from the folder Model
features = pd.read_csv('feature/feature.csv')
feas = list(features.feature.values)
model_rf= joblib.load('model/MLP_classf_kf_cv_best_0.pkl')
scaleed = joblib.load('model/scaled.pkl')

replacement_map = {0: "Rape honey", 1: "Acacia honey"}

# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href




# Logo image
image = Image.open('cover.png')
st.image(image, use_column_width=True)
# Page title
st.markdown(
    """
# Acacia honey and rape honey Identifier
This tool is an intelligent tool specially developed for the identification of acacia honey and rape honey in China. It aims to help users accurately identify the origin of wines by providing HS-SPME-GC-MS data analysis of wines.
It is designed to help users accurately identify low-value rape honey and high-value acacia honey by performing rapid identification of honey with sugar and amino acid data.
The tool uses advanced machine learning models to ensure high accuracy and reliability of the identification results and to promote the development of the honey health industry.
Note: This tool is not applicable for the identification of other honey varieties
"""
)

# Sidebar
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("OR, upload your input file", type=['csv'])
    st.sidebar.markdown(
        """
[Example input file](https://github.com/panernie/AHTPeptideFusion/blob/main/test192.csv)
"""
    )
    



    #title = st.text_input("Input your sequence, eg. IRW")
    
    
if st.sidebar.button('Predict'):

    T1 = time.time()

    df = pd.read_csv(uploaded_file)
    new_df = pd.DataFrame(0, index=df.index, columns=feas)
    common_columns = df.columns.intersection(new_df.columns)
    new_df[common_columns] = df[common_columns]
    st.header('**Original input data**')
    st.write(f"{new_df.shape[0]} samples were identified")
    with st.spinner("Please wait..."):
        time.sleep(1)
        # load_data.to_csv('molecule.smi', sep = '\t', header = False, index = False)
        
        
        X_scaled = scaleed.fit_transform(new_df)
        rf_pred = model_rf.predict(X_scaled)
        
        
        dfa = pd.DataFrame(np.array(rf_pred).T,columns=["MLP model"])
        dfa.index.name = 'sample'
        dfa.index = pd.RangeIndex(start=1, stop=len(dfa)+1, step=1, name='sample')
        dfa.replace(replacement_map, inplace=True)

        dfa.to_csv("output\prediction.csv")
    file_names = time.time()

    # print(df_all)
    #df_all = pd.read_csv("output\prediction.csv")
    df_10 = dfa[:10]
    T2 = time.time()
    st.success("Done!")
    st.write('Program run time:%sms' % ((T2 - T1) * 1000))
    st.header('**Output data**')
    st.write("Only the first 10 results are displayed!")
    st.write(df_10)
    st.markdown(filedownload(dfa), unsafe_allow_html=True)
else:
    st.info('Upload input data in the sidebar to start!')
