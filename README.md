# 💰 AI Offer 选大米助手 

[![GLM-4V](https://img.shields.io/badge/Model-GLM--4V-blue)](https://open.bigmodel.cn/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red)](https://streamlit.io/)

 **基于多模态大模型 (GLM-4V) + RAG 的智能 Offer 决策助手。**
解决 Offer 截图隐私清洗、薪资公式解析及多维度对比难题。

## ✨ 核心功能
- **📸 截图即用**：自动识别 Offer 截图（OCR），提取薪资、职级、地点等硬核数据。
- **🛡️ 隐私清洗**：自动过滤姓名、ID 等敏感信息，只保留 Offer 条款。
- **📊 智能对比**：针对用户关注的公司（如字节/快手）自动生成 Markdown 对比表格，包含总包估算。
- **🧠 混合检索**：结合向量检索与白名单机制，精准召回相关 Offer。

## 🛠️ 技术栈
- **核心模型**: 智谱 GLM-4V-Flash (视觉理解 & 逻辑推理)
- **应用框架**: Streamlit
- **向量数据库**: ChromaDB + BAAI/bge-m3 Embedding
- **工程亮点**: 增量建库、长文本 RAG 优化、图片自动压缩

## 🚀 3步快速运行

### 1. 安装依赖
```bash
pip install -r requirements.txt
```
### 2. 配置API key
**在项目根目录手动创建 .streamlit/secrets.toml 文件**

### 3. 运行应用
**需先在 images/ 文件夹放入 Offer 截图，然后运行：**
```bash
streamlit run test.py
```
