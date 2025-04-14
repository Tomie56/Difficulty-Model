# 📚 Difficulty-Model

Project for AIR5101 — A machine learning model to estimate or rank task difficulty based on dialogue data.

---

## 📦 项目结构

```
Difficulty-Model/
├── datas/                    # 原始数据 & 处理脚本
│   └── load_lmsys.py
├── models/                   # 训练模型及推理脚本
│   └── *.pth, *.pkl, *.py
├── scripts/                  # 数据清洗、处理、训练脚本
│   ├── data_cleaning.py
│   ├── data_processing_v2.py
│   ├── embeddings.py
│   └── train_v2.py
├── requirements.txt          # Python依赖包
└── README.md                 # 项目说明文件
```

---

## 📥 安装依赖

请使用 Python 3.8+ 环境，推荐使用虚拟环境：

```bash
# 创建虚拟环境
python -m venv env
source env/bin/activate    # Linux/Mac
env\Scripts\activate       # Windows

# 安装依赖
pip install -r requirements.txt
```

---

## 🔄 数据处理

运行以下命令以加载和处理原始数据：

```bash
python ./datas/load_lmsys.py
python ./scripts/data_cleaning.py
python ./scripts/data_processing_v2.py
python ./scripts/embeddings.py
```

---

## 🧠 模型训练

使用以下命令进行模型训练：

```bash
python ./scripts/train_v2.py
```

---

## 📝 模型说明

模型使用预训练语言模型生成的嵌入表示，配合回归/分类模型预测对话样本的相对难度。训练后模型保存在 `models/` 目录下，支持直接推理或评估。

---


## 📮 联系方式

224040266@link.cuhk.edu.cn

---
