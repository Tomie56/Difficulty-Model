# 📚 Difficulty-Model

A machine learning framework for predicting prompt difficulty based on dialogue data. The system uses reward prediction models (MLP, Random Forest) trained on embeddings extracted from LMSYS-Chat conversations. It supports simulation of difficulty-aware resource allocation and human validation experiments.

---

## 📦 项目结构

```
Difficulty-Model/
├── datas/                      # 数据目录（原始数据、嵌入数据）
├── models/                     # 保存训练好的模型（.pth/.pkl）及加载脚本
├── results/                    # 模型输出结果（图表、分配结果等）
├── scripts/                    # 数据处理与训练评估脚本
│   ├── embeddings.py           # 生成对话嵌入
│   ├── allocate.py             # 难度感知资源分配模拟
│   ├── data_cleaning.py        # 清洗原始LMSYS数据
│   ├── data_processing_v2.py   # 格式化训练数据
│   ├── train_v2.py             # 训练 MLP 主模型
│   ├── train_v3_RF.py          # 训练 Random Forest 模型
│   ├── test_model_v3_RF.py     # 测试 RF 模型
│   ├── test_model_v1/v2/v4/v5.py  # 不同 MLP 模型版本测试
│   └── generate_rewards.py     # 使用 reward model 生成打分标签
├── README.md                   # 当前说明文件
├── requirements.txt            # Python 依赖项
└── LICENSE
```

---

## 📥 安装依赖

使用 Python 3.8+ 环境，建议使用虚拟环境隔离：

```bash
# 创建并激活虚拟环境
python -m venv env
source env/bin/activate        # macOS/Linux
env\Scripts\activate         # Windows

# 安装依赖包
pip install -r requirements.txt
```

---

## 🔄 数据处理流程

请顺序执行以下命令以生成训练数据：

```bash
python ./scripts/data_cleaning.py
python ./scripts/data_processing_v2.py
python ./scripts/embeddings.py
```

---

## 🧠 模型训练与评估

训练主模型（MLP）：

```bash
python ./scripts/train_v2.py
```

训练基线模型（Random Forest）：

```bash
python ./scripts/train_v3_RF.py
```

模型评估：

```bash
python ./scripts/test_model_v3_RF.py
python ./scripts/test_model_v2.py
```

---

## 🚀 难度感知资源分配模拟

使用训练好的模型对测试集进行分组预测与资源分配：

```bash
python ./scripts/allocate.py
```

结果包括：平均 reward 分布、t 检验显著性分析、人类标注验证结果等，输出文件存于 `results/resource_allocation/`。

---

## 📝 人工验证实验

我们随机抽样 50 条测试对话，人工与 DeepSeek-V3 协同进行“易/难”标签标注。运行后生成混淆矩阵、准确率、F1 分数报告等：

```bash
# 已集成于 allocate.py
```

---

## ⚙️ 软件环境

| 库              | 版本     |
|----------------|----------|
| Python         | 3.12     |
| PyTorch        | 2.1.0    |
| NumPy          | 1.24.0   |
| pandas         | 1.5.3    |
| scikit-learn   | 1.2.0    |
| scipy          | 1.10.0   |
| transformers   | 4.27.0   |
| tqdm           | 4.65.0   |

---

## 📮 联系我们

如有问题，请联系项目负责人：

📧 224040266@link.cuhk.edu.cn
