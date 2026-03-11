# 🎬 Scalable Video Recommendation System

> 企业级视频推荐系统 · 参考阿里巴巴 / 快手工业架构

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 项目简介

本项目从零构建一套**工业级视频推荐系统**，完整覆盖：

- **数据 Pipeline**：清洗、特征工程、样本工厂
- **特征存储**：双层 Feature Store（离线 Parquet + 在线 Redis 模拟）
- **召回层**：双塔模型（Two-Tower DSSM）+ I2I 协同过滤
- **排序层**：DIN 精排模型 + 多目标 Loss
- **实验平台**：A/B 分流框架 + 统计显著性检验
- **在线监控**：PSI 特征漂移 + 预测分布监控

---

## 🏗️ 系统架构

```
用户行为日志
      ↓
┌─────────────────────────────────────────┐
│           数据 Pipeline                  │
│  清洗 → 特征工程 → 样本工厂 → Feature Store │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│           推荐主链路                      │
│  多路召回 → 粗排 → 精排(DIN) → 重排       │
└─────────────────────────────────────────┘
      ↓
┌──────────────┐    ┌──────────────────────┐
│  实验平台     │    │     在线监控          │
│  A/B 分流    │    │  PSI + 预测分布       │
└──────────────┘    └──────────────────────┘
```

---

## 📁 项目结构

```
VideoRecSys/
├── notebooks/                  # Colab Notebooks（每步一个）
│   ├── step1_data_pipeline.ipynb
│   ├── step2_recall_model.ipynb
│   ├── step3_ranking_model.ipynb
│   └── step4_ab_testing.ipynb
│
├── pipeline/                   # 数据处理模块
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── sample_factory.py
│   ├── feature_store.py
│   └── dqc_monitor.py
│
├── models/                     # 模型代码
│   ├── recall/
│   │   ├── two_tower_model.py  # 双塔召回
│   │   ├── i2i_recall.py       # I2I 协同过滤
│   │   └── faiss_index.py      # ANN 向量检索
│   ├── ranking/
│   │   ├── din_model.py        # DIN 精排
│   │   ├── multi_task_loss.py  # 多目标 Loss
│   │   └── model_eval.py       # 离线评估
│   └── rerank/
│       └── mmr_diversity.py    # 多样性重排
│
├── serving/                    # 推理服务
│   ├── serving_api.py          # FastAPI 服务
│   ├── feature_server.py       # 特征服务
│   └── start_server.sh
│
├── experiment/                 # 实验平台
│   ├── ab_framework.py
│   ├── metrics_calculator.py
│   └── experiments.yaml
│
├── monitoring/                 # 在线监控
│   ├── psi_monitor.py
│   ├── prediction_monitor.py
│   └── alert_rules.yaml
│
├── configs/                    # 配置文件
│   ├── pipeline_config.yaml
│   ├── recall_config.yaml
│   ├── ranking_config.yaml
│   └── feature_config.yaml
│
├── docker/                     # 容器化
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── .github/workflows/          # CI/CD
│   └── ci.yml
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 快速开始

### 方式一：Google Colab（推荐，零配置）

1. 打开 [Google Colab](https://colab.research.google.com)
2. `文件` → `上传笔记本` → 选择 `notebooks/step1_data_pipeline.ipynb`
3. `运行时` → `全部运行`

### 方式二：本地运行

```bash
git clone https://github.com/YOUR_USERNAME/VideoRecSys.git
cd VideoRecSys
pip install -r requirements.txt
jupyter notebook notebooks/step1_data_pipeline.ipynb
```

---

## 📊 数据集

| 阶段 | 数据集 | 用途 |
|------|--------|------|
| 流程验证 | MovieLens-1M | 跑通完整 Pipeline |
| 正式训练 | KuaiRec 2.0 | 真实短视频交互数据 |
| 多模态 | MicroLens | 视频内容特征 |

---

## 📈 开发进度

- [x] **Step 1**: 数据 Pipeline + Feature Store
- [ ] **Step 2**: 双塔召回模型
- [ ] **Step 3**: DIN 精排模型
- [ ] **Step 4**: 实验平台 + 在线监控

---

## 🛠️ 技术栈

| 层次 | 技术 |
|------|------|
| 数据处理 | Pandas, PySpark, PyArrow |
| 特征工程 | Scikit-learn, NumPy |
| 深度学习 | PyTorch |
| 向量检索 | FAISS |
| 推理服务 | FastAPI |
| 容器化 | Docker |
| 云部署 | Google Cloud Run |

---

## 📚 参考资料

- [DIN: Deep Interest Network (阿里, 2018)](https://arxiv.org/abs/1706.06978)
- [DIEN: Deep Interest Evolution Network (阿里, 2019)](https://arxiv.org/abs/1809.03672)
- [KuaiRec Dataset](https://kuairec.com)
- [YouTube DNN (Google, 2016)](https://dl.acm.org/doi/10.1145/2959100.2959190)

---

## 📄 License

MIT License
