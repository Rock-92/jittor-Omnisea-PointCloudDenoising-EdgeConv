# Point Cloud Denoising Baseline

本仓库是计图比赛点云降噪赛题 baseline。任务目标是给定含噪点云，
预测逐点位移向量，将点推回真实物体表面附近。模型基于 Dynamic Edge
Convolution 提取局部几何特征，并使用 MLP 解码器回归三维位移。

## 项目结构

```text
.
├── configs/          # 数据、模型、训练和推理配置
├── data/             # 数据集说明，不提交原始数据
├── datalist/         # 训练、验证、测试样本列表
├── scripts/          # 训练、推理、评测入口脚本
├── src/              # 核心代码
│   ├── data/         # 数据读取、采样、增强和 patch 构建
│   ├── model/        # EdgeConv 模型和位移预测逻辑
│   └── system/       # 训练、推理、保存结果流程
├── tools/            # 数据处理或统计工具
├── outputs/          # 日志、权重和预测结果，默认不提交
├── evaluate.py       # 本地评测实现
└── run.py            # 兼容原 baseline 的统一入口
```

## 环境安装

推荐使用 Python 3.9。

```bash
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 libgomp -y
python -m pip install -r requirements.txt
python -m pip install point-cloud-utils
```

`point-cloud-utils` 用于更精确地计算 Point-to-Surface Distance。如果只跑训练或推理，可以先不安装。

## 数据准备

训练集解压到仓库根目录：

```bash
tar xzf dataset_clean.tar.gz
```

期望结构：

```text
dataset_clean/
└── shapenet/
    └── <synset_id>/
        └── <model_id>/
            └── models/
                └── model_normalized.obj
```

测试集解压到仓库根目录：

```bash
unzip test_noisy.zip
```

期望结构：

```text
test_noisy/
└── shapenet/
    └── <synset_id>/
        └── <model_id>/
            └── noisy.npy
```

数据根目录在 [configs/data/train.yaml](configs/data/train.yaml) 和
[configs/data/predict.yaml](configs/data/predict.yaml) 中配置。

## 训练

```bash
python scripts/train.py --config configs/task/train_vm.yaml --seed 123
```

训练权重默认保存到 `outputs/experiments/vm/`。原始入口仍可使用：

```bash
python run.py --task configs/task/train_vm.yaml --seed 123
```

训练配置默认会在每个 epoch 结束后基于验证集输出榜单式
`final_score`、`cd_score` 和 `p2s_score`。相关开关在
[configs/task/train_vm.yaml](configs/task/train_vm.yaml) 的 `trainer`
字段中，包括 `log_score`、`score_every_n_epochs` 和 `score_max_samples`。
当验证集 `final_score` 刷新最高值时，会额外保存
`outputs/experiments/vm/checkpoint_best.pkl`，并写入
`outputs/experiments/vm/checkpoint_best.txt` 记录 best epoch 和分数。
训练结束后默认会加载 `checkpoint_best.pkl` 在 `test_noisy/` 上推理，
并自动生成提交包 `outputs/result.zip`。如果只想训练不打包，可以在
[configs/task/train_vm.yaml](configs/task/train_vm.yaml) 中将
`create_submission_on_train_end` 改为 `False`。

## 推理

默认使用 `outputs/experiments/vm/checkpoint_best.pkl` 推理。若要使用其他权重，
先在 [configs/task/predict_vm.yaml](configs/task/predict_vm.yaml) 中修改
`load_ckpt`，然后运行：

```bash
python scripts/infer.py --config configs/task/predict_vm.yaml --seed 123
```

预测结果默认保存到 `outputs/results/`，文件名为
`denoised.npy`，目录结构与测试集保持一致。

打包提交：

```bash
cd outputs/results
zip -r ../result.zip shapenet/
```

提交包结构：

```text
result.zip
└── shapenet/
    └── <synset_id>/
        └── <model_id>/
            └── denoised.npy
```

## 评测

如果拥有本地真值和 mesh，可以运行：

```bash
python scripts/eval.py \
  --pred_dir outputs/results \
  --gt_dir test_gt \
  --noisy_dir test_noisy \
  --mesh_dir dataset_clean \
  --workers 8
```

指标包括 Chamfer Distance (CD) 和 Point-to-Surface Distance (P2S)。评分会分别比较预测点云与含噪点云相对真值或原始网格的改善比例，并映射到百分制。

## 可复现说明

- `run.py` 提供 `--seed`，会统一设置 Jittor、NumPy 和 Python 随机种子。
- 关键训练参数位于 `configs/`，包括学习率、batch size、epoch、数据路径和模型结构。
- `outputs/`、数据集目录、权重和中间结果默认不提交。

## 引用

Starter Code 的方法设计参考：

Dasith de Silva Edirimuni, Xuequan Lu, Gang Li, Lei Wei, Antonio Robles-Kelly, Hongdong Li.
StraightPCF: Straight Point Cloud Filtering. CVPR 2024, pp. 20721-20730.
