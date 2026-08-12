# Seal OCR

一个面向中文印章公司名识别的轻量 TrOCR 训练项目，覆盖：

- 真实数据准备与固定切分
- 公司名单词表初始化
- 印章合成与空间标注
- 合成数据预训练
- 真实标注后训练
- 一键评估
- encoder / decoder 双 ONNX + vocab 三文件打包

项目只保留两种训练阶段：

| 阶段 | 数据 | 默认学习率 | 用途 |
| --- | --- | ---: | --- |
| `pretrain` | 合成印章 | `2e-4` | 学习字符、形制、颜色、方向和扫描退化 |
| `finetune` | 真实标注 | `5e-6` | 适配真实扫描、盖章与业务分布 |

长度控制、空间辅助监督、颜色一致性和预训练定向退化均使用代码内固定配置，不再作为命令行实验参数。训练日志只保留主要 `loss`、`grad_norm`、`learning_rate`、`epoch`，以及验证集的 `loss / CER / exact match`。

## 项目结构

```text
seal_ocr/
├── prepare.py                 # 一键生成词表并初始化模型
├── train.py                   # 预训练 / 真实数据后训练
├── evaluate.py                # 一键测试 PyTorch 模型
├── export_onnx.py             # 导出三文件部署包
├── infer_onnx.py              # ONNX 参考推理
├── seal_ocr/                  # 数据、增强、模型辅助头、评估实现
├── synthesis/
│   ├── generate.py            # 唯一合成入口
│   ├── company_names.txt      # 完整公司清单，每行一个名称
│   ├── fonts/                 # 用户放置有授权的中文字体
│   └── backgrounds/           # 用户放置真实单据背景图，可为空
├── tests/
├── requirements.txt
└── README.md
```

权重、真实数据、合成结果、字体、背景图、评估报告与 ONNX 文件均不提交到 Git。

## 1. 安装

建议 Python 3.10–3.12。先按训练机 CUDA 版本安装官方 PyTorch，再安装其余依赖：

```bash
python -m venv .venv
source .venv/bin/activate

# 先从 https://pytorch.org/get-started/locally/ 选择正确的 CUDA 命令
python -m pip install -U pip
python -m pip install -r requirements.txt
```

开发机可只做静态检查；完整训练、CUDA、ONNX 数值对齐结论应在实际训练服务器上验证。

## 2. 数据格式

图片与同名 UTF-8 文本一一对应，支持递归子目录：

```text
data/real/train/
├── 000001.jpg
├── 000001.txt     # 上海示例科技有限公司
├── 000002.png
└── 000002.txt
```

标签文件只写模型最终需要输出的文字，不写 JSON，不附加说明。建议把真实数据预先拆成：

```text
data/real/
├── train/
├── val/
└── test/
```

最终测试集在整个调参过程中保持冻结。训练入口还会检查缺标签、损坏图片、跨集合重复图片、公司名泄漏和词表缺字。

## 3. 准备完整公司清单并一键初始化

编辑 [`synthesis/company_names.txt`](synthesis/company_names.txt)，每行一个完整公司名；以 `#` 开头的行是说明。初始化时会合并公司清单与真实标签中的字符：

```bash
python prepare.py \
  --base-model ./models/base-trocr \
  --company-list synthesis/company_names.txt \
  --real-data data/real/train data/real/val data/real/test \
  --output models/init \
  --image-size 384
```

`--base-model` 可以是本地 TrOCR 目录，也可以是可访问的 Hugging Face 模型名。若基础模型不含业务字符，初始化会默认停止；确认接受随机初始化这些字符后才使用 `--allow-unknown-chars`。

## 4. 准备字体、背景和合成参数

把有合法使用权、且覆盖公司清单全部字符的字体放入 `synthesis/fonts/`。项目不附带第三方字体。

背景图放入 `synthesis/backgrounds/`，可按任意子目录组织。目录为空时仍可生成带轻微纸张底噪的空白背景。

默认合成分布：

| 维度 | 默认值 |
| --- | --- |
| 颜色 | 红 89%、蓝 7%、紫 1%、黑 3% |
| 形状 | 圆形 90%、椭圆 7%、矩形 3% |
| 版式 | 标准 90%、双语双环 6%、椭圆业务章 2%、多行中心章 2% |
| 空白纸面 | 58% |
| 大角度旋转 | 25%，绝对角度 45°–180° |
| 正向/小倾角 | 约 75% |

先只校验资源、词表和预计数量：

```bash
python -m synthesis.generate --validate-only
```

正式生成：

```bash
python -m synthesis.generate \
  --company-list synthesis/company_names.txt \
  --vocab models/init/vocab.json \
  --fonts synthesis/fonts \
  --backgrounds synthesis/backgrounds \
  --output data/synthetic \
  --samples-per-name 14 \
  --exclude-labels data/real/val \
  --exclude-labels data/real/test
```

按真实业务调整时，只需要覆盖需要改变的分布：

```bash
python -m synthesis.generate \
  --color-ratios red=80,blue=15,purple=2,black=3 \
  --shape-ratios circle=75,ellipse=20,rectangle=5 \
  --layout-ratios standard=85,bilingual_ring=8,oval_service=4,multiline_center=3 \
  --blank-background-ratio 0.45 \
  --rotation-probability 0.30 \
  --rotation-min 45 \
  --rotation-max 180
```

生成结果写入单一图片目录 `data/synthetic/`；空间监督写入同级 `data/synthetic_spatial/`，两者保持相同分片和文件名。每张图片还包含同名 `.meta.json`，便于审计实际颜色、形状、版式和旋转分布。

## 5. 合成预训练

单卡：

```bash
python train.py \
  --stage pretrain \
  --model models/init \
  --data data/synthetic \
  --spatial-data data/synthetic_spatial \
  --output checkpoints/pretrain
```

多卡：

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --stage pretrain \
  --model models/init \
  --data data/synthetic \
  --spatial-data data/synthetic_spatial \
  --output checkpoints/pretrain
```

默认学习率是 `2e-4`。常用可调项只保留 batch、epoch、梯度累积、验证间隔、精度、显存检查点和续训：

```bash
python train.py --help
```

最佳模型保存到 `checkpoints/pretrain/best/`。

## 6. 用真实条件校验合成学习效果

在不参与预训练的真实验证集上测试：

```bash
python evaluate.py \
  --model checkpoints/pretrain/best \
  --data data/real/val \
  --output reports/pretrain_on_real
```

重点查看：

- `exact_match_accuracy`：整行公司名完全正确率
- `cer`：字符错误率
- `by_label_length`：长名称是否明显退化
- `output_length_metrics`：漏字、超长、长度偏差
- `wrong_top_20` 和 `predictions.csv`：真实错误样本

如果真实错误集中在某种颜色、形状、版式、背景密度或旋转角度，调整合成参数后重新生成和预训练。不要用最终测试集做这一步。

## 7. 真实标注后训练

```bash
python train.py \
  --stage finetune \
  --model checkpoints/pretrain/best \
  --data data/real/train \
  --val-data data/real/val \
  --test-data data/real/test \
  --output checkpoints/finetune
```

默认学习率是 `5e-6`。若真实数据很少，可选用少量合成回放防止遗忘：

```bash
python train.py \
  --stage finetune \
  --model checkpoints/pretrain/best \
  --data data/real/train \
  --val-data data/real/val \
  --test-data data/real/test \
  --replay-data data/synthetic \
  --replay-ratio 0.20 \
  --output checkpoints/finetune
```

最佳模型保存到 `checkpoints/finetune/best/`。

## 8. 一键测试最终模型

```bash
python evaluate.py \
  --model checkpoints/finetune/best \
  --data data/real/test \
  --output reports/final_test
```

也可以使用训练时冻结的数据清单复现同一集合：

```bash
python evaluate.py \
  --model checkpoints/finetune/best \
  --manifest checkpoints/finetune/data_split.json \
  --split test \
  --output reports/final_test
```

## 9. 模型打包与 ONNX 推理

导出器会执行结构检查和 PyTorch / ONNX Runtime 数值对齐。最终目录严格只有三个文件：

```bash
python export_onnx.py \
  --model checkpoints/finetune/best \
  --output deploy/seal_ocr
```

```text
deploy/seal_ocr/
├── encoder_model.onnx
├── decoder_model.onnx
└── vocab.json
```

推理：

```bash
python infer_onnx.py \
  --model deploy/seal_ocr \
  --image path/to/seal.jpg
```

空间辅助头只服务训练，不进入部署包；长度预测头在 encoder ONNX 中，长度条件在 decoder ONNX 中。

## 测试

不依赖 GPU 的检查：

```bash
python -m compileall -q train.py prepare.py evaluate.py export_onnx.py infer_onnx.py seal_ocr synthesis
python -m unittest discover -s tests -v
```

完整训练和真实 ONNX 导出仍需在安装了 PyTorch、ONNX、ONNX Runtime 的目标服务器执行。

## 数据与模型边界

仓库不提供：

- 真实印章图片或标签
- 训练权重和 ONNX 权重
- 字体文件
- 单据背景图片

请确认公司清单、字体、背景图、真实数据和基础模型各自的授权范围后再发布或商用。

## 许可与致谢

本项目采用 [MIT License](LICENSE) 开源。

项目开发参考了 [Gmgge/TrOCR-Seal-Recognition](https://github.com/Gmgge/TrOCR-Seal-Recognition)、[Microsoft TrOCR](https://github.com/microsoft/unilm/tree/master/trocr) 和 [Hugging Face Transformers](https://github.com/huggingface/transformers)，感谢相关项目及其贡献者。
