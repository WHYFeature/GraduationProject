# GraduationProject

本项目用于复现、分析和扩展 MEMIT，并支持通过独立的模型配置文件适配不同结构的自回归语言模型。

## 项目结构

- `configs/models/`
  存放模型结构配置。这里描述模型的层命名、嵌入层、MLP 输出层、最终归一化层等信息。
- `src/memit_project/algorithms/memit/*.json`
  存放 MEMIT 的算法超参数，例如编辑层范围、优化步数、学习率、KL 系数等。
- `src/memit_project/evaluation/runner.py`
  主运行入口。运行时会先读取 MEMIT 超参数，再读取模型配置，并用模型配置覆盖其中与模型结构相关的字段。

## 现在的配置机制

模型适配信息已经从运行代码中抽离到 `configs/models/*.yml`。

运行时：

1. 读取 MEMIT 算法超参数文件，例如 `Qwen3-4B.json`
2. 读取模型配置文件，例如 `configs/models/qwen3-4b.yml`
3. 用模型配置覆盖算法文件中的这些结构字段：
   - `rewrite_module_tmp`
   - `layer_module_tmp`
   - `mlp_module_tmp`
   - `attn_module_tmp`
   - `ln_f_module`
   - `lm_head_module`

这意味着：

- 算法文件负责“怎么编辑”
- 模型配置文件负责“模型长什么样”

## 模型配置文件怎么写

模型配置文件位于 `configs/models/`，使用 YAML 格式。

例如 `configs/models/qwen3-4b.yml`：

```yml
model_key: qwen3-4b
aliases:
  - qwen3-4b
  - qwen/qwen3-4b
hidden_size_attrs:
  - hidden_size
  - n_embd
  - d_model
context_length_attrs:
  - max_position_embeddings
  - n_positions
num_layers_attr: num_hidden_layers
embed_layer: model.embed_tokens
rewrite_module_tmp: model.layers.{}.mlp.down_proj
layer_module_tmp: model.layers.{}
mlp_module_tmp: model.layers.{}.mlp
attn_module_tmp: model.layers.{}.self_attn
ln_f_module: model.norm
lm_head_module: lm_head
```

### 各字段含义

- `model_key`
  模型配置文件的主标识。
- `aliases`
  可匹配的模型名。可以写 HuggingFace 名称，也可以写本地目录名。
- `hidden_size_attrs`
  模型配置对象里表示隐藏维度的字段名候选，按顺序尝试。
- `context_length_attrs`
  模型配置对象里表示上下文长度的字段名候选。
- `num_layers_attr`
  模型配置对象里表示层数的字段名。
- `embed_layer`
  token embedding 模块名。
- `rewrite_module_tmp`
  MEMIT 实际修改的权重层模板。
- `layer_module_tmp`
  每一层 block 的模块名模板。
- `mlp_module_tmp`
  MLP 子模块模板。
- `attn_module_tmp`
  Attention 子模块模板。
- `ln_f_module`
  最终归一化层。
- `lm_head_module`
  输出投影层。

## 如何新增一个模型

如果你要适配任意新模型，按下面步骤做。

### 1. 新建模型配置文件

在 `configs/models/` 下新建一个 yml，例如：

`configs/models/my-model.yml`

建议先复制一个最接近的已有文件，再改字段。

### 2. 填写结构字段

你至少要确认下面几个模块名：

- 嵌入层
- 每层 block 的路径
- MLP 输出投影层
- Attention 子模块
- 最终归一化层
- LM head

通常可以通过查看模型目录中的：

- `config.json`
- `model.safetensors.index.json`

来判断参数命名。

### 3. 准备 MEMIT 超参数文件

在 `src/memit_project/algorithms/memit/` 下放一个对应的 json，例如：

`MyModel.json`

这个文件主要调：

- `layers`
- `v_num_grad_steps`
- `v_lr`
- `v_loss_layer`
- `mom2_update_weight`

说明：

即使这里保留了结构字段，运行时也会优先使用 `configs/models/*.yml` 中的结构配置。

### 4. 先做因果分析选层

推荐先运行：

```powershell
python -m memit_project.evaluation.select_memit_layers `
  --model_name "你的模型路径或模型名" `
  --hparams_fname 你的超参数文件.json `
  --dataset knowns `
  --dataset_size_limit 50
```

它会输出一组推荐的连续层范围，用来更新 MEMIT 的 `layers`。

## 如何运行

### MEMIT 主评测

```powershell
python -m memit_project.evaluation.runner `
  --alg_name MEMIT `
  --model_name "d:\Desktop\Graduation_Project\Project\GraduationProject\models\Qwen3-4B" `
  --hparams_fname Qwen3-4B.json `
  --ds_name cf `
  --dataset_size_limit 1 `
  --num_edits 1 `
  --skip_generation_tests
```

### 显式指定模型配置文件

如果模型名无法自动匹配到 `configs/models/` 中的配置，可以手动传：

```powershell
python -m memit_project.evaluation.runner `
  --alg_name MEMIT `
  --model_name "你的模型路径" `
  --model_config configs/models/qwen3-4b.yml `
  --hparams_fname Qwen3-4B.json `
  --ds_name cf `
  --dataset_size_limit 1 `
  --num_edits 1 `
  --skip_generation_tests
```

### 预计算统计量

```powershell
python -m memit_project.stats.layer_stats `
  --model_name "你的模型路径" `
  --model_config configs/models/qwen3-4b.yml `
  --layers 14,15,16,17,18,19
```

## 适配模型时的建议

- 先确保模型本身能被 `transformers` 正常加载。
- 先跑 `dataset_size_limit=1` 的最小样本，确认模块路径正确。
- 如果报 `LookupError`，优先检查模型配置文件中的模块名。
- 如果报维度错误，优先检查：
  - `rewrite_module_tmp`
  - `lm_head_module`
  - `ln_f_module`
- 如果统计计算报层找不到，优先检查 `configs/models/*.yml` 里的 MLP 输出层模板。

## 当前状态

目前项目已经把“模型结构配置”从运行代码中独立出来，主链路会在运行时读取模型配置文件。
如果需要适配新模型，优先修改 `configs/models/*.yml`，而不是直接改 Python 源码。
