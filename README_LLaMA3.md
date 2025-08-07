# LLaMA3 支持说明

本文档说明如何在SALMONN项目中使用LLaMA3-8B-Instruct模型替代原来的Vicuna模型。

## 主要修改内容

### 1. 配置文件修改

- 新增了 `config_llama3.yaml` 配置文件，专门用于LLaMA3模型
- 添加了以下新的配置参数：
  - `model_type`: 模型类型，可选 "vicuna" 或 "llama3"
  - `llama3_path`: LLaMA3模型路径
  - `use_bf16`: 是否使用bf16精度（LLaMA3推荐使用）

### 2. 模型代码修改

- 修改了 `SALMONN` 类的 `__init__` 方法，添加对LLaMA3的支持
- 更新了 `maybe_autocast` 方法，支持bf16精度
- 改进了tokenizer加载逻辑，对LLaMA3使用 `AutoTokenizer`
- 修改了 `from_config` 类方法，支持新的配置参数
- **新增**: 更新了 `generate` 方法的停止条件，支持LLaMA3的特殊token
- **新增**: 添加了 `_get_stop_token_ids` 辅助方法，自动检测合适的停止token
- **新增**: 改进了文本解码，自动清理LLaMA3的特殊token

### 3. 数据类型变更

- Vicuna模型使用 `torch.float16`
- LLaMA3模型使用 `torch.bfloat16`（当 `use_bf16=True` 时）

## 使用方法

### 配置LLaMA3模型

1. 编辑 `configs/config_llama3.yaml` 文件
2. 设置正确的 `llama3_path` 路径：
   ```yaml
   model:
     model_type: "llama3"
     llama3_path: "/path/to/your/llama3-8b-instruct"  # 替换为实际路径
     use_bf16: True
   ```

3. 根据需要调整其他配置参数，特别是：
   - `prompt_template`: LLaMA3使用不同的对话模板
   - `end_sym`: LLaMA3的结束符号

### 训练命令

使用LLaMA3配置进行训练：
```bash
python train.py --config configs/config_llama3.yaml
```

### 向后兼容性

原有的Vicuna配置仍然可以正常使用：
```bash
python train.py --config configs/config.yaml
```

## 注意事项

1. **模型路径**: 确保LLaMA3模型路径正确，模型应该是HuggingFace格式
2. **内存需求**: bf16相比fp16可能会有不同的内存占用
3. **对话模板**: LLaMA3使用不同的对话模板，已在配置文件中预设
4. **特殊token**: LLaMA3有自己的特殊token体系，代码已自动处理
5. **停止条件**: LLaMA3使用 `<|eot_id|>` 和 `<|end_of_text|>` 作为停止token
6. **文本清理**: 生成的文本会自动清理LLaMA3的特殊标记

## 配置参数对比

| 参数 | Vicuna | LLaMA3 |
|------|--------|--------|
| `model_type` | "vicuna" | "llama3" |
| `torch_dtype` | `torch.float16` | `torch.bfloat16` |
| `prompt_template` | "USER: {}\nASSISTANT:" | LLaMA3 chat template |
| `end_sym` | "</s>" | "<\|eot_id\|>" |
| tokenizer | `LlamaTokenizer` | `AutoTokenizer` |
| stop_tokens | EOS (ID: 2) | `<\|eot_id\|>`, `<\|end_of_text\|>` |

## 停止条件详细说明

### LLaMA3 停止条件
- 主要停止token: `<|eot_id|>` (End of Turn)
- 备用停止token: `<|end_of_text|>` 
- 自动检测tokenizer中的停止token ID
- 如果检测失败，使用常见的LLaMA3 token ID作为fallback

### Vicuna 停止条件
- 使用标准的EOS token (通常是ID 2)
- 保持原有的停止逻辑不变

## 故障排除

如果遇到问题：
1. 检查LLaMA3模型路径是否正确
2. 确认GPU支持bf16（较新的GPU如A100、H100等）
3. 检查transformers库版本是否支持LLaMA3
4. 查看日志中的模型加载信息
5. **新增**: 如果生成无法正确停止，检查停止token是否正确识别
6. **新增**: 可以运行 `python test_stop_tokens.py` 来测试停止token检测

## 测试工具

项目中包含了 `test_stop_tokens.py` 脚本，用于：
- 测试不同模型的停止token检测
- 验证SALMONN的停止逻辑
- 调试生成过程中的停止问题

使用方法：
```bash
cd /cpfs02/user/liuzhan/Project/SALMONN
python test_stop_tokens.py
```
