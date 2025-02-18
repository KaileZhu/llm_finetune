from transformers import AutoModelForCausalLM, AutoTokenizer  # 从transformers库导入所需的类
import torch  # 导入torch库，用于深度学习相关操作
from safetensors.torch import load_file

# torch.manual_seed(0)  # 设置随机种子以确保结果的可复现性
new = False  # True表示加载微调后的模型，False表示加载原始模型

T=0.5  # 这是模型温度
top_p = 0.8  # 这是top_p
repetition_penalty = 1.02  # 这是重复惩罚因子
user_input = "山东省最高的山是哪座山, 它比黄山高还是矮？差距多少？"  # 这是用户的输入
prompt = ''  # 在这里进行prompt engineering

# 定义模型路径
path = './OpenBMB/MiniCPM-2B-sft-fp32'
#调参路径
path_for_weight = './output/MiniCPM'

# 从模型路径加载分词器，
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, trust_remote_code=True)
# 从模型路径加载模型，设置为使用bfloat16精度以优化性能，并将模型部署到支持CUDA的GPU上,trust_remote_code=True允许加载远程代码
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True)

if new:
    # 加载权重，使用safetensors库
    state_dict = load_file(f"{path_for_weight}/model-00001-of-00002.safetensors")
    model.load_state_dict(state_dict, strict=False)

# 将模型部署到支持CUDA的GPU上
model.to('cuda')

# 使用模型进行聊天，提出问题并设置生成参数，如temperature、top_p值和repetition_penalty（重复惩罚因子）
responds, history = model.chat(tokenizer, prompt + user_input, temperature=T, top_p=top_p, repetition_penalty=repetition_penalty)

# 显示生成的回答
if new:
    print('微调大模型的回答为：')
else:
    print('原来大模型的回答为：')
print(responds)