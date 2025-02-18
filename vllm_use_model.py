from vllm import LLM, SamplingParams

# 模型目录
model_dir = '/media/shuozhang/PKUWorking/model_01/OpenBMB/MiniCPM-2B-sft-fp32'


# 初始化LLM模型
max_model_len = 2048  # 模型最大长度
llm = LLM(model=model_dir, max_model_len=max_model_len,trust_remote_code=True)

# 初始化采样参数
temperature = 0.8  # 温度参数，用于控制输出的随机性
top_p = 0.95  # 核采样比例，只选择累积概率达到此值的最小概率的token
max_tokens = 1024  # 最大生成token数
sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)


def chat():
    """
    聊天函数，循环读取用户输入并生成AI回复
    """
    print("您好，请输入您的问题：")
    while True:
        print("User: ", end="")
        user_input = input()
        if user_input == "exit":
            break
        print("AI: ", end="")
        # 生成AI回复
        output = llm.generate([user_input], sampling_params,use_tqdm=False)
        print(output[0].outputs[0].text)

if __name__ == "__main__":
    chat()