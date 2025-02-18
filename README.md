# llm_finetune
- 部署流程：
-- 首先运行download.py下载模型

-- 而后运行vllm_use_model.py测试，可以进行对话


- 微调流程
-- 在终端中运行以下命令(详见test.ipynb)
chmod +x train.sh
./train.sh


- 测试流程
-- 运行llm.py
  new为False和True分别代表微调前后的模型
