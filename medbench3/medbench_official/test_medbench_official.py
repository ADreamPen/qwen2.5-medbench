

import json
import requests
import time
# from openai import OpenAI
from modelscope import AutoModelForCausalLM, AutoTokenizer
import os

# model_name = "/data1/shenjiangu/model_weights/qwen2.5-7b/"

def simple_evaluate():
    """最简单的评测函数"""
    # 配置信息
    # model_name = "qwen2.5-7b" 
    # model_name = "deepseek-chat" 
    # model_name = "/data1/shenjiangu/model_weights/qwen2.5-7b/"
    model_name = "/data1/shenjiangu/model_weights/qwen2.5-14b/"
    input_file = "IMCS-V2-MRG_test.jsonl"
    # output_file = "results_7b.jsonl"
    output_file = "results_14b.jsonl"
    
    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    results = []
    
    print(f"开始处理 {len(lines)} 条数据...")

    # client = OpenAI(
    #     base_url = 'https://localhost:11434/v1',
    #     api_key = 'ollama'
    # )

    # client = OpenAI(
    #     base_url = "https://api.deepseek.com/v1",
    #     api_key = "sk-57096aa19f3f4892b6c0df1b89a0ff89"
    # )
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    low_cpu_mem_usage=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    for i, line in enumerate(lines):
        data = json.loads(line)
        
        # 如果已经有答案，跳过
        if data.get("answer") is not None:
            results.append(data)
            continue

        prompt = data['question']
        messages = [
            {"role":"user", "content":prompt}
        ]
        text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True)

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=16384
        )

        generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in
        zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if response != None:
            ans = response
            data['answer'] = ans

        else:
            data['answer'] = ''
            print(f"第{i+1}条出错:{e}")
        
        results.append(data)
        
        # 保存进度
        if (i + 1) % 5 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"💾 已保存 {i+1} 条结果")
        
        # 延迟
        time.sleep(0.5)
    
    # 最终保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"🎉 全部完成! 结果保存在 {output_file}")

if __name__ == "__main__":
    simple_evaluate()