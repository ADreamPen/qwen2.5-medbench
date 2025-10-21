
import json
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time
import os

class LLMTester:
    def __init__(self, model_path):
        """初始化模型和tokenizer"""

        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

        print(f"正在加载模型: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("模型加载完成！")
    
    def generate_response(self, prompt, max_new_tokens=16384):
        """生成模型响应"""
        try:
            messages = [{"role": "user", "content": prompt}]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in
                zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return response.strip()
            
        except Exception as e:
            print(f"生成响应时出错: {e}")
            return f"错误: {str(e)}"
    
    def extract_json_from_response(self, response):
        """从模型响应中提取JSON内容"""
        try:
            # 尝试直接解析整个响应
            if response.strip().startswith('{') and response.strip().endswith('}'):
                return json.loads(response.strip())
            
            # 尝试从响应中提取JSON部分
            import re
            json_pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            if matches:
                # 选择最长的匹配（可能是最完整的JSON）
                longest_match = max(matches, key=len)
                return json.loads(longest_match)
            else:
                # 如果没有找到JSON，返回原始响应
                return {"raw_response": response}
                
        except json.JSONDecodeError:
            # 如果JSON解析失败，返回原始响应
            return {"raw_response": response}
        except Exception as e:
            return {"error": str(e), "raw_response": response}

def test_model_on_dataset(model_path, test_data_path, output_path, num_samples=None):
    """
    在测试数据集上测试模型
    
    Args:
        model_path: 模型路径
        test_data_path: 测试数据路径
        output_path: 输出结果路径
        num_samples: 测试样本数量（None表示全部）
    """
    
    # 加载测试数据
    print(f"加载测试数据: {test_data_path}")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if num_samples:
        test_data = test_data[:num_samples]
        print(f"测试前 {num_samples} 个样本")
    else:
        print(f"测试全部 {len(test_data)} 个样本")
    
    # 初始化模型
    tester = LLMTester(model_path)
    
    results = []
    
    # 遍历测试数据
    for i, item in enumerate(tqdm(test_data, desc="测试进度")):
        try:
            # 复制原始数据
            result_item = item.copy()
            
            # 获取问题
            question = item["question"]
            
            # 生成响应
            print(f"\n正在处理第 {i+1}/{len(test_data)} 个样本...")
            start_time = time.time()
            
            response = tester.generate_response(question, max_new_tokens=2048)
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            print(f"生成时间: {generation_time:.2f}秒")
            print(f"模型响应: {response[:200]}...")  # 只打印前200个字符
            
            # 提取JSON响应
            model_answer = tester.extract_json_from_response(response)
            
            # 更新结果
            result_item["model_response"] = response
            result_item["model_answer"] = model_answer
            result_item["generation_time"] = generation_time
            
            results.append(result_item)
            
            # 每处理10个样本保存一次中间结果
            if (i + 1) % 10 == 0:
                with open(output_path.replace('.json', '_temp.json'), 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"已保存中间结果到 {output_path.replace('.json', '_temp.json')}")
                
        except Exception as e:
            print(f"处理第 {i+1} 个样本时出错: {e}")
            # 保存错误信息
            error_item = item.copy()
            error_item["error"] = str(e)
            error_item["model_response"] = ""
            error_item["model_answer"] = {}
            error_item["generation_time"] = 0
            results.append(error_item)
    
    # 保存最终结果
    print(f"\n保存最终结果到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 输出统计信息
    total_time = sum(item.get("generation_time", 0) for item in results)
    avg_time = total_time / len(results) if results else 0
    
    print(f"\n测试完成！")
    print(f"总样本数: {len(results)}")
    print(f"总生成时间: {total_time:.2f}秒")
    print(f"平均生成时间: {avg_time:.2f}秒/样本")
    
    return results

def analyze_results(results_path):
    """分析测试结果"""
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"\n结果分析:")
    print(f"总测试样本: {len(results)}")
    
    # 统计成功提取JSON的数量
    valid_json_count = 0
    error_count = 0
    
    for item in results:
        model_answer = item.get("model_answer", {})
        if isinstance(model_answer, dict) and "raw_response" not in model_answer and "error" not in model_answer:
            valid_json_count += 1
        if "error" in item:
            error_count += 1
    
    print(f"有效JSON响应: {valid_json_count}/{len(results)} ({valid_json_count/len(results)*100:.1f}%)")
    print(f"错误数量: {error_count}")
    
    # 显示前几个结果示例
    print(f"\n前3个结果示例:")
    for i, item in enumerate(results[:3]):
        print(f"\n示例 {i+1}:")
        print(f"问题长度: {len(item['question'])} 字符")
        print(f"生成时间: {item.get('generation_time', 0):.2f}秒")
        print(f"模型答案: {item.get('model_answer', {})}")

# 使用示例
if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = "/data1/shenjiangu/model_weights/qwen2.5-7b/"  # 模型权重路径
    # MODEL_PATH = "/data1/shenjiangu/model_weights/qwen2.5-14b/"  # 模型权重路径

    # TEST_DATA_PATH = "./test_data/imcs_qa_pairs.json"  # 测试数据路径
    # TEST_DATA_PATH = "./test_data/imcs_qa_pairs_wo_few_shot.json"  # 测试数据路径
    TEST_DATA_PATH = "./test_data/imcs_qa_pairs_cot.json"  # 测试数据路径

    # OUTPUT_PATH = "./results/test_results.json"
    # OUTPUT_PATH = "./results/test_results_14b.json"
    OUTPUT_PATH = "./results/test_results_7b_cot.json"
    
    # 测试模型（可以设置num_samples来限制测试数量）
    results = test_model_on_dataset(
        model_path=MODEL_PATH,
        test_data_path=TEST_DATA_PATH,
        output_path=OUTPUT_PATH,
        num_samples=None  # 先测试10个样本，如果顺利可以设置为None测试全部
    )
    
    # 分析结果
    analyze_results(OUTPUT_PATH)