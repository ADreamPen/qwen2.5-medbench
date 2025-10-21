
import json
import re

def process_imcs_to_qa_format(input_file, output_file, num_cases=40):
    """
    将IMCS JSON文件处理成QA对格式,提示不包含few shot
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
        num_cases: 处理的病例数量
    """
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    qa_pairs = []
    
    # 处理前num_cases个病例
    for case_id, case_data in list(data.items())[:num_cases]:
        # 构建对话文本
        dialogue_text = build_dialogue_text(case_data['dialogue'])
        
        # 构建标准报告（使用第一个报告版本）
        if case_data['report']:
            standard_report = case_data['report'][0]
        else:
            # 如果没有报告，创建一个空报告
            standard_report = {
                "主诉": "",
                "现病史": "", 
                "辅助检查": "",
                "既往史": "",
                "诊断": "",
                "建议": ""
            }
        
        # 构建QA对
        qa_pair = {
            "question": build_question(dialogue_text),
            "answer": build_answer(standard_report),
            "other": {
                "source": "IMCS-V2-MRG",
                "id": len(qa_pairs) + 1
            }
        }
        
        qa_pairs.append(qa_pair)
    
    # 保存处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    print(f"成功处理 {len(qa_pairs)} 个QA对，已保存到 {output_file}")
    return qa_pairs

def build_dialogue_text(dialogue):
    """构建对话文本"""
    dialogue_lines = []
    for utterance in dialogue:
        speaker = utterance['speaker']
        sentence = utterance['sentence'].strip()
        
        # 过滤空句子和无效内容
        if not sentence or sentence in ['（空）', '?', '。', '嗯', '哦', '啊']:
            continue
            
        dialogue_lines.append(f"{speaker}：{sentence}")
    
    return "\n".join(dialogue_lines)

def build_question(dialogue_text):
    """构建问题部分"""
    question_template = """## 任务：请根据以下医患对话记录，为患者总结问诊报告。输出格式为：```json{"主诉": "", "现病史": "", "辅助检查": "", "既往史": "", "诊断": "", "建议": ""}```

## 要求：只输出json，不要输出其他内容。

"""

    return question_template + f"\n\n## 问诊对话：\n{dialogue_text}\n\n问诊报告："

def build_answer(report):
    """构建答案部分"""
    # 将报告转换为JSON字符串格式
    report_json = {
        "主诉": report.get("主诉", ""),
        "现病史": report.get("现病史", ""),
        "辅助检查": report.get("辅助检查", ""),
        "既往史": report.get("既往史", ""),
        "诊断": report.get("诊断", ""),
        "建议": report.get("建议", "")
    }
    
    return json.dumps(report_json, ensure_ascii=False)

def preview_qa_pairs(qa_pairs, num_preview=3):
    """预览生成的QA对"""
    print("=" * 80)
    print("预览前3个生成的QA对：")
    print("=" * 80)
    
    for i, qa in enumerate(qa_pairs[:num_preview]):
        print(f"\n第 {i+1} 个QA对：")
        print(f"ID: {qa['other']['id']}")
        print(f"Source: {qa['other']['source']}")
        
        # 截取问题的一部分进行预览
        question_preview = qa['question'][-500:]  # 显示最后500个字符
        print(f"问题预览: ...{question_preview}")
        
        print(f"答案: {qa['answer']}")
        print("-" * 60)

if __name__ == "__main__":
    input_file = "IMCS-V2_dev.json"  # 文件路径
    output_file = "./test_data/imcs_qa_pairs_wo_few_shot.json"
    
    # 处理数据
    qa_pairs = process_imcs_to_qa_format(input_file, output_file, num_cases=40)
    
    # 预览结果
    preview_qa_pairs(qa_pairs)
    
    # 输出统计信息
    print(f"\n处理完成！")
    print(f"总QA对数量: {len(qa_pairs)}")
    print(f"输出文件: {output_file}")

    