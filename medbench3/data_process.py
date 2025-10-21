
import json
import re

def process_imcs_to_qa_format(input_file, output_file, num_cases=40):
    """
    将IMCS JSON文件处理成QA对格式
    
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

## 任务示例：问诊对话：
患者：四个月大的宝宝感冒，无发烧，现在只有小儿氨酚黄那敏颗粒跟双黄连颗粒，请问可以吃吗？怎么吃？孩子母亲感冒有咳嗽是否也可以吃？
医生：您好。孩子有没有受凉引起的感冒？
医生：是流涕，鼻塞吗？
患者：声音哑了
患者：有一些流涕
医生：哦，咳嗽吗？精神好吗？
患者：孩子妈妈感冒几天了，纯母乳，目前不知道是受凉还是母亲传染
患者：昨天开始有些没神
患者：医生，主要他们现在在国外，目前药物就上述那两种
医生：哦，应该是家属传染的
医生：只要没有发热，没有咳嗽，就说明病情不严重。不要担心
医生：是大夫开的药物吗？
患者：不是
患者：是自己家里之前买了寄过去的
患者：孩子的母亲有咳嗽，可以吃吗
医生：哦，小儿氨酚黄那敏颗粒最好是1岁以上开的。目前不建议吃
患者：那孩子母亲呢，纯母乳的
患者：母亲感冒有咳嗽
医生：双黄连颗粒是可以吃的，这个清热解毒，抗病毒消炎。宝宝可以吃
患者：好的
患者：医生，孩子妈妈可以吃吗
医生：剂量的话
患者：什么
医生：每次三分之一包，一日2次
医生：妈妈也可以吃双黄连的，
患者：好
患者：好吧
医生：只要孩子精神可以，就说明病情不严重
患者：宝宝昨天开始有些没神
患者：老是打喷嚏
医生：这个打喷嚏的话。也是感冒的症状，可以吃氯雷他定口服液治疗
医生：这个药物可以抗过敏，也可以治疗流涕，打喷嚏
患者：没有这个药
医生：每次2ml，一天1次
医生：一般药店就有
医生：假如出现发热，出现咳嗽的表现，就要及时去医院看看。因为孩子小，怕病情加重
患者：在国外
医生：国外呀。
患者：是的
医生：一般来说打喷嚏流鼻涕的表现，就是不加上治疗打喷嚏的药物，也能缓解。建议先吃上双黄连看看
患者：好的，谢谢
医生：客气啦
医生：有问题留言
医生：还有问题吗？
患者：没有了谢谢

问诊报告：{"主诉": "流涕、声音哑、打喷嚏。", "现病史": "患儿四个月大，有流涕和声音哑，昨天开始精神不佳，老是打喷嚏，无发烧。孩子母亲感冒有咳嗽，母乳喂养。", "辅助检查": "无", "既往史": "无", "诊断": "小儿感冒", "建议": "不建议使用小儿氨酚黄那敏颗粒，可以给宝宝吃双黄连颗粒，每次三分之一包，一日2次。如果宝宝出现发热或咳嗽，应及时就医。可以考虑使用氯雷他定口服液，可以治疗流涕，打喷嚏。"}

## 问诊对话：
医生：孩子感冒有多久了？
患者：感冒3天，体温38.5度。
医生：有做过什么检查吗？
患者：血常规显示白细胞数值和中性粒细胞比例偏高。
医生：既往有什么疾病吗？
患者：没有。
医生：考虑小儿感冒。
医生：建议去公立小儿科就医。

问诊报告："""

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
    output_file = "./test_data/imcs_qa_pairs.json"
    
    # 处理数据
    qa_pairs = process_imcs_to_qa_format(input_file, output_file, num_cases=40)
    
    # 预览结果
    preview_qa_pairs(qa_pairs)
    
    # 输出统计信息
    print(f"\n处理完成！")
    print(f"总QA对数量: {len(qa_pairs)}")
    print(f"输出文件: {output_file}")

    