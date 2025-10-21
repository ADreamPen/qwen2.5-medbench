import json
import numpy as np
import torch
from bert_score import BERTScorer
from collections import defaultdict
import jieba
import re

class MedicalReportEvaluator:
    def __init__(self, model_type="bert-base-chinese"):
        """初始化评估器"""
        print("初始化BERTScore评估器...")
        self.bertscorer = BERTScorer(model_type=model_type, lang="zh")
        
        # 医疗实体类别
        self.entity_categories = {
            'symptoms': ['咳嗽', '发热', '发烧', '流涕', '鼻塞', '呕吐', '腹泻', '腹痛', '气喘', '痰', '头痛', '咽痛'],
            'diagnosis': ['感冒', '支气管炎', '肺炎', '消化不良', '上呼吸道感染', '腹泻', '支气管肺炎'],
            'treatments': ['吃药', '服药', '就医', '检查', '休息', '多喝水', '物理降温'],
            'examinations': ['血常规', '胸片', '听诊', '化验']
        }
    
    def calculate_bertscore(self, pred_texts, true_texts):
        """计算BERTScore"""
        if not pred_texts or not true_texts:
            return {
                'precision': 0.0,
                'recall': 0.0, 
                'f1': 0.0
            }
        
        try:
            P, R, F1 = self.bertscorer.score(pred_texts, true_texts)
            return {
                'precision': P.mean().item(),
                'recall': R.mean().item(),
                'f1': F1.mean().item()
            }
        except Exception as e:
            print(f"BERTScore计算错误: {e}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
    
    def extract_entities_by_category(self, text):
        """按类别提取医疗实体"""
        entities = {category: set() for category in self.entity_categories.keys()}
        
        if not text:
            return entities
        
        for category, keywords in self.entity_categories.items():
            for keyword in keywords:
                if keyword in text:
                    entities[category].add(keyword)
        
        return entities
    
    def calculate_macro_recall(self, pred_entities_list, true_entities_list):
        """计算Macro-Recall"""
        all_categories = list(self.entity_categories.keys())
        category_recalls = {category: [] for category in all_categories}
        
        for pred_entities, true_entities in zip(pred_entities_list, true_entities_list):
            for category in all_categories:
                # 确保使用集合类型
                pred_set = set(pred_entities.get(category, [])) if isinstance(pred_entities.get(category), list) else pred_entities.get(category, set())
                true_set = set(true_entities.get(category, [])) if isinstance(true_entities.get(category), list) else true_entities.get(category, set())
                
                if len(true_set) == 0:
                    # 如果真实实体为空，召回率为1（没有需要召回的内容）
                    recall = 1.0
                else:
                    # 计算该类别的召回率
                    tp = len(pred_set & true_set)
                    recall = tp / len(true_set) if len(true_set) > 0 else 0.0
                
                category_recalls[category].append(recall)
        
        # 计算每个类别的平均召回率
        category_avg_recalls = {}
        for category, recalls in category_recalls.items():
            category_avg_recalls[category] = np.mean(recalls) if recalls else 0.0
        
        # 计算Macro-Recall（所有类别的平均召回率）
        macro_recall = np.mean(list(category_avg_recalls.values()))
        
        return {
            'macro_recall': macro_recall,
            'category_recalls': category_avg_recalls,
            'detailed_recalls': category_recalls
        }
    
    def evaluate_single_pair(self, pred_text, true_text):
        """评估单个预测-真实对"""
        # BERTScore评估
        bertscore_result = self.calculate_bertscore([pred_text], [true_text])
        
        # 实体提取
        pred_entities = self.extract_entities_by_category(pred_text)
        true_entities = self.extract_entities_by_category(true_text)
        
        return {
            'bertscore': bertscore_result,
            'pred_entities': {k: list(v) for k, v in pred_entities.items()},  # 存储为列表用于JSON序列化
            'true_entities': {k: list(v) for k, v in true_entities.items()},  # 存储为列表用于JSON序列化
            'pred_text': pred_text,
            'true_text': true_text
        }
    
    def evaluate_field_level(self, pred_report, true_report):
        """字段级别的评估"""
        fields = ['主诉', '现病史', '辅助检查', '既往史', '诊断', '建议']
        field_results = {}
        
        for field in fields:
            pred_field = pred_report.get(field, '')
            true_field = true_report.get(field, '')
            
            field_eval = self.evaluate_single_pair(pred_field, true_field)
            field_results[field] = field_eval
        
        return field_results
    
    def comprehensive_evaluation(self, pred_reports, true_reports):
        """综合评估所有报告"""
        print("开始综合评估...")
        
        # 字段级别的BERTScore和实体统计
        field_bertscores = defaultdict(list)
        field_entities = defaultdict(lambda: {'pred': [], 'true': []})
        
        # 整体文本评估（合并所有字段）
        overall_pred_texts = []
        overall_true_texts = []
        overall_pred_entities = []
        overall_true_entities = []
        
        for i, (pred_report, true_report) in enumerate(zip(pred_reports, true_reports)):
            if i % 10 == 0:
                print(f"处理第 {i+1}/{len(pred_reports)} 个样本...")
            
            # 字段级别评估
            field_results = self.evaluate_field_level(pred_report, true_report)
            
            for field, result in field_results.items():
                field_bertscores[field].append(result['bertscore']['f1'])
                field_entities[field]['pred'].append(result['pred_entities'])
                field_entities[field]['true'].append(result['true_entities'])
            
            # 构建整体文本
            pred_full_text = " ".join([str(pred_report.get(f, '')) for f in ['主诉', '现病史', '诊断', '建议']])
            true_full_text = " ".join([str(true_report.get(f, '')) for f in ['主诉', '现病史', '诊断', '建议']])
            
            overall_pred_texts.append(pred_full_text)
            overall_true_texts.append(true_full_text)
            
            # 合并所有字段的实体
            pred_all_entities = self.extract_entities_by_category(pred_full_text)
            true_all_entities = self.extract_entities_by_category(true_full_text)
            
            overall_pred_entities.append(pred_all_entities)
            overall_true_entities.append(true_all_entities)
        
        # 计算整体BERTScore
        print("计算整体BERTScore...")
        overall_bertscore = self.calculate_bertscore(overall_pred_texts, overall_true_texts)
        
        # 计算整体Macro-Recall
        print("计算整体Macro-Recall...")
        overall_macro_recall = self.calculate_macro_recall(overall_pred_entities, overall_true_entities)
        
        # 计算字段级别的Macro-Recall
        print("计算字段级别Macro-Recall...")
        field_macro_recalls = {}
        for field in field_entities:
            print(f"  处理字段: {field}")
            field_macro_recalls[field] = self.calculate_macro_recall(
                field_entities[field]['pred'], 
                field_entities[field]['true']
            )
        
        # 汇总结果
        summary = {
            'overall': {
                'bertscore': overall_bertscore,
                'macro_recall': overall_macro_recall
            },
            'field_level': {}
        }
        
        for field in field_bertscores:
            summary['field_level'][field] = {
                'bertscore_f1_mean': np.mean(field_bertscores[field]),
                'bertscore_f1_std': np.std(field_bertscores[field]),
                'macro_recall': field_macro_recalls[field]['macro_recall'],
                'category_recalls': field_macro_recalls[field]['category_recalls']
            }
        
        return summary

def load_data(results_path):
    """加载数据"""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pred_reports = []
    true_reports = []
    sample_info = []
    
    for item in data:
        # 提取真实报告
        true_answer_str = item.get('answer', '{}')
        try:
            true_report = json.loads(true_answer_str)
        except:
            true_report = {}
        
        # 提取预测报告
        model_answer = item.get('model_answer', {})
        if isinstance(model_answer, str):
            try:
                model_answer = json.loads(model_answer)
            except:
                model_answer = {}
        elif model_answer is None:
            model_answer = {}
        
        pred_reports.append(model_answer)
        true_reports.append(true_report)
        sample_info.append({
            'id': item.get('other', {}).get('id', 'unknown'),
            'question': item.get('question', '')[:100] + '...'  # 截取前100字符
        })
    
    return pred_reports, true_reports, sample_info

def safe_convert_numpy_types(obj):
    """安全转换numpy类型为Python原生类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: safe_convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)  # 将集合转换为列表
    else:
        return obj

def run_evaluation(results_path, output_path):
    """运行完整评估"""
    print("加载数据...")
    pred_reports, true_reports, sample_info = load_data(results_path)
    
    print(f"加载完成: {len(pred_reports)} 个样本")
    
    # 初始化评估器
    evaluator = MedicalReportEvaluator(model_type="bert-base-chinese")
    
    # 执行评估
    print("执行BERTScore和Macro-Recall评估...")
    evaluation_results = evaluator.comprehensive_evaluation(pred_reports, true_reports)
    
    # 添加样本信息
    evaluation_results['sample_info'] = sample_info
    evaluation_results['total_samples'] = len(pred_reports)
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(safe_convert_numpy_types(evaluation_results), f, ensure_ascii=False, indent=2)
    
    # 打印评估报告
    print_evaluation_summary(evaluation_results)
    
    return evaluation_results

def print_evaluation_summary(results):
    """打印评估摘要"""
    print("\n" + "="*80)
    print("医疗报告生成模型评估报告 (BERTScore + Macro-Recall)")
    print("="*80)
    
    overall = results['overall']
    field_level = results['field_level']
    
    print(f"\n📊 总体评估指标:")
    print(f"  样本数量: {results['total_samples']}")
    print(f"  Overall BERTScore F1: {overall['bertscore']['f1']:.4f}")
    print(f"  Overall Macro-Recall: {overall['macro_recall']['macro_recall']:.4f}")
    
    print(f"\n📈 字段级别BERTScore F1:")
    for field, metrics in field_level.items():
        print(f"  {field}: {metrics['bertscore_f1_mean']:.4f} (±{metrics['bertscore_f1_std']:.4f})")
    
    print(f"\n🎯 字段级别Macro-Recall:")
    for field, metrics in field_level.items():
        print(f"  {field}: {metrics['macro_recall']:.4f}")
    
    print(f"\n🔍 实体类别召回率 (Overall):")
    category_recalls = overall['macro_recall']['category_recalls']
    for category, recall in category_recalls.items():
        category_name = {
            'symptoms': '症状',
            'diagnosis': '诊断', 
            'treatments': '治疗',
            'examinations': '检查'
        }.get(category, category)
        print(f"  {category_name}: {recall:.4f}")
    
    print(f"\n💡 关键指标总结:")
    print(f"  ✅ 整体语义相似度 (BERTScore F1): {overall['bertscore']['f1']:.1%}")
    print(f"  ✅ 医疗实体召回率 (Macro-Recall): {overall['macro_recall']['macro_recall']:.1%}")
    print(f"  ✅ 诊断实体召回率: {category_recalls['diagnosis']:.1%}")
    print(f"  ✅ 症状实体召回率: {category_recalls['symptoms']:.1%}")

# 使用示例
if __name__ == "__main__":
    # RESULTS_PATH = "./results/test_results.json"  # 测试结果文件
    # RESULTS_PATH = "./results/test_results_14b.json"  # 测试结果文件
    # RESULTS_PATH = "./results/test_results_7b_wo_few_shot.json"  # 测试结果文件
    RESULTS_PATH = "./results/test_results_7b_cot.json"  # 测试结果文件

    # EVALUATION_OUTPUT = "./evaluate_results/bertscore_macro_recall_evaluation.json"
    # EVALUATION_OUTPUT = "./evaluate_results/bertscore_macro_recall_evaluation_14b.json"
    # EVALUATION_OUTPUT = "./evaluate_results/bertscore_macro_recall_evaluation_7b_wo_few_shot.json"
    EVALUATION_OUTPUT = "./evaluate_results/bertscore_macro_recall_evaluation_7b_cot.json"

    try:
        # 执行评估
        evaluation_results = run_evaluation(RESULTS_PATH, EVALUATION_OUTPUT)
        print(f"\n评估完成！结果已保存到: {EVALUATION_OUTPUT}")
        
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()