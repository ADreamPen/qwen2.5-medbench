
### IMCS-V2_dev 
原始数据集，数据集链接：https://github.com/Lafitte1573/NLCorpora?tab=readme-ov-file

### data_process
data_process.py --- 基于few shot提示，处理完的数据与medbench提供的原始数据格式完全相同
data_process_wo_few_shot.py --- zero shot提示
data_process_cot.py --- 基于cot提示

### LLM reasoning
tset_IMCS.py 使用LLM进行推理

### 模型评估
evaluate.py 计算评价指标

### folder
test_data：存储数据预处理完成后的数据集
results: 存储LLM推理完成的结果
evaluate_results：存储模型评估结果

### medbench official
medbench官方网站的测评要求，实现代码和实现结果，链接https://medbench.opencompass.org.cn/dataset