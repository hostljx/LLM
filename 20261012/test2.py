import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import evaluate
#pip install scikit-learn


#查看可用的评估指标

list_evaluate = evaluate.list_evaluation_modules(include_community=True, with_details=False)

#使用评估指标

# accuracy = evaluate.load("accuracy")
# f1 = evaluate.load("f1")

# print(accuracy.description)
# print(accuracy.inputs_description)

#组合评估指标
clf_metrics = evaluate.combine(["accuracy", "f1"])

results = clf_metrics.compute(references=[1, 0, 0, 0], predictions=[1, 1, 0, 0])

print(results)
