#pipeline 管道


from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification,AutoTokenizer, pipeline

from PIL import Image, ImageDraw

#语言模型
# model = AutoModelForCausalLM.from_pretrained("/teacher_train/model/qwen2.5-3B-Instruct",local_files_only=True)

# tokenizer = AutoTokenizer.from_pretrained("/teacher_train/model/qwen2.5-3B-Instruct",local_files_only=True)

# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# print(pipe("你是谁"))


#分类模型
# model = AutoModelForSequenceClassification.from_pretrained("/teacher_train/model/rbt3",local_files_only=True)
# tokenizer = AutoTokenizer.from_pretrained("/teacher_train/model/rbt3",local_files_only=True)
# pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
# print(pipe("真术相成真好!"))


#图像模型

checkpoint=r"/teacher_train/model/owlvit-base-patch32/"
detecter = pipeline(model=checkpoint, task='zero-shot-object-detection')


path = "/teacher_data/wtb/教学文档/大模型教学文档/Snipaste_2025-09-10_11-03-34.png"
image = Image.open(path)

predictions = detecter(image, candidate_labels=['man'])

print(predictions)


draw = ImageDraw.Draw(image)	
	
for prediction in predictions:	
    box = prediction['box']	
    label = prediction['label']	
    score = prediction['score']	
    xmin, ymin, xmax, ymax = box.values()	
    draw.rectangle((xmin, ymin, xmax, ymax), outline='red', width=1)	
    draw.text((xmin, ymin), f"{label}:  {round(score, 2)}", fill='red')	

image.save("result.jpg")


