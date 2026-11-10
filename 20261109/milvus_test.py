from pymilvus import MilvusClient
import random

client = MilvusClient(uri="http://127.0.0.1:19530")
client.create_collection("demo", dimension=8, metric_type="COSINE")

rows = [{"id": i, "vector": [random.random() for _ in range(8)]} for i in range(100)]
client.insert("demo", rows)
client.flush("demo")

hits = client.search("demo", data=[[random.random() for _ in range(8)]], limit=3)
print(hits)
print("collections:", client.list_collections())