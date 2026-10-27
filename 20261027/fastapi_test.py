from fastapi import FastAPI,Request
from httpx import get
from pydantic import BaseModel
import uvicorn



'''
POST /v1.0/todo/users/PUoixxxxGiP6g/tasks?operatorId=PUoiinWIpxxx   #？前的这是路由  ？后的就是qury部分  
Host:api.dingtalk.com  #这是主机地址
x-acs-dingtalk-access-token:xxxxx  #这是请求头
Content-Type:application/json      #请求头

#这是数据主体
{
  "sourceId" : "isv_dingtalkTodo1",
  "subject" : "接入钉钉待办",
  "creatorId" : "PUoiiP6g",
  "description" : "应用可以调用该接口发起一个钉钉待办任务，该待办事项会出现在钉钉客户端“待办”页面，需要注意的是，通过开放接口发起的待办，目前仅支持直接跳转ISV应用详情页（ISV在调该接口时需传入自身应用详情页链接）。",
  "dueTime" : 1617675000000,
  "executorIds" : [ "PUoiixxxxhiiGiP6g" ],
  "participantIds" : [ "PUoxxxxiGiP6g" ],
  "detailUrl" : {
    "appUrl" : "https://www.dingtalk.com",
    "pcUrl" : "https://www.dingtalk.com"
  }}

'''

app = FastAPI() #实例化FastAPI对象

@app.post("/add_one")
async def add_one(request: Request):
    # 手动解析请求体（JSON）
    data = await request.json()
    number = data.get('number')
    
    # 手动获取请求头
    token = request.headers.get('x-acs-dingtalk-access-token')
    content_type = request.headers.get('content-type')
    
    # 手动获取 Query 参数
    operator_id = request.query_params.get('operatorId')
    
    # 手动获取 URL 路径
    path = request.url.path
    
    # 手动获取方法
    method = request.method
    
    print(f"请求体: {data}")
    print(f"请求头 token: {token}")
    print(f"Query 参数: {operator_id}")
    print(f"路径: {path}, 方法: {method}")
    
    # 自己做校验（可选）
    if not number:
        return {"error": "缺少 number"}, 400
    if not token:
        return {"error": "缺少 token"}, 401
    
    return {"result": number + 1}

@app.get("/hello")
async def hello(request: Request):
    # 解析 query 参数
    name = request.query_params.get("name", "匿名")
    age = request.query_params.get("age", "0")

    # 解析请求头
    token = request.headers.get("x-token")

    # 解析路径、方法
    path = request.url.path
    method = request.method

    return {
        "msg": f"你好，{name}",
        "age": age,
        "token": token,
        "path": path,
        "method": method,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)