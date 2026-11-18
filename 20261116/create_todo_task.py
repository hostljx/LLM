from get_acctoken_id import get_acctoken,get_unionid_id
import requests


acctoken = get_acctoken()
userid = "01374841151421446344"

unionid = get_unionid_id(acctoken,userid)


def todo_task(access_token,unionid):

    url = f"https://api.dingtalk.com/v1.0/todo/users/{unionid}/tasks"

    headers = {
        "Content-Type":"application/json",
        "x-acs-dingtalk-access-token":access_token
    }
    data = {
        "subject":"测试任务2",
        "todoType":"TODO",
        "detailUrl" : {
            "appUrl" : "https://www.dingtalk.com",
            "pcUrl" : "https://www.dingtalk.com"
        },
        "priority":40

    }

    response = requests.post(url,headers=headers,json=data)
    return response


res = todo_task(acctoken,unionid)
print(res)

