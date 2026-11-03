"""
天气查询工具（LangChain Tool）。

只负责“给城市名 -> 返回当日天气文本”，不包含大模型逻辑。
由 chat.py 导入使用；也可以单独运行测试：

    python weather_tool.py
"""

import os

import requests
from langchain_core.tools import tool

# ----------------- 配置 -----------------
# 和风天气 API Key
QWEATHER_KEY = os.getenv("QWEATHER_KEY", "c82a7a7babda4f2a8e7b05079edd576b")
# 和风天气 API Host：登录控制台 → 「设置」→「API Host」查看，形如 https://xxxxx.re.qweatherapi.com
# 注意：新版账号不能用 devapi.qweather.com / geoapi.qweather.com，会返回 403 Invalid Host。
QWEATHER_API_HOST = os.getenv("QWEATHER_HOST", "nd5egpqr9e.re.qweatherapi.com")
# 允许不带 https:// 的写法
if not QWEATHER_API_HOST.startswith("http"):
    QWEATHER_API_HOST = "https://" + QWEATHER_API_HOST
QWEATHER_API_HOST = QWEATHER_API_HOST.rstrip("/")

TIMEOUT = 10

# 和风天气常见返回码说明
QWEATHER_ERRORS = {
    "204": "该地区暂无数据",
    "400": "请求参数错误",
    "401": "API Key 无效（请在和风天气控制台核对 Key）",
    "402": "超过每日访问次数或余额不足",
    "403": "没有访问该接口的权限",
    "404": "查询的地区不存在",
    "500": "和风天气服务端错误",
}


def _request(url: str, params: dict) -> dict:
    """发 GET 请求，失败时返回带 error 字段的字典，避免工具直接抛异常。"""
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        # 和风天气出错时 HTTP 状态码也可能不是 200，但 JSON 里仍有 code，尽量交给调用方判断
        data = resp.json()
    except ValueError:
        return {"error": f"接口返回的不是 JSON（HTTP {resp.status_code}）：{resp.text[:200]}"}
    except requests.RequestException as e:
        return {"error": f"请求失败：{e}（请检查 QWEATHER_API_HOST 是否填对）"}

    # 新版 API 出错时返回 {"error": {"title": ..., "detail": ...}}
    if isinstance(data.get("error"), dict):
        err = data["error"]
        title = err.get("title", "")
        detail = err.get("detail", "")
        hint = ""
        if "security restriction" in detail.lower():
            hint = "（该 Key 的「安全设置」限制了来源 IP/域名，请把本机公网 IP 加入白名单或取消限制）"
        elif "invalid host" in detail.lower():
            hint = "（请到控制台「设置 → API Host」核对 QWEATHER_API_HOST）"
        return {"error": f"和风天气请求被拒绝：{title} - {detail}{hint}"}
    if not isinstance(data, dict):
        return {"error": f"接口返回格式异常：{data}"}
    return data


@tool
def get_today_weather(city: str) -> str:
    """查询指定城市今天的天气情况。city 传中文城市名，例如“北京”“上海”“广州”。"""
    # 1. 城市名 -> 和风天气 Location ID
    geo = _request(
        f"{QWEATHER_API_HOST}/geo/v2/city/lookup",
        {"location": city, "key": QWEATHER_KEY, "number": 1},
    )
    if "error" in geo:
        return geo["error"]
    if geo.get("code") != "200":
        return QWEATHER_ERRORS.get(geo.get("code"), f"查询城市失败，返回码：{geo.get('code')}")
    if not geo.get("location"):
        return f"没找到城市“{city}”，换个写法试试（如“北京”“杭州市”）。"

    place = geo["location"][0]
    location_id = place["id"]
    adm1, adm2, name = place.get("adm1", ""), place.get("adm2", ""), place.get("name", "")
    # 避免出现“四川省成都成都”这种重复
    place_name = f"{adm1}{'' if adm2 == name else adm2}{name}"

    # 2. 查询未来 3 天预报，第 1 条就是今天
    weather = _request(
        f"{QWEATHER_API_HOST}/v7/weather/3d",
        {"location": location_id, "key": QWEATHER_KEY},
    )
    if "error" in weather:
        return weather["error"]
    if weather.get("code") != "200" or not weather.get("daily"):
        return QWEATHER_ERRORS.get(weather.get("code"), f"查询天气失败，返回码：{weather.get('code')}")

    today = weather["daily"][0]
    return (
        f"{place_name} {today['fxDate']} 天气：\n"
        f"白天 {today['textDay']}，夜间 {today['textNight']}；\n"
        f"最高气温 {today['tempMax']}°C，最低气温 {today['tempMin']}°C；\n"
        f"{today['windDirDay']} {today['windScaleDay']} 级；\n"
        f"相对湿度 {today['humidity']}%，降水量 {today['precip']} mm；\n"
        f"紫外线指数 {today['uvIndex']}，日出 {today['sunrise']}，日落 {today['sunset']}。"
    )


if __name__ == "__main__":
    print(get_today_weather.invoke({"city": "成都"}))
