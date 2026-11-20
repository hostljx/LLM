"""
知识图谱可视化

 Neo4j 读出实体图，用 pyvis 生成可交互的 HTML 页面：
- 节点颜色 = 所属社区（一眼看出蜀魏吴阵营）
- 节点大小 = 度数（关系越多的实体越大，核心人物一目了然）
- 鼠标悬停节点/边可查看描述详情，可拖拽、缩放

输出：output/graph.html（双击用浏览器打开）

另外：Neo4j Browser (http://localhost:7474) 中可直接执行
  MATCH (a:Entity)-[r:RELATES]->(b:Entity) RETURN a, r, b

"""
from pyvis.network import Network

from config import OUTPUT_DIR, run_cypher

# 社区配色板（社区数超过颜色数时循环使用）
PALETTE = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
           "#1abc9c", "#e67e22", "#34495e", "#fd79a8", "#00cec9"]


def fetch_graph():
    """从 Neo4j 取出所有实体节点和关系边。"""
    nodes = run_cypher("""
        MATCH (e:Entity)
        RETURN e.name AS name, e.type AS type, e.description AS description,
               e.communityId AS cid,
               COUNT { (e)-[:RELATES]-() } AS degree
    """)
    edges = run_cypher("""
        MATCH (a:Entity)-[r:RELATES]->(b:Entity)
        RETURN a.name AS source, b.name AS target,
               r.relation AS relation, r.description AS description, r.weight AS weight
    """)
    return nodes, edges


def build_html(nodes: list[dict], edges: list[dict]) -> str:
    """用 pyvis 构建交互式网络图并输出 HTML 文件。"""
    net = Network(
        height="900px", width="100%",
        bgcolor="#1a1a2e", font_color="white",   # 深色背景更好看
        cdn_resources="in_line",                 # 依赖内嵌进 HTML，离线也能打开
    )

    # 把社区 id 映射到颜色
    community_ids = sorted({n["cid"] for n in nodes if n["cid"] is not None})
    color_map = {cid: PALETTE[i % len(PALETTE)] for i, cid in enumerate(community_ids)}

    for n in nodes:
        net.add_node(
            n["name"],
            label=n["name"],
            color=color_map.get(n["cid"], "#95a5a6"),        # 无社区的节点用灰色
            size=12 + n["degree"] * 2,                        # 度数越大节点越大
            # title 是鼠标悬停时显示的提示内容
            title=f"{n['name']}（{n['type']}）| 社区{n['cid']} | 度数{n['degree']}\n{n['description']}",
        )

    for e in edges:
        net.add_edge(
            e["source"], e["target"],
            title=f"{e['relation']}：{e['description']}",     # 悬停显示关系详情
            width=1 + float(e["weight"] or 1) / 10,            # 权重越大边越粗
            arrows="to",                                       # 显示方向箭头
        )

    # 物理引擎参数：让布局更舒展、社区自然聚簇
    net.barnes_hut(gravity=-3000, central_gravity=0.2, spring_length=150)

    out_file = OUTPUT_DIR / "graph.html"
    # 注意：不用 net.write_html()——它在 Windows 上以系统默认编码(GBK)写文件会报错，
    # 改为先生成 HTML 字符串，再显式用 UTF-8 写入
    html = net.generate_html(notebook=False)
    out_file.write_text(html, encoding="utf-8")
    return str(out_file)


def main():
    nodes, edges = fetch_graph()
    print(f"读取到 {len(nodes)} 个节点、{len(edges)} 条边")
    out = build_html(nodes, edges)
    print(f"可视化已生成：{out}（用浏览器打开即可交互浏览）")


if __name__ == "__main__":
    main()