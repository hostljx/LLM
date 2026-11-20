"""
社区检测（GDS Leiden 优先，NetworkX Louvain 自动降级）

社区检测：把"内部连接紧密、对外连接稀疏"的实体自动分组。
在三国图谱上，算法会自动聚出蜀汉、曹魏、东吴等阵营——无需人工标注。

我们提供两套实现，运行时自动选择：
  方案 A（首选）：Neo4j GDS 插件的 gds.leiden
      - 微软 GraphRAG 官方用的正是 Leiden 算法（Louvain 改进版，
        速度更快且保证社区内部连通）
      - 在数据库内计算，适合大图
  方案 B（降级）：未安装 GDS 时，把图读到 Python 内存，
      用 NetworkX 内置的 Louvain 算法计算，再把结果写回 Neo4j
      - 零插件依赖，课堂环境 100% 可跑
      - Louvain 与 Leiden 同属模块度优化算法，小图上结果几乎一致

两套方案殊途同归：给每个 Entity 节点写上 communityId 属性。
"""
from config import run_cypher

GRAPH_NAME = "sanguo_graph"  # GDS 内存投影图的名字


def gds_available() -> bool:
    """探测 GDS 插件是否安装（查询已注册的 gds.* 过程）。"""
    rows = run_cypher(
        "SHOW PROCEDURES YIELD name WHERE name = 'gds.leiden.write' RETURN name"
    )
    return len(rows) > 0


# 方案 A：Neo4j GDS Leiden（与微软 GraphRAG 同款算法）
def detect_with_gds():
    """GDS 标准三步走：project 投影 → 执行算法 → drop 释放。"""
    # 步骤 1：创建内存投影（GDS 算法只在内存图上跑）
    exists = run_cypher(
        "CALL gds.graph.exists($name) YIELD exists RETURN exists", name=GRAPH_NAME
    )[0]["exists"]
    if exists:   # 上次运行残留的同名投影，先删掉
        run_cypher("CALL gds.graph.drop($name)", name=GRAPH_NAME)

    # 投影说明：
    # - 节点：所有 Entity
    # - 关系：RELATES，设为 UNDIRECTED（Leiden 要求无向图）
    # - 平行边聚合：同一对实体间的多条关系把 weight 求和（SUM），
    #   关系越多的两个实体在算法眼中越"紧密"
    run_cypher("""
        CALL gds.graph.project(
            $name,
            'Entity',
            {
                RELATES: {
                    orientation: 'UNDIRECTED',
                    properties: {weight: {property: 'weight', aggregation: 'SUM'}}
                }
            }
        )
    """, name=GRAPH_NAME)

    # 步骤 2：跑 Leiden，社区编号写回节点的 communityId 属性
    result = run_cypher("""
        CALL gds.leiden.write($name, {
            writeProperty: 'communityId',            // 结果写回的属性名
            relationshipWeightProperty: 'weight',    // 用关系权重参与模块度计算
            randomSeed: 42,                          // 固定随机种子
            concurrency: 1                           // 单线程 + 种子 => 结果可复现
        })
        YIELD communityCount, modularity
        RETURN communityCount, modularity
    """, name=GRAPH_NAME)[0]
    print(f"[GDS Leiden] 共 {result['communityCount']} 个社区，"
          f"模块度 modularity = {result['modularity']:.4f}")
    # 模块度是社区划分质量指标，0.3 以上说明社区结构明显

    # 步骤 3：释放内存投影（好习惯，避免占用数据库内存）
    run_cypher("CALL gds.graph.drop($name)", name=GRAPH_NAME)


# 方案 B：NetworkX Louvain（无 GDS 插件时的降级实现）
def detect_with_networkx():
    """把图读到内存 → NetworkX Louvain → 结果写回 Neo4j。"""
    import networkx as nx   # 延迟导入：只有走降级路线才需要装 networkx

    # 1) 从 Neo4j 读出所有实体和关系（权重求和，与 GDS 投影的 SUM 聚合等价）
    nodes = [r["name"] for r in run_cypher("MATCH (e:Entity) RETURN e.name AS name")]
    edges = run_cypher("""
        MATCH (a:Entity)-[r:RELATES]->(b:Entity)
        RETURN a.name AS source, b.name AS target, sum(r.weight) AS weight
    """)

    # 2) 构建无向加权图
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for e in edges:
        if G.has_edge(e["source"], e["target"]):
            # 双向关系（A->B 和 B->A）在无向图中合并为一条边，权重累加
            G[e["source"]][e["target"]]["weight"] += e["weight"]
        else:
            G.add_edge(e["source"], e["target"], weight=e["weight"])

    # 3) Louvain 社区检测（NetworkX >= 3.0 内置，返回节点集合的列表）
    communities = nx.community.louvain_communities(G, weight="weight", seed=42)
    modularity = nx.community.modularity(G, communities, weight="weight")
    print(f"[NetworkX Louvain] 共 {len(communities)} 个社区，"
          f"模块度 modularity = {modularity:.4f}")

    # 4) 把社区编号批量写回 Neo4j 节点属性（与 GDS 方案产出完全一致的结构）
    for cid, members in enumerate(communities):
        run_cypher("""
            MATCH (e:Entity) WHERE e.name IN $names
            SET e.communityId = $cid
        """, names=list(members), cid=cid)

# 公共部分
def print_communities():
    """打印每个社区的成员，直观检查划分效果。"""
    rows = run_cypher("""
        MATCH (e:Entity)
        WHERE e.communityId IS NOT NULL
        WITH e.communityId AS cid, collect(e.name) AS members
        WHERE size(members) >= 2              // 只看有意义的社区（孤立点忽略）
        RETURN cid, members
        ORDER BY size(members) DESC
    """)
    print(f"\n成员数 >= 2 的社区共 {len(rows)} 个：")
    for row in rows:
        names = "、".join(row["members"][:12])
        more = f" ...等{len(row['members'])}个" if len(row["members"]) > 12 else ""
        print(f"  社区 {row['cid']}: {names}{more}")


def main():
    if gds_available():
        print("检测到 GDS 插件，使用方案 A：gds.leiden")
        detect_with_gds()
    else:
        print("未检测到 GDS 插件，使用方案 B：NetworkX Louvain（降级方案）")
        detect_with_networkx()
    print_communities()


if __name__ == "__main__":
    main()