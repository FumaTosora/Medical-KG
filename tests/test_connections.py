import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.database import NEO4J_CONFIG, get_connection

results = []

results.append("=== SQLite ===")
try:
    conn = get_connection()
    results.append("SQLite: Connection OK")
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM entities_for_neo4j")
    results.append(f"  Entities: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM relations_for_neo4j")
    results.append(f"  Relations: {cur.fetchone()[0]}")
    cur.close()
    conn.close()
except Exception as e:
    results.append(f"SQLite FAILED: {type(e).__name__}: {e}")

results.append("=== Neo4j ===")
try:
    from neo4j import GraphDatabase
    results.append(f"  URI: {NEO4J_CONFIG['uri']}")
    results.append(f"  User: {NEO4J_CONFIG['user']}")
    results.append(f"  Database: {NEO4J_CONFIG['database']}")
    results.append(f"  Password set: {bool(NEO4J_CONFIG['password'])}")
    driver = GraphDatabase.driver(NEO4J_CONFIG["uri"], auth=(NEO4J_CONFIG["user"], NEO4J_CONFIG["password"]))
    driver.verify_connectivity()
    results.append("Neo4j: Connection OK")
    with driver.session(database=NEO4J_CONFIG["database"]) as s:
        n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        r = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        results.append(f"  Nodes: {n}")
        results.append(f"  Relationships: {r}")
    driver.close()
except Exception as e:
    results.append(f"Neo4j FAILED: {type(e).__name__}: {e}")

output = "\n".join(results)
print(output)
with open(os.path.join(os.path.dirname(__file__), "connection_test_result.txt"), "w") as f:
    f.write(output)
