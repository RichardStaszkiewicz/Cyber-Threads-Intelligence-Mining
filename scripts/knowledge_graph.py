import json
import networkx as nx
from py2neo import Graph, Node, Relationship

class ThreatKnowledgeGraph:
    """
    Constructs a knowledge graph from validated IOCs.

    Attributes:
        ioc_file (str): Path to validated IOCs JSON file.
        graph (networkx.Graph): Graph object for local visualization.
    """

    def __init__(self, ioc_file="logs/validated_iocs.json"):
        """
        Initializes the ThreatKnowledgeGraph.

        Args:
            ioc_file (str, optional): Path to validated IOCs JSON file. Defaults to "logs/validated_iocs.json".
        """
        self.ioc_file = ioc_file
        self.graph = nx.Graph()
        self.neo4j_conn = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

    def load_iocs(self):
        """Loads validated IOCs."""
        with open(self.ioc_file, "r", encoding="utf-8") as file:
            return json.load(file)

    def build_graph(self):
        """Constructs the knowledge graph."""
        ioc_data = self.load_iocs()

        for entry in ioc_data:
            ioc_node = Node("IOC", value=entry["ioc"])
            self.neo4j_conn.create(ioc_node)

            if entry["virustotal"]:
                vt_node = Node("ThreatFeed", name="VirusTotal")
                rel = Relationship(ioc_node, "REPORTED_IN", vt_node)
                self.neo4j_conn.create(rel)

            if entry["alienvault_otx"]:
                otx_node = Node("ThreatFeed", name="AlienVaultOTX")
                rel = Relationship(ioc_node, "REPORTED_IN", otx_node)
                self.neo4j_conn.create(rel)

        print("Knowledge graph constructed!")

# Example usage:
if __name__ == "__main__":
    graph_builder = ThreatKnowledgeGraph()
    graph_builder.build_graph()
