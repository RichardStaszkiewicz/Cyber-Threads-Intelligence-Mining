import networkx as nx
import matplotlib.pyplot as plt
import json
from collections import defaultdict
from py2neo import Graph

class ThreatGraphAnalyzer:
    """
    Analyzes a threat intelligence knowledge graph.
    
    Attributes:
        ioc_file (str): Path to validated IOCs JSON file.
        graph (networkx.Graph): Graph object for analysis.
        neo4j_conn (py2neo.Graph): Neo4j connection.
    """

    def __init__(self, ioc_file="logs/validated_iocs.json"):
        """
        Initializes the ThreatGraphAnalyzer.

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
        """Constructs the network graph from IOCs."""
        ioc_data = self.load_iocs()

        for entry in ioc_data:
            self.graph.add_node(entry["ioc"], type="IOC")

            if entry["virustotal"]:
                self.graph.add_edge(entry["ioc"], "VirusTotal", relationship="REPORTED_IN")

            if entry["alienvault_otx"]:
                self.graph.add_edge(entry["ioc"], "AlienVaultOTX", relationship="REPORTED_IN")

    def plot_graph(self):
        """Visualizes the threat intelligence graph."""
        plt.figure(figsize=(12, 8))
        nx.draw(self.graph, with_labels=True, node_size=500, node_color="lightblue", font_size=8)
        plt.title("Threat Intelligence Graph")
        plt.show()

    def detect_communities(self):
        """Detects clusters of related threats using the Louvain method."""
        from networkx.algorithms.community import greedy_modularity_communities

        communities = list(greedy_modularity_communities(self.graph))
        community_mapping = {node: i for i, community in enumerate(communities) for node in community}
        
        # Color nodes by community
        colors = [community_mapping.get(node, 0) for node in self.graph.nodes()]
        plt.figure(figsize=(12, 8))
        nx.draw(self.graph, with_labels=True, node_size=500, node_color=colors, cmap=plt.cm.jet, font_size=8)
        plt.title("Threat Communities")
        plt.show()

    def compute_centrality(self):
        """Computes key players using betweenness centrality."""
        centrality = nx.betweenness_centrality(self.graph)
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]

        print("Top 10 most influential threat actors:")
        for node, score in top_nodes:
            print(f"{node}: {score}")

    def run_analysis(self):
        """Executes all graph analysis functions."""
        self.build_graph()
        self.plot_graph()
        self.detect_communities()
        self.compute_centrality()

# Example usage:
if __name__ == "__main__":
    analyzer = ThreatGraphAnalyzer()
    analyzer.run_analysis()
