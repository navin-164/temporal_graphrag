import asyncio
from temporal_graphrag.storage.document_store import DocumentStore
from temporal_graphrag.storage.vector_store import VectorStore
from temporal_graphrag.retrieval.vector_retriever import VectorRetriever
from temporal_graphrag.retrieval.graph_retriever import GraphRetriever
from temporal_graphrag.retrieval.hybrid_scorer import HybridScorer
from temporal_graphrag.knowledge_graph.neo4j_client import Neo4jClient

async def main():
    document_store = DocumentStore.from_directory("parsed")
    vector_store = VectorStore.load("vector_index")
    vector_retriever = VectorRetriever(vector_store)
    scorer = HybridScorer(document_store)

    query = "large language models"
    vector_hits = vector_retriever.search_papers(query, k=15)

    async with Neo4jClient() as client:
        graph_retriever = GraphRetriever(client)
        # seed graph search from the top vector hit
        graph_hits = await graph_retriever.k_hop_neighbors(vector_hits[0].id, hops=2, k=15)

    fused = scorer.fuse(vector_hits=vector_hits, graph_hits=graph_hits, k=10)
    for hit in fused:
        print(hit)

asyncio.run(main())