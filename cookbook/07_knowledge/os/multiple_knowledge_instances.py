"""
Content Sources for Knowledge — DX Design
============================================================

This cookbook demonstrates the API for adding content from various
remote sources (S3, GCS, SharePoint, GitHub, etc.) to Knowledge.

Key Concepts:
- RemoteContentConfig: Base class for configuring remote content sources
- Each source type has its own config: S3Config, GcsConfig, SharePointConfig, GitHubConfig
- Configs are registered on Knowledge via `content_sources` parameter
- Configs have factory methods (.file(), .folder()) to create content references
- Content references are passed to knowledge.insert()
"""

from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.knowledge.knowledge import Knowledge
from agno.models.openai import OpenAIChat
from agno.os import AgentOS
from agno.vectordb.pgvector import PgVector

# Database connections
contents_db = PostgresDb(
    db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    knowledge_table="knowledge_contents",
)
vector_db = PgVector(
    table_name="knowledge_vectors",
    db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
)

# Create Knowledge instances
company_knowledge = Knowledge(
    name="Company Knowledge Base",
    description="Unified knowledge from multiple sources",
    contents_db=contents_db,
    vector_db=vector_db,
    # content_sources=[sharepoint, github_docs, azure_blob],
)

personal_knowledge = Knowledge(
    name="Personal Knowledge Base",
    description="Unified knowledge from multiple sources",
    contents_db=contents_db,
    vector_db=vector_db,
    # content_sources=[sharepoint, github_docs, azure_blob],
)

company_knowledge_db = PostgresDb(
    db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    knowledge_table="knowledge_contents2",
)

company_knowledge_additional = Knowledge(
    name="Company Knowledge Base",
    description="Unified knowledge from multiple sources",
    contents_db=company_knowledge_db,
    vector_db=vector_db,
    # content_sources=[sharepoint, github_docs, azure_blob],
)

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    knowledge=company_knowledge,
    search_knowledge=True,
)

agent_os = AgentOS(
    knowledge=[company_knowledge, company_knowledge_additional, personal_knowledge],
    agents=[agent],
)
app = agent_os.get_app()

# ============================================================================
# Run AgentOS
# ============================================================================
if __name__ == "__main__":
    # Serves a FastAPI app exposed by AgentOS. Use reload=True for local dev.
    agent_os.serve(app="multiple_knowledge_instances:app", reload=True)
