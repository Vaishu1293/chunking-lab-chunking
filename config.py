PROJECT_NAME = "Enterprise Search"
VERSION = "1.0"
AUTHOR = "Vaishali"
EMBEDDING_MODEL = "gemini-embedding-001"
MODEL= "gemini-3.5-flash"
COLLECTION_NAME = "FAQ_RECORDS"
DATABASE_PATH = "./databases/faq-records"
BANNER_WIDTH = 40
SAMPLE_DOC = """
Python is widely used for AI development.
NumPy provides numerical array operations.
PyTorch is commonly used to train neural networks.

Full-time employees receive 20 days of paid annual leave per year.
Leave requests must be submitted through the corporate HR portal.
Managers approve requests exceeding standard limits.

Vector databases store high-dimensional embeddings efficiently.
ChromaDB performs efficient vector similarity search for RAG.
"""
DOCUMENT = """
The Enterprise AI Platform is an initiative designed to streamline internal data analysis and automate customer support workflows. By leveraging modern large language models, the platform aims to reduce manual request processing time by 40% across all operational departments. Initial rollout focuses on high-frequency, low-complexity queries from internal staff.

The system architecture follows an event-driven microservices model hosted on cloud infrastructure. API gateways route incoming user prompts to dedicated worker pools, ensuring high availability and fault tolerance during peak load periods. All service communication is encrypted in transit using standard TLS protocols.

Key architecture components include an ingestion engine, a vector retrieval pipeline, and an LLM orchestration layer. The ingestion engine parses incoming multi-format documents, while the vector database maintains real-time index embeddings for rapid semantic lookup. The orchestration layer coordinates context assembly before calling the primary model end point.

Deployment occurs via containerized pipelines managed by Kubernetes clusters across primary and failover availability zones. Automated CI/CD workflows run unit and integration test suites before any container image is promoted to production. Blue-green deployment strategies guarantee zero-downtime updates during minor patch releases.

Post-deployment monitoring relies on real-time telemetry tracking token consumption, model latency, and API error rates. Distributed tracing identifies potential bottlenecks in the vector retrieval step, triggering automatic scale-out events when query volume spikes. Custom alert thresholds notify on-call engineering teams of anomalies.

Full-time employees receive 20 days of paid annual leave per calendar year, accrued on a monthly basis. All leave requests must be submitted through the corporate HR portal at least two weeks prior to the intended start date. Managers are responsible for reviewing and approving or denying requests within three business days.
"""
SYSTEM_INSTRUCTION = """
You are an expert text segmentation assistant. Your task is to analyze a document and group its paragraphs into semantically coherent chunks based on topic boundaries.

Instructions:
1. Read the entire document carefully.
2. Do NOT automatically create one chunk per paragraph. If two or more adjacent paragraphs discuss the same broader topic, combine them into one chunk.
3. Create a new chunk ONLY when there is a meaningful topic transition.
4. Group adjacent paragraphs that discuss the same core subject into the same chunk.
5. PRESERVE THE ORIGINAL TEXT EXACTLY. Do not summarize, rewrite, rephrase, or drop any words from the original paragraphs.
6. Maintain the exact original order of the text.

Output Format:
Return a JSON array of strings, where each string is a single grouped chunk containing the full, verbatim text of its constituent paragraphs.
"""

