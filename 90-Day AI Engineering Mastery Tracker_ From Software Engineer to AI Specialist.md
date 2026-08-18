### 90-Day AI Engineering Mastery Tracker: From Software Engineer to AI Specialist

#### 1\. TRACKER USAGE INSTRUCTIONS

This document provides a high-fidelity roadmap to transition from a traditional software engineer to an AI Engineer capable of shipping production-grade systems. While the core curriculum from Codebasics is structured over 11 weeks (77 days), this tracker extends to 90 days to provide a 13-day "Hardening & Portfolio Buffer" for capstone project stability and career positioning.

##### The Study-Build-Review Cycle

As an engineer, passive consumption is your enemy. Follow this cycle:

* **Weekend Live Sessions (Theory & Architecture):**  Focus on the 4–7 PM IST live sessions to understand the "Why" and "How" of architectural patterns.  
* **Weekday Deep Work (Implementation):**  Dedicate 2–4 hours daily to the specific "Build" tasks listed below.  
* **Discord/Peer Review:**  Before moving to the next day, commit your code to GitHub and resolve at least one technical doubt in the community channel to reinforce learning through teaching.

##### Required Hardware Configuration

Your development environment must be robust enough to handle local inference and orchestration.

* **OS:**  Windows 11\.  
* **Processor:**  Intel Core i7 (10th Gen+) or AMD Ryzen 7 (4th Gen+). An i5 is acceptable only for API-centric workflows.  
* **RAM:**  8GB minimum;  **16GB strongly recommended.**  Note: 8GB will be a significant bottleneck during the Phase 6 fine-tuning and local LLM modules.  
* **Storage:**  512GB SSD (essential for fast model weight loading).  
* **GPU:**  NVIDIA GTX 1660 or higher is required for local GPU-accelerated tasks and quantization testing.

#### 2\. PHASE 1: AI FOUNDATIONS & LLM FUNDAMENTALS (DAYS 1–14)

**Day 1:**  Refactor a legacy Python script to use advanced functions and strict type hinting (Pydantic).  **Day 2:**  Implement file manipulation scripts that handle large-scale JSONL data processing for AI datasets.  **Day 3:**  Build a class-based Python wrapper for an LLM API that handles retries and error logging.  **Day 4:**  Write a Python decorator to measure token usage and latency for every LLM call.  **Day 5:**  Build a FastAPI endpoint that integrates an LLM and uses decorators for authentication.  **Day 6:**  Develop a Streamlit UI that takes user input and interacts with the Groq API for sub-second responses.  **Day 7:**  Deploy your first LLM-powered app on Streamlit Cloud and integrate basic logging.  **Day 8:**  Implement a Scaled Dot-Product Attention mechanism from scratch using NumPy.  **Day 9:**  Map the Transformer architecture by coding the Positional Encoding and Multi-Head Attention blocks.  **Day 10:**  Visualize attention weights using a pre-trained GPT-2 model to see how context is processed.  **Day 11:**  Generate and store text embeddings using sentence-transformers and compute cosine similarity.  **Day 12:**  Implement a semantic search script that finds the most relevant document from a local text batch.  **Day 13:**  Benchmark LLM parameter variance: Run 10 iterations of the same prompt while varying Temperature, Top-P, and Frequency Penalty.  **Day 14:**  Implement a "System Prompt" versioning system to track how different instructions change model outputs.**Foundational Checklist:**

*  Output: Deployed Streamlit \+ Groq App  
*  Output: From-scratch Attention Mechanism (NumPy)  
*  Output: Semantic Similarity Engine

#### 3\. PHASE 2: VECTOR DATABASES & RAG FOUNDATIONS (DAYS 15–28)

**Day 15:**  Spin up a local Qdrant instance via Docker and initialize a collection with custom vector dimensions.  **Day 16:**  Perform CRUD operations in Qdrant: Insert, update, and delete vectors with specific metadata payloads.  **Day 17:**  Implement a "Pure Python RAG" pipeline: Manually fetch context from a JSON file based on similarity and inject it into a prompt.  **Day 18:**  Implement Advanced Qdrant Operations: Use filtered searches and scroll APIs to manage large datasets.  **Day 19:**  Build a script to handle RAG "Context Overflow" by implementing a basic sliding window for retrieved text.  **Day 20:**  Refactor your Pure Python RAG to use Qdrant as the permanent storage layer.  **Day 21:**  Implement a performance benchmark: Measure search latency in Qdrant across 1,000 vs 10,000 vectors.  **Day 22:**  Initialize a LangChain RAG chain and replace your manual retrieval logic.  **Day 23:**  Implement LangChain "Chains" vs "Expressions" (LCEL) to create a modular retrieval pipeline.  **Day 24:**  Integrate Docling to parse complex PDFs with tables into structured markdown for the LLM.  **Day 25:**  Implement Recursive Character Text Splitting and compare it with token-based chunking.  **Day 26:**  Build a "Hierarchical Chunking" strategy: Create parent chunks for context and child chunks for retrieval.  **Day 27:**  Refactor the LangChain pipeline to support multiple document types (PDF, MD, TXT).  **Day 28:**  Finalize the "Smart Chunking" project: A pipeline that maintains table structure during retrieval.**Project Milestone:**

*   **LangChain RAG Pipeline:**  A production-ready system with Docling parsing and Hierarchical Chunking.

#### 4\. PHASE 3: ADVANCED RAG & AGENTIC AI (DAYS 29–42)

**Day 29:**  Implement Hybrid Search in Qdrant: Combine Dense Vectors with Sparse Vectors (BM25) for keyword accuracy.  **Day 30:**  Build a "Self-Querying Retriever" that uses an LLM to turn natural language into Qdrant metadata filters.  **Day 31:**  Implement SQL RAG: Use LangChain to translate natural language queries into executable SQL against a SQLite DB.  **Day 32:**  Build a Graph RAG prototype: Extract entities and relationships from text to build a Knowledge Graph.  **Day 33:**  Implement a Cross-Encoder Reranker to re-order the top 10 retrieved documents for maximum relevance.  **Day 34:**  Optimize the RAG pipeline using  **DSPy**  to programmatically improve your retrieval prompts.  **Day 35:**  Audit RAG performance: Identify "Lost in the Middle" phenomena in long-context retrieval.  **Day 36:**  Implement the ReAct (Reasoning and Acting) loop manually in Python to understand agent logic.  **Day 37:**  Define a Pydantic-based schema for "Tool Calling" and integrate a simple search tool.  **Day 38:**  Build your first autonomous Agent using LangChain that can use a calculator and search the web.  **Day 39:**  Implement a  **Semantic Router**  to classify user intent and route queries to different agents.  **Day 40:**  Integrate conversational memory: Use WindowBufferMemory and SummaryMemory to maintain context.  **Day 41:**  Build a professional UI for your agent using  **Chainlit** , enabling real-time token streaming.  **Day 42:**  Implement "Human-in-the-loop" (HITL) patterns in Chainlit to approve sensitive agent actions.

#### 5\. PHASE 4: AGENT ORCHESTRATION & MULTI-AGENT SYSTEMS (DAYS 43–56)

**Day 43:**  Map a complex workflow into a LangGraph state machine with Nodes (functions) and Edges (logic).  **Day 44:**  Implement "Statelessness" in LangGraph: Ensure each node execution only depends on the current state object.  **Day 45:**  Rebuild the ReAct loop inside LangGraph using conditional edges for tool execution.  **Day 46:**  Implement "Persistence" in LangGraph to allow agents to resume conversations across sessions.  **Day 47:**  Build a "Correction Loop" where an agent critiques and fixes its own output based on tool results.  **Day 48:**  Implement "Timeouts" and "Retry Logic" within your graph to handle API rate limits.  **Day 49:**  Create a visualization of your graph architecture using LangGraph's built-in drawing tools.  **Day 50:**  Design a Multi-Agent "Supervisor" architecture where one agent delegates tasks to specialists.  **Day 51:**  Implement Agent "Hand-offs": Code the logic for a "Triage Agent" to pass a state to a "Technical Agent."  **Day 52:**  Build a Multi-Agent workflow for "GitHub Automation": One agent reads issues, another writes code.  **Day 53:**  Integrate a "Reviewer Agent" into the GitHub flow to check code quality before "finalizing."  **Day 54:**  Implement parallel execution: Have multiple agents work on sub-tasks simultaneously in LangGraph.  **Day 55:**  Build a "Conflict Resolver" for agents that provides conflicting tool outputs.  **Day 56:**  Finalize the GitHub Agent: Automate the process from "Issue Description" to "Draft PR."

#### 6\. PHASE 5: EVALUATION, OBSERVABILITY & MULTIMODAL AI (DAYS 57–70)

**Day 57:**  Setup  **Ragas**  and compute Faithfulness and Answer Relevancy for your RAG pipeline.  **Day 58:**  Implement "LLM-as-a-Judge": Create an evaluation prompt that scores agent performance.  **Day 59:**  Integrate  **Weights & Biases (W\&B)**  to track LLM traces and visualize prompt iterations.  **Day 60:**  Run an "Adversarial Attack" test on your agent to check for prompt injection vulnerabilities.  **Day 61:**  Build an Eval Harness: Create a golden dataset of 50 Q\&A pairs for your capstone topic.  **Day 62:**  Interpret Eval metrics: Use W\&B to identify which chunking strategy led to the highest RAG score.  **Day 63:**  Implement  **Guardrails** : Use a secondary LLM call to verify the safety of the agent's output.  **Day 64:**  Extract structured JSON data from a complex image (invoice/chart) using a Vision model (GPT-4o/Claude).  **Day 65:**  Build a Multimodal RAG system: Embed both text descriptions and image features for retrieval.  **Day 66:**  Implement a pipeline to process multi-page PDFs where context is split between text and diagrams.  **Day 67:**  Build a "Visual Reasoning" agent that can answer questions about a uploaded video frame.  **Day 68:**  Optimize Multimodal performance: Implement image resizing and preprocessing to reduce latency.  **Day 69:**  Integrate the Multimodal RAG into a Chainlit UI for file uploads.  **Day 70:**  Benchmark Multimodal retrieval: Does adding image context actually improve the Ragas score?

#### 7\. PHASE 6: CLOUD DEPLOYMENT, MCP & CONTEXT ENGINEERING (DAYS 71–84)

**Day 71:**  Containerize your LangChain/LangGraph application using Docker.  **Day 72:**  Deploy your agent container to  **AWS AgentCore**  and configure environment secrets.  **Day 73:**  Implement AWS-specific logging and monitoring for your deployed agent.  **Day 74:**  Build your first  **Model Context Protocol (MCP)**  Server to expose local files to an AI client.  **Day 75:**  Implement an MCP Client that can dynamically query your MCP Server for context.  **Day 76:**  Standardize data exchange: Use MCP to bridge the gap between your local database and the LLM.  **Day 77:**  Build a "Compound AI System": Integrate a LangGraph multi-agent flow with an MCP server.  **Day 78:**  Master Context Engineering: Implement a "Context Pruning" script to keep prompts under token limits.  **Day 79:**  Integrate  **deepagents**  for specialized context handling in complex multi-agent hand-offs.  **Day 80:**  Setup  **LangSmith**  to debug context propagation and find exactly where an agent lost its "memory."  **Day 81:**  Prepare a synthetic dataset for fine-tuning using an LLM to generate high-quality instruction-response pairs.  **Day 82:**  Use  **Unsloth**  to perform a LoRA/QLoRA fine-tuning run on a Small Language Model (SLM).  **Day 83:**  Export your fine-tuned model weights and convert them to GGUF format.  **Day 84:**  Setup  **Ollama**  and run your fine-tuned SLM locally, testing it against your RAG pipeline.

#### 8\. PHASE 7: SYSTEM DESIGN, VOICE AI & CAREER PREP (DAYS 85–90)

**Day 85:**  Build a Voice AI stack: Integrate Speech-to-Text (Whisper) and Text-to-Speech (ElevenLabs).  **Day 86:**  Implement "Interruption Handling" and "Silence Detection" for a low-latency (\<800ms) voice agent.  **Day 87:**  Inference Engineering: Serve  **Qwen 2.5-3B**  as a REST API using  **vLLM**  with quantization (AWQ/FP8).  **Day 88:**   **AI Security & Optimization:**  Implement  **Semantic Caching**  to reduce costs and apply the  **OWASP Top 10 for AI**  (checking for Data Poisoning and SSRF).  **Day 89:**   **AI System Design:**  Refactor your project with Rate Limiting, Circuit Breakers, and Model Routing logic.  **Day 90:**   **Career Branding:**  Finalize your GitHub READMEs for storytelling and update LinkedIn with your AI Engineering portfolio.**AI Security & System Checklist:**

*  Implement Semantic Caching (Redis/GPTCache)  
*  Audit against OWASP Top 10 for LLMs  
*  Apply Rate Limiting middleware to the API  
*  Configure Inference Benchmarking (Token/sec)**Personal Branding Checklist:**  
*  GitHub: 3+ Repos with Architectural Diagrams  
*  LinkedIn: Post 1 technical breakdown of your Capstone  
*  Resume: Highlight "AI Engineering Stack" (LangGraph, MCP, vLLM)

#### 9\. CAPSTONE PROJECT TRACKER

Milestone Phase,Target Tasks,Completion Status  
Scoping,"Define problem, AI stack, and ""Production-First"" constraints.",  
Architecture,Design LangGraph state machine and MCP server integration.,  
Implementation,"Build multi-agent backend, vLLM inference, and Chainlit UI.",  
Evaluation,Run Ragas/W\&B harness and apply OWASP security fixes.,

#### 10\. DAILY LOG & NOTES TEMPLATE

\#\#\# Daily Entry: Day \[X\]  
\- \*\*Topic:\*\* \[e.g., LangGraph Persistence\]  
\- \*\*Hours Spent:\*\*   
\- \*\*Engineering Constraint:\*\* \[e.g., Handling race conditions in shared state\]  
\- \*\*Key Implementation Challenge:\*\*   
\- \*\*Solution/Workaround:\*\*   
\- \*\*Link to GitHub Commit:\*\* 

#### 11\. RESOURCE MAPPING TABLE

Tool / Framework,Relevance,Target Implementation Days  
Groq / Streamlit,Fast UI and initial LLM integration,Days 6–7  
Qdrant,"Vector DB, Dense/Sparse Hybrid Search","Days 15–21, 29"  
LangChain,Core RAG and Agent framework,Days 22–42  
LangGraph,Multi-agent orchestration & statefulness,Days 43–56  
DSPy,Programmatic Prompt Optimization,Day 34  
Ragas / W\&B,"Evaluation, Observability & Tracing",Days 57–63  
AWS AgentCore,Cloud deployment for agentic systems,Days 71–73  
MCP,Data protocol for context standardization,Days 74–77  
deepagents,Advanced multi-agent context,Day 79  
LangSmith,Context Engineering & Debugging,Day 80  
Unsloth / LoRA,Local fine-tuning and synthetic data,Days 81–83  
vLLM / Qwen 2.5,Production inference & SLM serving,Day 87  
