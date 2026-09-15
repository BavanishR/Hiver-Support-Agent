DECISION LOG:

    - LangGraph State Machine instead of  Sequential Chains: Chosen graph-based Framework (LangGraph) instead of sequential chains in order to allow for cyclic state transitions (clarify_count), conditional state transitions depending on the confidence of the model, and explicit terminal states. It also provides custom routings in which we can carry our state across different nodes which is very useful according to architecture i made because it contains multi-routing principles

    - Vector DB Subsampling (3,000 pairs out of 150k+): Subsampled 3,000 pairs of historical data for indexing into ChromaDB in order to comply with Hiver’s < 15 minutes of reproducibility requirement while covering 95%+ semantic intent coverage

    - Golden Set Leakage Protection through Programmatic Approach: Normalized set filtration enforced (~df['clean_customer'].isin(golden_queries)) in ingest.py before vector indexing to ensure zero test data leakage during RAG extraction process 

    - Local CPU Embeddings (all-MiniLM-L6-v2) instead Cloud APIs: Deployed HuggingFace's local 384-dimensional model to deliver retrieval latencies under 10ms while avoiding API key hassles, rate limiting

    - Direct Escalation without Reply for ACCOUNT_ACCESS: Routed login, password, and OTP queries directly to human support without public RAG reply generation to prevent customer data leaks on public social threads

    - Two-Tier "Reply-Then-Escalate" Routing: Designed high-friction intents (REFUND_AND_RETURN, WRONG_DAMAGED_ITEM) to execute initial RAG reply generation before immediately transitioning to escalate_node for human agent follow-up

    - Dual-Condition Clarification Guardrail: Combined a $0.70$ confidence cutoff with a strict retry limit (clarify_count >= 2) to handle ambiguous queries cleanly without looping indefinitely

    - State-Preserving Escalation Handoff: Added an explicit check (state.get("status") == "auto_handled") inside escalate_node to prevent secondary handoff notes from overwriting previously generated RAG responses

    - Handled other langauges to escalate to a human because we can't embedd other langauges in the embedding model i used so other lenguage inputs will escalate to a human

    - State-Preserving Escalation Handoff: Added an explicit check (state.get("status") == "auto_handled") inside escalate_node to prevent secondary handoff notes from overwriting previously generated RAG responses

    - Enforced Generation Cap (max_tokens=300): Constrained Groq LLM outputs to 300 tokens to prevent rate-limit throttling during evaluation runs