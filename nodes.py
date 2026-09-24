import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from schemas import IntentResult
from state import AgentState

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    max_tokens=300,
    api_key=os.getenv("GROQ_API_KEY")
)
structured_llm = llm.with_structured_output(IntentResult,method="function_calling")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

SYSTEM_PROMPT = """You are an intent classification agent for Amazon customer support on Twitter.
Analyze the customer's message and categorize it into EXACTLY one of the allowed intents.
Also rate your confidence from 0.0 to 1.0.

If the customer's message is NOT written in English, set intent to "UNCLEAR_VAGUE" and confidence to 0.3, regardless of how clear the message seems in its own language.

Allowed Intents:
- DELIVERY_DELAY: Delayed packages, late shipments, missing deliveries.
- REFUND_AND_RETURN: Refund status, returns, double charges.
- DIGITAL_CONTENT_APP: Kindle, Prime Video, digital orders, app crashes.
- ORDER_STATUS: Tracking order updates, dispatch confirmation.
- WRONG_DAMAGED_ITEM: Received broken, defective, or incorrect items.
- ACCOUNT_ACCESS: Login issues, OTPs, password resets.
- UNCLEAR_VAGUE: Incomplete info, vague complaints, plain venting without details, or non-English messages.
"""

def classify_intent_node(state: AgentState):
    latest_message = state["messages"][-1].content
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "{input}")
    ])
    
    chain = prompt | structured_llm
    result: IntentResult = chain.invoke({"input": latest_message})
    
    return {
        "intent": result.intent,
        "confidence_score": result.confidence_score
    }

def clarify_node(state: AgentState):
    current_count = state.get("clarify_count", 0) + 1
    
    clarification_msg = AIMessage(
        content="Could you please provide a bit more detail about the issue you are facing or your order number so I can help you better?"
    )
    
    return {
        "messages": [clarification_msg],
        "clarify_count": current_count,
        "status": "clarifying"
    }

def escalate_node(state: AgentState):
    intent = state.get("intent")
    clarify_count = state.get("clarify_count", 0)
    already_replied = state.get("status") == "auto_handled"

    if intent == "ACCOUNT_ACCESS":
        reason = "Account access issues require identity verification, which cannot be completed via Twitter."
    elif intent in {"REFUND_AND_RETURN", "WRONG_DAMAGED_ITEM"}:
        reason = f"{intent} requires order/account-level action a support bot cannot perform on Twitter."
    elif clarify_count >= 2:
        reason = "Unable to identify clear intent after 2 clarification attempts."
    else:
        reason = "Escalated per routing policy."

    if already_replied:
        handoff_msg = AIMessage(
            content="I've flagged this for our support team to take the next steps — they'll follow up with you directly."
        )
        return {
            "messages": [handoff_msg],
            "status": "escalated",
            "escalation_reason": reason
        }

    escalation_msg = AIMessage(
        content=f"I'm transferring your request to a human support agent. (Reason: {reason})"
    )
    return {
        "messages": [escalation_msg],
        "status": "escalated",
        "escalation_reason": reason
    }

def generate_reply_node(state: AgentState):
    latest_message = state["messages"][-1].content
    intent = state.get("intent", "GENERAL_INQUIRY")
    
    retrieved_docs = retriever.invoke(latest_message)
    context_examples = "\n".join([f"- {doc.metadata.get('historical_reply')}" for doc in retrieved_docs])
    
    reply_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an official Amazon Customer Support representative on Twitter.\n"
               "Draft a helpful, concise response addressing the customer issue.\n"
               "IMPORTANT: Always respond in English, regardless of what language the customer's message is written in.\n"
               "Intent: {intent}\n\n"
               "Use the following historical resolution examples as guidance for style and resolution flow:\n"
               "{context}"),
    ("user", "{input}")
    ])
    
    chain = reply_prompt | llm
    response = chain.invoke({
        "intent": intent, 
        "context": context_examples, 
        "input": latest_message
    })
    
    return {
        "messages": [response],
        "status": "auto_handled"
    }