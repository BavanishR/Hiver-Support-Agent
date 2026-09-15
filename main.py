from langchain_core.messages import HumanMessage
from graph import app

if __name__ == "__main__":
    print("=== Amazon Twitter Support Agent (Interactive CLI) ===")
    print("Type 'exit' to quit.\n")

    state = {"messages": [], "clarify_count": 0}

    while True:
        user_input = input("\nCustomer: ").strip()
        if not user_input or user_input.lower() == "exit":
            break

        state["messages"].append(HumanMessage(content=user_input))

        state = app.invoke(state)

        status = state.get("status")
        agent_reply = state["messages"][-1].content
        
        print(f"\n[Intent: {state.get('intent')} | Confidence: {state.get('confidence_score')} | Clarify Count: {state.get('clarify_count', 0)}]")
        print(f"Agent ({status}): {agent_reply}")

        if status in ["escalated", "auto_handled"]:
            print("\n--- Interaction Finished. Starting new session ---")
            state = {"messages": [], "clarify_count": 0}