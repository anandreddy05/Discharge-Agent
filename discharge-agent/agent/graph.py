from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from schema.agent_state import AgentState
from agent.nodes import reasoner, validator
from agent.tools import AGENT_TOOLS

tool_node = ToolNode(AGENT_TOOLS)


def router(state: AgentState) -> str:
    if state.get("iteration_count", 0) >= 15:
        return END

    # If the Validator passed it, we are done!
    if state.get("validator_critique") == "PASS":
        return END

    last_message = state["messages"][-1]
    if last_message.tool_calls:
        for tc in last_message.tool_calls:
            if tc["name"] == "DischargeSummaryDraft":
                # Don't end! Send it to the validator to be graded.
                return "validator"
            else:
                return "tools"

    return END


def build_graph():

    workflow = StateGraph(AgentState)
    workflow.add_node("reasoner", reasoner)
    workflow.add_node("tools", tool_node)
    workflow.add_node("validator", validator)

    workflow.set_entry_point("reasoner")

    workflow.add_conditional_edges(
        "reasoner", router, {"tools": "tools", "validator": "validator", END: END}
    )
    workflow.add_conditional_edges(
        "validator",
        lambda s: END if s.get("validator_critique") == "PASS" else "reasoner",
        {"reasoner": "reasoner", END: END},
    )
    workflow.add_edge("tools", "reasoner")

    return workflow.compile()
