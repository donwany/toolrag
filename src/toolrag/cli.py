"""
CLI entry point for the toolrag agent.
"""
import toolrag.langgraph_setup  # noqa: F401  # before langgraph (Reviver deprecation)

from toolrag.agent import main as agent_main

def main():
    """
    This function is the entry point for the 'toolrag' command defined in pyproject.toml.
    It simply calls the main function in the agent module.
    """
    agent_main()

if __name__ == '__main__':
    main()
