"""
CLI entry point for the toolrag agent.
"""
from toolrag.agent import main as agent_main

def main():
    """
    This function is the entry point for the 'toolrag' command defined in pyproject.toml.
    It simply calls the main function in the agent module.
    """
    agent_main()

if __name__ == '__main__':
    main()
