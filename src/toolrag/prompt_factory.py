# reasoning_prompt = f"""
# You are analyzing a user query to select the most appropriate tool.

# User Query: "{user_query}"

# Available Tools:
# {tools_text}

# IMPORTANT: You must select ONE of these exact tool IDs: {valid_tool_ids}

# Analyze the query and determine which tool best matches the user's intent.
# Provide a confidence score (0.0-1.0) based on how well the tool matches the query.
# 1.0 = perfect match, 0.5 = moderate match, 0.0 = no match

# Respond with ONLY a JSON object:
# {{
#     "reasoning": "your reasoning here",
#     "selected_tool_id": "MUST BE ONE OF: {', '.join(valid_tool_ids)}",
#     "confidence": 0.85
# }}
# """


from langchain.prompts import ChatPromptTemplate

tool_selection_prompt = ChatPromptTemplate.from_messages([
    (
    "system",
        """You are an expert AI assistant responsible for selecting the most appropriate tool based on a user's query.

    You MUST follow these rules:
    - Select exactly ONE tool
    - The selected tool ID MUST be one of the provided valid tool IDs
    - Provide a confidence score between 0.0 and 1.0
    - Respond ONLY with a valid JSON object
"""
    ),
    (
    "human",
        """User Query:
    "{user_query}"

    Available Tools:
    {tools_text}

    Valid Tool IDs:
    {valid_tool_ids}

    Analyze the query and determine which tool best matches the user's intent.

    Output format (JSON ONLY):
    {{
        "reasoning": "your reasoning here",
        "selected_tool_id": "one of the valid tool IDs",
        "confidence": 0.0
    }}
"""
    )
])



param_extraction_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert AI assistant responsible for extracting structured parameters
for a tool invocation.

Rules:
- Extract ONLY parameters explicitly required by the tool
- Use values inferred directly from the user query when possible
- Do NOT invent parameters or values
- Respond ONLY with a valid JSON object
"""
    ),
    (
        "human",
        """User Query:
"{user_query}"

Tool ID:
{tool_id}

Required Inputs (JSON Schema):
{tool_inputs}

Extract the parameter values for this tool call.

Output format (JSON ONLY):
Example:
{{"location": "Accra", "days": 3}}
"""
    )
])
