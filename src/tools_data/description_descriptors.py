FULL_TOOL_DESCRIPTORS = [
    {
       "tool_id": "get_forecast",
       "description": "Get 7-day weather forecast for a city",
       "inputs": {
           "city": "string",
           "units": "metric | imperial"
       },
       "when_to_use": "User asks about future weather conditions",
       "examples": [
           "What will the weather be in Accra next week?",
           "Is it going to rain tomorrow in NYC?"
       ]
   },
   {
       "tool_id": "calendar_get_events",
       "description": "Retrieve calendar events for a date range",
       "inputs": {
           "start_date": "ISO date string",
           "end_date": "ISO date string"
       },
       "when_to_use": "User asks about scheduled events or appointments",
       "examples": [
           "What's on my calendar this week?",
           "Do I have any meetings tomorrow?"
       ]
   },
   {
       "tool_id": "stock_get_price",
       "description": "Get current stock price and market data",
       "inputs": {
           "symbol": "string (stock ticker)",
           "exchange": "NYSE | NASDAQ | etc"
       },
       "when_to_use": "User asks about stock prices or market information",
       "examples": [
           "What's the current price of Apple stock?",
           "How is Tesla performing today?"
       ]
   },
   {
       "tool_id": "calculator_compute",
       "description": "Perform mathematical calculations",
       "inputs": {
           "expression": "string (mathematical expression)"
       },
       "when_to_use": "User asks to calculate or solve math problems",
       "examples": [
           "What's 15% of 250?",
           "Calculate the square root of 144"
       ]
   },
   {
       "tool_id": "to_upper",
       "description": "Convert text to uppercase",
       "inputs": {"text": "string"},
       "when_to_use": "User wants text transformed to uppercase",
       "examples": ["Make this text uppercase", "Convert hello to uppercase"]
   },
   {
       "tool_id": "to_lower",
       "description": "Convert text to lowercase",
       "inputs": {"text": "string"},
       "when_to_use": "User wants text transformed to lowercase",
       "examples": ["Make this lowercase", "Convert HELLO to lowercase"]
   },
   {
       "tool_id": "reverse_string",
       "description": "Reverse a string",
       "inputs": {"text": "string"},
       "when_to_use": "User wants characters reversed",
       "examples": ["Reverse this word", "Flip this string"]
   },
   {
       "tool_id": "word_count",
       "description": "Count number of words in text",
       "inputs": {"text": "string"},
       "when_to_use": "User asks how many words are in a text",
       "examples": ["How many words are here?", "Count words in this sentence"]
   },
   {
       "tool_id": "is_palindrome",
       "description": "Check if text is a palindrome",
       "inputs": {"text": "string"},
       "when_to_use": "User asks whether a word or phrase is a palindrome",
       "examples": ["Is racecar a palindrome?", "Check if this reads the same backwards"]
   },
   {
       "tool_id": "concat",
       "description": "Concatenate two strings",
       "inputs": {"a": "string", "b": "string"},
       "when_to_use": "User wants to join two strings",
       "examples": ["Join hello and world", "Concatenate two words"]
   },
   {
       "tool_id": "remove_spaces",
       "description": "Remove all spaces from text",
       "inputs": {"text": "string"},
       "when_to_use": "User wants spaces removed from text",
       "examples": ["Remove spaces from this", "Delete all spaces"]
   },
   {
       "tool_id": "count_vowels",
       "description": "Count vowels in text",
       "inputs": {"text": "string"},
       "when_to_use": "User asks how many vowels are in a string",
       "examples": ["Count vowels in education", "How many vowels are here?"]
   },
   {
       "tool_id": "base64_encode",
       "description": "Encode text using base64",
       "inputs": {"text": "string"},
       "when_to_use": "User wants to encode text",
       "examples": ["Encode this text in base64"]
   },
   {
       "tool_id": "base64_decode",
       "description": "Decode base64 encoded text",
       "inputs": {"text": "string"},
       "when_to_use": "User wants to decode base64 text",
       "examples": ["Decode this base64 string"]
   },
   {
       "tool_id": "get_current_time",
       "description": "Get the current date and time",
       "inputs": {},
       "when_to_use": "User asks for current time",
       "examples": ["What time is it now?", "Current time please"]
   },
   {
       "tool_id": "get_today_date",
       "description": "Get today's date",
       "inputs": {},
       "when_to_use": "User asks for today's date",
       "examples": ["What is today's date?"]
   },
   {
       "tool_id": "days_between",
       "description": "Calculate number of days between two dates",
       "inputs": {"date1": "YYYY-MM-DD", "date2": "YYYY-MM-DD"},
       "when_to_use": "User asks for date difference",
       "examples": ["Days between 2024-01-01 and 2024-02-01"]
   },
   {
       "tool_id": "add_days",
       "description": "Add days to a date",
       "inputs": {"date": "YYYY-MM-DD", "days": "integer"},
       "when_to_use": "User wants a future or past date",
       "examples": ["Add 10 days to 2024-01-01"]
   },
   {
       "tool_id": "weekday_name",
       "description": "Get weekday name of a date",
       "inputs": {"date": "YYYY-MM-DD"},
       "when_to_use": "User asks what day of the week a date falls on",
       "examples": ["What day is 2024-12-25?"]
   },
   {
       "tool_id": "list_sum",
       "description": "Sum values in a list",
       "inputs": {"numbers": "list[number]"},
       "when_to_use": "User wants total of numbers",
       "examples": ["Sum these numbers"]
   },
   {
       "tool_id": "list_max",
       "description": "Find maximum value in a list",
       "inputs": {"numbers": "list[number]"},
       "when_to_use": "User asks for largest number",
       "examples": ["What's the max value here?"]
   },
   {
       "tool_id": "list_min",
       "description": "Find minimum value in a list",
       "inputs": {"numbers": "list[number]"},
       "when_to_use": "User asks for smallest number",
       "examples": ["What's the minimum value?"]
   },
   {
       "tool_id": "list_sort",
       "description": "Sort list in ascending order",
       "inputs": {"numbers": "list[number]"},
       "when_to_use": "User wants sorted numbers",
       "examples": ["Sort this list"]
   },
   {
       "tool_id": "unique_elements",
       "description": "Remove duplicates from list",
       "inputs": {"items": "list"},
       "when_to_use": "User wants unique values",
       "examples": ["Remove duplicates from list"]
   },
   {
       "tool_id": "merge_lists",
       "description": "Merge two lists",
       "inputs": {"a": "list", "b": "list"},
       "when_to_use": "User wants to combine lists",
       "examples": ["Merge these lists"]
   },
   {
       "tool_id": "flatten_list",
       "description": "Flatten nested list",
       "inputs": {"nested": "list"},
       "when_to_use": "User wants nested list flattened",
       "examples": ["Flatten this list"]
   },
   {
       "tool_id": "count_occurrences",
       "description": "Count occurrences of value in list",
       "inputs": {"items": "list", "value": "any"},
       "when_to_use": "User wants frequency count",
       "examples": ["How many times does 3 appear?"]
   },
   {
       "tool_id": "average",
       "description": "Calculate average of list",
       "inputs": {"numbers": "list[number]"},
       "when_to_use": "User asks for mean value",
       "examples": ["Average these numbers"]
   },
   {
       "tool_id": "json_pretty",
       "description": "Pretty-print JSON data",
       "inputs": {"data": "object"},
       "when_to_use": "User wants formatted JSON",
       "examples": ["Pretty print this JSON"]
   },
   {
       "tool_id": "get_weather",
       "description": "Get weather for a location",
       "inputs": {"location": "string"},
       "when_to_use": "User asks about weather",
       "examples": ["Weather in Dallas,TX"]
   },
   {
       "tool_id": "fetch_ip_data",
       "description": "fetch my IP data",
       "inputs": {"url": "string"},
       "when_to_use": "User asks about their IP address",
       "examples": ["what is my IP address?"]  
   },
   {
       "tool_id": "greet",
       "description": "Generate greeting message",
       "inputs": {"name": "string"},
       "when_to_use": "User wants a greeting",
       "examples": ["Greet John"]
   },
   {
       "tool_id": "roll_dice",
       "description": "Roll a six-sided dice",
       "inputs": {},
       "when_to_use": "User wants a random dice roll",
       "examples": ["Roll a dice"]
   },
   {
       "tool_id": "temperature_c_to_f",
       "description": "Convert Celsius to Fahrenheit",
       "inputs": {"c": "number"},
       "when_to_use": "User wants temperature from Celsius to Fahrenheit conversion",
       "examples": ["Convert 30C to Fahrenheit"]
   },
   {
       "tool_id": "temperature_f_to_c",
       "description": "Convert Fahrenheit to Celsius",
       "inputs": {"f": "number"},
       "when_to_use": "User wants temperature from Fahrenheit to Celsius conversion",
       "examples": ["Convert 86F to Celsius"]
   },
   {
       "tool_id": "area_circle",
       "description": "Calculate area of a circle",
       "inputs": {"radius": "number"},
       "when_to_use": "User asks for circle area",
       "examples": ["Area of circle with radius 5"]
   },
   {
       "tool_id": "perimeter_rectangle",
       "description": "Calculate perimeter of a rectangle",
       "inputs": {"length": "number", "width": "number"},
       "when_to_use": "User asks for rectangle perimeter",
       "examples": ["Perimeter of rectangle"]
   },
   {
       "tool_id": "bmi",
       "description": "Calculate Body Mass Index",
       "inputs": {"weight": "kg", "height": "meters"},
       "when_to_use": "User asks for BMI calculation",
       "examples": ["Calculate BMI"]
   },
   {
       "tool_id": "is_even",
       "description": "Check if number is even",
       "inputs": {"n": "integer"},
       "when_to_use": "User asks if number is even",
       "examples": ["Is 10 even?"]
   },
   {
       "tool_id": "is_prime",
       "description": "Check if number is prime",
       "inputs": {"n": "integer"},
       "when_to_use": "User asks if number is prime",
       "examples": ["Is 17 prime?"]
   },
   {
       "tool_id": "random_choice",
       "description": "Select a random element from list",
       "inputs": {"items": "list"},
       "when_to_use": "User wants random selection",
       "examples": ["Pick a random item"]
   },
   {
       "tool_id": "word_shuffle",
       "description": "Shuffle letters in a word",
       "inputs": {"word": "string"},
       "when_to_use": "User wants randomized word",
       "examples": ["Shuffle this word"]
   },
   {
       "tool_id": "percent_change",
       "description": "Calculate percentage change",
       "inputs": {"old": "number", "new": "number"},
       "when_to_use": "User wants percent increase or decrease",
       "examples": ["Percent change from 50 to 75"]
   },
   {
       "tool_id": "string_repeat",
       "description": "Repeat a string multiple times",
       "inputs": {"text": "string", "n": "integer"},
       "when_to_use": "User wants repeated text",
       "examples": ["Repeat hello 5 times"]
   },
   {
       "tool_id": "circle_circumference",
       "description": "Calculate circumference of a circle",
       "inputs": {"radius": "number"},
       "when_to_use": "User asks for circle circumference",
       "examples": ["Circumference of radius 7"]
   },
   {
       "tool_id": "convert_currency",
       "description": "Convert between currencies",
       "inputs": {
           "amount": "number",
           "from_currency": "string",
           "to_currency": "string"
       },
       "when_to_use": "User wants currency conversion",
       "examples": ["Convert 100 USD to GHS"]
   },
   {
       "tool_id": "email_send",
       "description": "Send an email to a recipient",
       "inputs": {
           "to": "string (email address)",
           "subject": "string (email subject)",
           "body": "string (email body)"
       },
       "when_to_use": "User wants to send an email",
       "examples": [
           "Send an email to john@example.com",
           "Email my boss about the meeting"
       ]
   },
   {
       "tool_id": "search_papers",
       "description": "search papers",
       "inputs": {"topic": "string"},
       "when_to_use": "search papers from arxiv",
       "examples": ["search papers on the topic:llm"]
   },
   {
       "tool_id": "extract_info",
       "description": "extract information on search papers",
       "inputs": {"paper_id": "string"},
       "when_to_use": "extract paper information with an id",
       "examples": ["extract paper information on the given a paper id"]
   }
]
