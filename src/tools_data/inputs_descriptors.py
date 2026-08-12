INPUT_DESCRIPTORS = [
    {
       "tool_id": "get_forecast",
       "inputs": {
           "city": "string",
           "units": "metric | imperial"
       }
   },
   {
       "tool_id": "calendar_get_events",
       "inputs": {
           "start_date": "ISO date string",
           "end_date": "ISO date string"
       }
   },
   {
       "tool_id": "stock_get_price",
       "inputs": {
           "symbol": "string (stock ticker)",
           "exchange": "NYSE | NASDAQ | etc"
       }
   },
   {
       "tool_id": "calculator_compute",
       "inputs": {
           "expression": "string (mathematical expression)"
       }
   },
   {
       "tool_id": "to_upper",
       "inputs": {"text": "string"}
   },
   {
       "tool_id": "to_lower",
       "inputs": {"text": "string"},
   },
   {
       "tool_id": "reverse_string",
       "inputs": {"text": "string"},
   },
   {
       "tool_id": "word_count",
       "inputs": {"text": "string"},
   },
   {
       "tool_id": "is_palindrome",
       "inputs": {"text": "string"},
   },
   {
       "tool_id": "concat",
       "inputs": {"a": "string", "b": "string"},
   },
   {
       "tool_id": "remove_spaces",
       "inputs": {"text": "string"},
   },
   {
       "tool_id": "count_vowels",
       "inputs": {"text": "string"},
   },
   {
       "tool_id": "base64_encode",
       "inputs": {"text": "string"},
   },
   {
       "tool_id": "base64_decode",
       "inputs": {"text": "string"},
   },
   {
       "tool_id": "get_current_time",
       "inputs": {},
   },
   {
       "tool_id": "get_today_date",
       "inputs": {},
   },
   {
       "tool_id": "days_between",
       "inputs": {"date1": "YYYY-MM-DD", "date2": "YYYY-MM-DD"},
   },
   {
       "tool_id": "add_days",
       "inputs": {"date": "YYYY-MM-DD", "days": "integer"},
   },
   {
       "tool_id": "weekday_name",
       "inputs": {"date": "YYYY-MM-DD"},
   },
   {
       "tool_id": "list_sum",
       "inputs": {"numbers": "list[number]"},
   },
   {
       "tool_id": "list_max",
       "inputs": {"numbers": "list[number]"},
   },
   {
       "tool_id": "list_min",
       "inputs": {"numbers": "list[number]"},
   },
   {
       "tool_id": "list_sort",
       "inputs": {"numbers": "list[number]"},
   },
   {
       "tool_id": "unique_elements",
       "inputs": {"items": "list"},
   },
   {
       "tool_id": "merge_lists",
       "inputs": {"a": "list", "b": "list"},
   },
   {
       "tool_id": "flatten_list",
       "inputs": {"nested": "list"},
   },
   {
       "tool_id": "count_occurrences",
       "inputs": {"items": "list", "value": "any"},
   },
   {
       "tool_id": "average",
       "inputs": {"numbers": "list[number]"},
   },
   {
       "tool_id": "json_pretty",
       "inputs": {"data": "object"},
   },
   {
       "tool_id": "get_weather",
       "inputs": {"location": "string"},
   },
   {
       "tool_id": "fetch_ip_data",
       "inputs": {"url": "string"},
   },
   {
       "tool_id": "greet",
       "inputs": {"name": "string"},
   },
   {
       "tool_id": "roll_dice",
       "inputs": {},
   },
   {
       "tool_id": "temperature_c_to_f",
       "inputs": {"c": "number"},
   },
   {
       "tool_id": "temperature_f_to_c",
       "inputs": {"f": "number"},
   },
   {
       "tool_id": "area_circle",
       "inputs": {"radius": "number"},
   },
   {
       "tool_id": "perimeter_rectangle",
       "inputs": {"length": "number", "width": "number"},
   },
   {
       "tool_id": "bmi",
       "inputs": {"weight": "kg", "height": "meters"},
   },
   {
       "tool_id": "is_even",
       "inputs": {"n": "integer"},
   },
   {
       "tool_id": "is_prime",
       "inputs": {"n": "integer"},
   },
   {
       "tool_id": "random_choice",
       "inputs": {"items": "list"},
   },
   {
       "tool_id": "word_shuffle",
       "inputs": {"word": "string"},
   },
   {
       "tool_id": "percent_change",
       "inputs": {"old": "number", "new": "number"},
   },
   {
       "tool_id": "string_repeat",
       "inputs": {"text": "string", "n": "integer"},
   },
   {
       "tool_id": "circle_circumference",
       "inputs": {"radius": "number"},
   },
   {
       "tool_id": "convert_currency",
       "inputs": {
           "amount": "number",
           "from_currency": "string",
           "to_currency": "string"
       },
   },
   {
       "tool_id": "email_send",
       "inputs": {
           "to": "string (email address)",
           "subject": "string (email subject)",
           "body": "string (email body)"
       },
   },
   {
       "tool_id": "search_papers",
       "inputs": {"topic": "string"},
   },
   {
       "tool_id": "extract_info",
       "inputs": {"paper_id": "string"},
   }
]
