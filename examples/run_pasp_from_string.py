import pasp

with open("simple.plp", "r") as f:
    plp_string = f.read()

P = pasp.parse(plp_string, from_str=True)

output = P(verbose=True) 
"""
Use

P(quiet=True, verbose=True) 

if you want to suppress the proabilities being printed to the terminal.
"""

# The result has a built-in string representation
result_str = str(output)
print(result_str)

# You can also access the raw numpy data
print(output.data)
