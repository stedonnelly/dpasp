import pasp

P = pasp.parse("simple.plp")

output = P(status=True, verbose=True)
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

