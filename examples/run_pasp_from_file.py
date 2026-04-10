import pasp

P = pasp.parse("simple.plp")

output = P(status=True, verbose=True)

# The result has a built-in string representation
result_str = str(output)
print(result_str)

# You can also access the raw numpy data
print(output.data)

