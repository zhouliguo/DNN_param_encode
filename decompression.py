from codec.arithmetic_decompress import read_frequencies, decompress
from codec import arithmeticcoding

# Command line main application function.
inputfile, outputfile = 'rescodea.npy', 'ori.npy'
	
# Perform file decompression
with open(outputfile, "wb") as out, open(inputfile, "rb") as inp:
	bitin = arithmeticcoding.BitInputStream(inp)
	freqs = read_frequencies(bitin)
	decompress(freqs, bitin, out)