import arithmeticcoding

# Returns a frequency table based on the bytes in the given file.
# Also contains an extra entry for symbol 256, whose frequency is set to 0.

def get_frequencies(file_or_array):
	freqs = arithmeticcoding.SimpleFrequencyTable([0] * 257)

	if isinstance(file_or_array, basestring):
		with open(file_or_array, "rb") as inp:
			while True:
				b = inp.read(1)
				if len(b) == 0:
					break
				freqs.increment(b[0])
	else:
		freqs.increment_array(file_or_array)
		for b in file_or_array:
			freqs.increment(b)
	return freqs


def write_frequencies(bitout, freqs):
	for i in range(256):
		write_int(bitout, 32, freqs.get(i))


def compress(freqs, file_or_array, bitout):
	enc = arithmeticcoding.ArithmeticEncoder(32, bitout)

	if isinstance(file_or_array, basestring):
		with open(file_or_array, "rb") as inp:
			while True:
				symbol = inp.read(1)
				if len(symbol) == 0:
					break
				enc.write(freqs, symbol[0])
	else:
		for symbol in file_or_array:
			enc.write(freqs, symbol)

	enc.write(freqs, 256)  # EOF
	enc.finish()  # Flush remaining code bits


# Writes an unsigned integer of the given bit width to the given stream.
def write_int(bitout, numbits, value):
	for i in reversed(range(numbits)):
		bitout.write((value >> i) & 1)  # Big endian
